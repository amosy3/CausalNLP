import pandas as pd
from transformers import Trainer, TrainingArguments
import torch
from torch import nn
import argparse
from utils import Logger, save_object, load_object
from backbones import LanguageModel, get_model
from cf_datasets import get_tokenized_datasets
from itertools import product
from training import CustomTrainer, compute_metrics
import torch.nn.functional as F
# import sys
# sys.path.append('..')
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataloaders import tokenize_text, TokenizedDataset
from tcav import *
import prettytable as pt
import h5py
import numpy as np
import os
from glob import glob
from concept_shap import *

from approx import get_logits_from_text, get_cf_sample

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",type=str, choices=['gpt2', 'deberta', 'qwen','t5'], default='gpt2')
    parser.add_argument("--method",type=str, choices=['approx', 'train_model', 'tcav', 'ConceptShap'], default='approx')
    parser.add_argument("--log_file", type=str, default='')
    parser.add_argument("--dataset", choices=['cv', 'disease', 'violence'], default='cv')
    parser.add_argument("--use_gpu", type=str, default='0')
    parser.add_argument("--batch_size", type=int, default='2')
    parser.add_argument("--seed", type=int, default='42')
    parser.add_argument("--load_from_dir",type=str, default='../logs/cv_tcav_gpt2/test_logs_2025-03-23_13-03-33')
    parser.add_argument("--debug", action='store_true', default=False)



    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    return args

def get_logits_from_text(model, tokenizer, sentences):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    # Get input_ids and attention mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    return outputs.logits

def get_cf_sample(concepts, current_concept, full_df, org_smaple):
    keep_concepts = [z for z in concepts if z != current_concept]
    diff_concepts = [current_concept] #['Candidate_id', c]
    matching_rows = full_df[(full_df[keep_concepts] == org_smaple[keep_concepts].values).all(axis=1)]
    valid_cf = matching_rows[(matching_rows[diff_concepts] != org_smaple[diff_concepts].values).all(axis=1)]
    if valid_cf.shape[0] == 0.0:
        print('No counterfactuals for %s->%s !!!!' % (org_concept, do_concept))
        return full_df.sample()
    else:
        return valid_cf.sample()


# def t5_get_class_probs(output, tokenizer, class_mapping):
#     """
#     Extract class probabilities from a T5 generate() output,
#     assuming labels are single-token words.

#     Args:
#         output: the result of model.generate(..., return_dict_in_generate=True, output_scores=True)
#         tokenizer: the tokenizer used with the T5 model
#         class_mapping: dict mapping class names to indices, e.g. {"Regular": 0, "Good": 1, "Exceptional": 2}

#     Returns:
#         predicted_probs: Tensor of shape [batch_size, num_classes]
#                          Each row corresponds to probabilities in the class index order.
#     """

#     # Validate and get token ID for each class label
#     label_token_ids = {}
#     for label in class_mapping:
#         token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
#         # if len(token_ids) != 1:
#         #     raise ValueError(f"Label '{label}' must be a single token. Got tokens: {token_ids}")
#         print(label, token_ids)
#         label_token_ids[label] = token_ids[0]

#     # Get logits of the first generated token
#     logits = output.scores[0]  # shape: (batch_size, vocab_size)
#     probs = F.softmax(logits, dim=-1)  # shape: (batch_size, vocab_size)

#     batch_size = logits.size(0)
#     num_classes = len(class_mapping)
#     predicted_probs = torch.zeros(batch_size, num_classes, device=logits.device)

#     # Fill probability tensor according to class index order
#     for label, class_idx in class_mapping.items():
#         token_id = label_token_ids[label]
#         predicted_probs[:, class_idx] = probs[:, token_id]

#     return predicted_probs


def t5_get_class_probs(output, tokenizer, class_mapping):
    """
    Extract class probabilities from a T5 generate() output.
    Supports:
      - class labels that tokenize into multiple tokens (uses only first token)
      - empty string ("") as a label (interpreted as generating nothing)

    Args:
        output: result of model.generate(..., return_dict_in_generate=True, output_scores=True)
        tokenizer: tokenizer used with the T5 model
        class_mapping: dict like {"": 0, "1": 1, "2": 2}

    Returns:
        predicted_probs: Tensor [batch_size, num_classes]
    """
    batch_size = output.sequences.shape[0]
    vocab_size = output.scores[0].shape[1]
    num_classes = len(class_mapping)
    predicted_probs = torch.zeros(batch_size, num_classes, device=output.scores[0].device)

    # Tokenize class labels (use only the first token for each label)
    label_token_ids = {}
    for label in class_mapping:
        token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        label_token_ids[label] = token_ids[0] if token_ids else None  # None means empty string

    # First token logits from generated output
    logits = output.scores[0]  # shape: (batch_size, vocab_size)
    probs = F.softmax(logits, dim=-1)  # shape: (batch_size, vocab_size)

    for label, class_idx in class_mapping.items():
        token_id = label_token_ids[label]
        if token_id is None:
            # Handle empty string: check if decoded output is empty
            for i in range(batch_size):
                decoded = tokenizer.decode(output.sequences[i], skip_special_tokens=True).strip()
                predicted_probs[i, class_idx] = 1.0 if decoded == "" else 0.0
        else:
            predicted_probs[:, class_idx] = probs[:, token_id]

    return predicted_probs


if __name__ == "__main__":
    args = get_args()

    print("Running! We now use %s for %s on %s dataset" % (args.backbone, args.method, args.dataset))

    logger = Logger(path='../logs/%s_%s_%s' % (args.dataset, args.method, args.backbone), filename = args.log_file)
    logger.log_object(args, 'args')
    device = "cuda:%s" % args.use_gpu

    model, tokenizer = get_model(args.backbone, dataset=args.dataset)

    if args.method == "train_model":
        
        train_dataset, val_dataset = get_tokenized_datasets(tokenizer)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='%s/results' % logger.log_dir,  # Output directory for model checkpoints
            logging_strategy="steps",         # Log metrics every few steps
            logging_steps=5,
            eval_strategy="epoch",  # Evaluation strategy
            learning_rate=2e-5,  # Learning rate
            per_device_train_batch_size=5,  # Batch size for training
            per_device_eval_batch_size=5,  # Batch size for evaluation
            num_train_epochs=100,  # Number of training epochs
            weight_decay=0.01,  # Weight decay
            logging_dir='%s/trainner_logs'% logger.log_dir,  # Directory for logging
            fp16=True,
        )

        # Prepare Trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Fine-tune the model
        trainer.train()

        # Evaluate the model
        trainer.evaluate()
        exit()

    # load model weights
    model, tokenizer = get_model(args.backbone, dataset=args.dataset, ckpt='../pretrain_models/%s' % args.backbone)
    model.eval()

    # load data
    data_path = "../datasets/%s" % args.dataset
    df_train = pd.read_csv('%s/train.csv' % data_path)
    df_test = pd.read_csv('%s/test.csv' % data_path)
    df_estimate_cf = pd.read_csv('%s/wo_f.csv' % data_path)
    df_cf = pd.read_csv('%s/w_cf.csv' % data_path)

    if args.dataset == 'cv':
        concepts = ['Gender', 'Education', 'Socioeconomic_Status', 'Age_group', 'Certificates', 'Volunteering', 'Race', 'Work_Experience_group']
        text = 'CV_statement'
        target = 'Good_Employee'
        class_mapping = {"Regular": 0, "Good": 1, "Exceptional": 2}
        num_classes = 3
            
    
    elif args.dataset == 'disease':
        concepts = ['Dizzy', 'Sensitivity_to_Light','Headache','Nasal_Congestion', 'Facial_Pain_Pressure','Fever','General_Weakness']
        text = 'Patient_consultation'
        target = 'Golden_Label'
        class_mapping = {"": 0, "1": 1, "2": 2}
        num_classes = 3


    elif  args.dataset == 'violence':
        concepts = ['Gender', 'Age_group', 'Race', 'Years_As_Nurse', 'License_Type', 'Department','Activity_At_Work']
        text = 'Dialogue'
        target = 'Violence'
        class_mapping = {"No": 0, "Verb": 1, "Physical": 2}
        num_classes = 3

    else:
        raise ValueError('Could not find dataset info')


    if args.method == "approx":
        metrics = {}    
        # Eval ICaCE
        
        df = pd.read_csv('../datasets/cv_w_cf.csv')
        df_org = df[df['CF_on'] == 'ORG']
        
        for current_concept in concepts:
            print("Concept = %s" % current_concept)
            posible_values = df_org[current_concept].unique()
            posible_cf = [p for p in product(posible_values, repeat=2)]
            for (org_concept, do_concept) in posible_cf:
                if org_concept == do_concept:
                    continue

                
                print("Estimating the effect of changing %s to %s" % (org_concept, do_concept))
                metrics[(current_concept, org_concept, do_concept)] = {'l2':[], 'cosine':[], 'norm_diff':[]}

                mask_org = df_org[current_concept] == org_concept
                cdf = df_org[mask_org]
                for i in range(3):
                    x_org = cdf.sample()
                    preds = get_logits_from_text(model, tokenizer, x_org[text].tolist())

                    x_cf = get_cf_sample(concepts, current_concept, cdf, x_org)
                    cf_preds = get_logits_from_text(model, tokenizer, x_cf[text].tolist(), allways_return=True)

                    l2 = torch.norm(preds - cf_preds, p=2)
                    metrics[(current_concept, org_concept, do_concept)]['l2'].append(l2.item())

                    cosine_similarity = torch.nn.functional.cosine_similarity(preds, cf_preds)
                    metrics[(current_concept, org_concept, do_concept)]['cosine'].append(1 - cosine_similarity.item())
                    
                    norm_diff = abs(torch.norm(preds) - torch.norm(cf_preds))
                    metrics[(current_concept ,org_concept, do_concept)]['norm_diff'].append(norm_diff.item())
        
    if args.method == "tcav":
        # Organize all dataloaders
        train_dataset = TokenizedDataset(df_train, tokenizer, text, target, args.dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        estimate_cf_dataset = TokenizedDataset(df_estimate_cf, tokenizer, text, target, args.dataset)
        estimate_cf_loader = DataLoader(estimate_cf_dataset, batch_size=1, shuffle=True)

        concept_dict = {}
        for c in concepts:
            c_vals = df_train[c].unique()
            for v in c_vals:
                concept_dataset = TokenizedDataset(df_train[df_train[c] == v], tokenizer, text, target, args.dataset)
                concept_dict['%s,%s' % (c,v)] = DataLoader(concept_dataset, batch_size=args.batch_size, shuffle=True)
        

        model = model.to(device)
        #pass the name of the desired embedding layer. will use the first element of the layer.
        if args.backbone == 'gpt2':
            extract_layer = 'transformer' 
        elif args.backbone == 't5':
            extract_layer = 'encoder.block.11' 
        elif args.backbone == 'qwen':
            extract_layer = 'transformer.h.31' 
        elif args.backbone == 'deberta':
            extract_layer = 'deberta.encoder.layer.11' 
        wmodel = ModelWrapper(model, [extract_layer], backbone=args.backbone)
        wmodel = wmodel.to(device)
        wmodel.eval()
        
        scorer = TCAV(wmodel, estimate_cf_loader, concept_dict, df_train[target].unique(), 2, device, logger.log_dir)
        print('Generating concepts...')
        scorer.generate_activations([extract_layer])
        scorer.load_activations()
        print('Concepts successfully generated and loaded!')
        
        print('Calculating TCAV scores...')
        scorer.generate_cavs(extract_layer)
        # For now ther is an exit command because all gradients are zero!!!
        scorer.calculate_tcav_score(extract_layer, '%s/tcav_result.npy' % logger.log_dir)
        scores = np.load('%s/tcav_result.npy' % logger.log_dir)
        scores = scores.T.tolist()
    
        table = pt.PrettyTable()
        class_dict = {
            'Bad': 0,
            'Mid': 1,
            'Good': 2
        }
        table.field_names = ['class'] + list(concept_dict.keys())
        for i, k in enumerate(class_dict.keys()):
            new_row = [k] + scores[i]
            table.add_row(new_row)
        print(table)
        metrics = scores
    # logger.log_object(metrics)
                        
    if args.method == "ConceptShap":

        train_dataset = TokenizedDataset(df_train, tokenizer, text, target, args.dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        estimate_cf_dataset = TokenizedDataset(df_estimate_cf, tokenizer, text, target, args.dataset)
        estimate_cf_loader = DataLoader(estimate_cf_dataset, batch_size=args.batch_size, shuffle=True)

        logfile_location = '../logs/%s_tcav_%s' % (args.dataset, args.backbone)
        cavs = load_object('%s/%s/embeddings/cavs.pkl' % (logfile_location, args.load_from_dir))
        cavs_names = load_object('%s/%s/embeddings/cavs_names.pkl' % (logfile_location, args.load_from_dir))

        # Create concept space for projection        
        concepts_matrix = torch.from_numpy(cavs)
        concepts_matrix = concepts_matrix.T
        proj_matrix = (concepts_matrix @ torch.inverse((concepts_matrix.T @ concepts_matrix))) \
                      @ concepts_matrix.T

        proj_matrix = proj_matrix.to(device)

        
        # Optimize prediction from the concept space
        g = G(768,768,768)
        g = g.to(device)
        
        if args.backbone == 't5':
            h_x = torch.nn.Linear(proj_matrix.shape[0], num_classes).to(device)
            optimizer = torch.optim.Adam(list(g.parameters()) + list(h_x.parameters()), lr=0.0001)
        elif args.backbone in ('gpt2', 'qwen'):
            h_x = list(model.modules())[-1]
            optimizer = torch.optim.Adam(g.parameters(), lr=0.0001)
        elif args.backbone == 'deberta':
            h_x = model.classifier
            optimizer = torch.optim.Adam(g.parameters(), lr=0.0001)
        else:
            raise Exception("Backbone does not recognized!")

        model = model.to(device)        
        model.float()
        
        epochs = 1 if args.debug else 10
        for i in range(epochs):
            running_loss = 0.0
            for X, y in tqdm(train_loader):
                X = X.to(device)


                if args.backbone == 't5':
                    with torch.no_grad():
                        encoder_outputs = model.encoder(
                            input_ids=X['input_ids'],
                            attention_mask=X.get('attention_mask'),
                            output_hidden_states=True,
                            return_dict=True
                        )
                    X_embedding = encoder_outputs[0][:,0,:] #Represent samples as embeddings - (batch_size x embedding_dim) = (2,768)
                elif args.backbone in ('gpt2', 'qwen', 'deberta'):
                    with torch.no_grad():
                        if 'attention_mask' in X and X['attention_mask'].dim() == 3:
                            X['attention_mask'] = X['attention_mask'].squeeze(1)
                        outputs = model(**X, output_hidden_states=True)
                    X_embedding = outputs.hidden_states[-1][:,0,:] #Represent samples as embeddings - (batch_size x embedding_dim) = (2,768)

                x = proj_matrix @ X_embedding.T #progect embedding to concept space - (embedding_dim x batch_size) = (768,2)
                x = g(x.T) #learn a mapping (g) that maximize the performance - (batch_size x embedding_dim)
                pred = h_x(x) #feed to original last layer - (batch_size x 3) = (2,3)

                y = y.to(torch.long)
                y = F.one_hot(y, num_classes=num_classes).to(device).float()
                loss = F.binary_cross_entropy_with_logits(pred, y)
            
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print('Epoch: %s Cross entropy loss = %s' % (i,running_loss))    
            
        concept2idx = dict()
        for c in concepts:
            concept2idx[c] = [i for (i,n) in enumerate(cavs_names) if c in n]



        # Calculate mebeddings ounce to reduce computation inside the shaply values iterations
        cached_embeddings = []
        cached_probs = []


        for X, y in estimate_cf_loader:
            X, y = X.to(device), y.to(device)
            proj_matrix = proj_matrix.to(device)

            if args.backbone == 't5':
                with torch.no_grad():
                    encoder_outputs = model.encoder(
                        input_ids=X['input_ids'],
                        attention_mask=X.get('attention_mask'),
                        output_hidden_states=True,
                        return_dict=True
                    )
                X_embedding = encoder_outputs[0][:,0,:]
                
                with torch.no_grad():
                    output = model.generate(input_ids=X['input_ids'],
                            attention_mask=X.get('attention_mask').squeeze(1),
                            return_dict_in_generate=True,
                            output_scores=True,
                            max_length=10)
                    
                    predicted_probs = t5_get_class_probs(output, tokenizer, class_mapping)
                    
            elif args.backbone in ('gpt2', 'qwen', 'deberta'):
                with torch.no_grad():
                    if 'attention_mask' in X and X['attention_mask'].dim() == 3:
                            X['attention_mask'] = X['attention_mask'].squeeze(1)
                    outputs = model(**X, output_hidden_states=True)
                X_embedding = outputs.hidden_states[-1][:,0,:] #Represent samples as embeddings - (batch_size x embedding_dim) = (2,768)
                predicted_probs = outputs['logits']
            else:
                raise Exception("Backbone does not recognized!")

            cached_embeddings.append(X_embedding)
            cached_probs.append(predicted_probs)


        for tested_concept in tqdm(concepts):
            print(tested_concept)
            exclude = [x for x in concepts if x != tested_concept]
            subsets = list(powerset(list(exclude)))
            concept_sum = 0
            for subset in subsets[3:]:
                # score 1:
                concept_name_list = subset + [tested_concept]
                nested_indices = [concept2idx[concept_name] for concept_name in concept_name_list]
                indices = sorted(list(chain(*nested_indices)))
                index_tensor = torch.tensor(indices, dtype=torch.long)
                filtered_concepts_matrix = concepts_matrix[:,index_tensor]
                proj_matrix = (filtered_concepts_matrix @ torch.inverse((filtered_concepts_matrix.T @ filtered_concepts_matrix))) \
                        @ filtered_concepts_matrix.T
                
                y_gt, y_pred, y_prec_from_concepts = [], [], []
                for X_embedding, predicted_probs, (X, y) in zip(cached_embeddings, cached_probs, estimate_cf_loader):
                    X_embedding, predicted_probs = X_embedding.to(device), predicted_probs.to(device)
                    proj_matrix = proj_matrix.to(device)
                    y = y.to(device)

                    x = proj_matrix @ X_embedding.T #progect embedding to concept space - (embedding_dim x batch_size) = (768,2)
                    x = g(x.T) #learn a mapping (g) that maximize the performance - (batch_size x embedding_dim)
                    pred = h_x(x) #feed to original last layer - (batch_size x 3) = (2,3)

                    
                    y_gt += y
                    y_pred += torch.argmax(predicted_probs, axis=1)
                    y_prec_from_concepts += torch.argmax(pred, axis=1)
                    
                y_gt, y_pred, y_prec_from_concepts = torch.stack(y_gt).int(), torch.stack(y_pred).int(), torch.stack(y_prec_from_concepts).int()
                score1 = n(y_gt, y_pred, y_prec_from_concepts, predicted_probs.shape[1])


                # score 2:
                concept_name_list = subset
                if concept_name_list == []:
                    score2 = torch.tensor(0)
                else:
                    nested_indices = [concept2idx[concept_name] for concept_name in concept_name_list]
                    indices = sorted(list(chain(*nested_indices)))
                    index_tensor = torch.tensor(indices, dtype=torch.long)
                    filtered_concepts_matrix = concepts_matrix[:,index_tensor]
                    proj_matrix = (filtered_concepts_matrix @ torch.inverse((filtered_concepts_matrix.T @ filtered_concepts_matrix))) \
                            @ filtered_concepts_matrix.T
                    
                    y_gt, y_pred, y_prec_from_concepts = [], [], []
                    for X_embedding, predicted_probs, (X, y) in zip(cached_embeddings, cached_probs, estimate_cf_loader):
                        X_embedding, predicted_probs = X_embedding.to(device), predicted_probs.to(device)
                        proj_matrix = proj_matrix.to(device)
                        y = y.to(device)

                        x = proj_matrix @ X_embedding.T #progect embedding to concept space - (embedding_dim x batch_size) = (768,2)
                        x = g(x.T) #learn a mapping (g) that maximize the performance - (batch_size x embedding_dim)
                        pred = h_x(x) #feed to original last layer - (batch_size x 3) = (2,3)
            
                        
                        y_gt += y
                        y_pred += torch.argmax(predicted_probs, axis=1)
                        y_prec_from_concepts += torch.argmax(pred, axis=1)
                        
                    
                    y_gt, y_pred, y_prec_from_concepts = torch.stack(y_gt).int(), torch.stack(y_pred).int(), torch.stack(y_prec_from_concepts).int()
                    score2 = n(y_gt, y_pred, y_prec_from_concepts, predicted_probs.shape[1])



                norm = (math.factorial(len(concepts) - len(subset) - 1) * math.factorial(len(subset))) / \
                                math.factorial(len(concepts))
                concept_sum += norm * (score1.data.item() - score2.data.item())
            logger.print_and_log('%s : %s' % (tested_concept, concept_sum))
    

    # logger.log_object(metrics)





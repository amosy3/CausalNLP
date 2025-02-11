import pandas as pd
from transformers import Trainer, TrainingArguments
import torch
from torch import nn
import argparse
from utils import Logger
from backbones import LanguageModel, get_model
from cf_datasets import get_tokenized_datasets
from itertools import product
from training import CustomTrainer, compute_metrics
import torch.nn.functional as F
# import sys
# sys.path.append('..')
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from tqdm import tqdm
from dataloaders import tokenize_text, TokenizedDataset
from tcav import *
import prettytable as pt



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",type=str, choices=['gpt2', 'deberta'], default='gpt2')
    parser.add_argument("--method",type=str, choices=['approx', 'train_model', 'tcav'], default='approx')
    parser.add_argument("--log_file", type=str, default='')
    parser.add_argument("--use_gpu", type=str, default='0')
    parser.add_argument("--batch_size", type=int, default='2')


    args = parser.parse_args()
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


if __name__ == "__main__":
    args = get_args()

    print("Running! We now us %s for %s" % (args.backbone, args.method))

    logger = Logger(path='../logs/%s_%s' % (args.method, args.backbone), filename = args.log_file)
    logger.log_object(args, 'args')
    device = "cuda:%s" % args.use_gpu

    model, tokenizer = get_model(args.backbone)

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
    model, tokenizer = get_model(args.backbone, ckpt='../pretrain_models/%s' % args.backbone)
    model.eval()

    # load data
    data_path = "../datasets/"
    df_train = pd.read_csv('%s/train.csv' % data_path)
    df_test = pd.read_csv('%s/test.csv' % data_path)
    df_estimate_cf = pd.read_csv('%s/estimate_cf.csv' % data_path)
    df_cf = pd.read_csv('%s/cf.csv' % data_path)

    concepts = ['Gender', 'Education', 'Socioeconomic_Status', 'Age_group', 'Certificates', 'Volunteering', 'Race', 'Work_Experience_group']
    text = 'CV_statement'

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
                    cf_preds = get_logits_from_text(model, tokenizer, x_cf[text].tolist())

                    l2 = torch.norm(preds - cf_preds, p=2)
                    metrics[(current_concept, org_concept, do_concept)]['l2'].append(l2.item())

                    cosine_similarity = torch.nn.functional.cosine_similarity(preds, cf_preds)
                    metrics[(current_concept, org_concept, do_concept)]['cosine'].append(1 - cosine_similarity.item())
                    
                    norm_diff = abs(torch.norm(preds) - torch.norm(cf_preds))
                    metrics[(current_concept ,org_concept, do_concept)]['norm_diff'].append(norm_diff.item())
        
    if args.method == "tcav":
        # Organize all dataloaders
        train_dataset = TokenizedDataset(df_train, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        estimate_cf_dataset = TokenizedDataset(df_estimate_cf, tokenizer)
        estimate_cf_loader = DataLoader(estimate_cf_dataset, batch_size=1, shuffle=True)

        concept_dict = {}
        for c in concepts:
            c_vals = df_train[c].unique()
            for v in c_vals:
                concept_dataset = TokenizedDataset(df_train[df_train[c] == v], tokenizer)
                concept_dict['%s,%s' % (c,v)] = DataLoader(concept_dataset, batch_size=args.batch_size, shuffle=True)
        

        model = model.to(device)
        extract_layer = 'transformer' #pass the name of the desired embedding layer. will use the first element of the layer.
        wmodel = ModelWrapper(model, [extract_layer])
        wmodel = wmodel.to(device)
        wmodel.eval()
        
        scorer = TCAV(wmodel, estimate_cf_loader, concept_dict, df_train["Good_Employee"].unique(), 2, device)
        print('Generating concepts...')
        scorer.generate_activations([extract_layer])
        scorer.load_activations()
        print('Concepts successfully generated and loaded!')
        
        print('Calculating TCAV scores...')
        scorer.generate_cavs(extract_layer)
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
    
    
    
    
    logger.log_object(metrics)
                        
                    




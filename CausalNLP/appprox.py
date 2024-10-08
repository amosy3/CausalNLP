import pandas as pd
from transformers import Trainer, TrainingArguments
import torch
from torch import nn
import argparse
from utils import Logger
from backbones import LanguageModel
from itertools import product

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",type=str, choices=['gpt2', 'deberta'], default='gpt2')
    parser.add_argument("--method",type=str, choices=['approx', 'train_model'], default='approx')
    parser.add_argument("--log_file", type=str, default='')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    logger = Logger(filename='../%s_%s/%s' % (args.method, args.backbone, args.log_file, ))
    logger.log_object(args, 'args')


    lm = LanguageModel()
    # df = pd.read_csv('../datasets/cv_w_cf.csv')

    if args.method == "train_model":
        # Create train and test loader
        # org_samples = df[df["CF_to"]=='ORG']
        # X, y = org_samples["CV_statement"][:1].tolist(), org_samples["Good_Employee"][:1].tolist()

        # Define training arguments
        training_args = TrainingArguments(
            output_dir='%s/results' % logger.log_dir,  # Output directory for model checkpoints
            evaluation_strategy="epoch",  # Evaluation strategy
            learning_rate=2e-5,  # Learning rate
            per_device_train_batch_size=8,  # Batch size for training
            per_device_eval_batch_size=8,  # Batch size for evaluation
            num_train_epochs=3,  # Number of training epochs
            weight_decay=0.01,  # Weight decay
            logging_dir='%s/trainner_logs'% logger.log_dir,  # Directory for logging
        )

        # Prepare Trainer
        trainer = Trainer(
            model=lm,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test'],
        )

        # Fine-tune the model
        trainer.train()

        # Evaluate the model
        trainer.evaluate()

    if args.method == "approx":
        # load model weights
        # Eval ICaCE
        concepts = ['Gender', 'Education', 'Socioeconomic_Status', 'Age_group', 'Certificates', 'Volunteering', 'Race',
                    'Work_Experience_group']
        df = pd.read_csv('../datasets/cv_w_cf.csv')
        df_org = df[df['CF_on'] == 'ORG']
        for c in concepts:
            print("Concept = %s" % c)
            posible_values = df_org[c].unique()
            posible_cf = [p for p in product(posible_values, repeat=2)]
            for (cf, ct) in posible_cf:
                if cf == ct:
                    continue
                print(cf, ct)
                mask = df_org[c] == cf
                x = df_org[mask].sample()
                keep_concepts = [z for z in concepts if z != c]
                change_concepts = ['Candidate_id', c]
                matching_rows = df_org[(df_org[keep_concepts] == x[keep_concepts].values).all(axis=1)]
                valid_cf = matching_rows[
                    (matching_rows[change_concepts] != matching_rows[change_concepts].values).all(axis=1)]
                if valid_cf.shape[0] == 0.0:
                    print('No counterfactuals for %s->%s' % (cf, ct))
                else:
                    x_cf = valid_cf.sample()
                    # calc the diff

            # for i, row in df_org.iterrows():

        # Testing
    X = df["CV_statement"][:1].tolist()
    y = df["Good_Employee"][:1].tolist()
    features = lm.extract_features(X)
    print(features["logits"])




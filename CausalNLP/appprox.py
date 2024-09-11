import pandas as pd
from transformers import Trainer, TrainingArguments
import torch
from torch import nn
import argparse
from utils import Logger
from backbones import LanguageModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",type=str, default='gpt2')
    parser.add_argument("--method",type=str, choices=['approx', 'train_model'], default='approx')
    parser.add_argument("--log_file", type=str, default='')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    logger = Logger(filename='../%s_%s/%s' % (args.method, args.backbone, args.log_file, ))
    logger.log_object(args, 'args')


    lm = LanguageModel()
    df = pd.read_csv('../datasets/cv_w_cf.csv')

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


    # Testing
    X = df["CV_statement"][:1].tolist()
    y = df["Good_Employee"][:1].tolist()
    features = lm.extract_features(X)
    print(features["logits"])




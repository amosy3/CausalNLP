import os

import numpy as np
import pandas as pd
import wandb
from transformers import DebertaTokenizer, DebertaForSequenceClassification, DebertaConfig, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_cv_statement_classifier(train_df, val_df, test_df, mode="regular", des_path="Trained_models/cv_statements_model"):
    # Initialize wandb
    if mode != "regular":
        print("Overfitting mode activated.")
        des_path = "Trained_models/cv_statement_model_overfit"
    else:
        print("Regular training mode activated.")

    wandb.init(project='cv_project', entity='gilat')

    X_train = train_df['CV_statement']
    y_train = train_df['Good_Employee']
    X_val = val_df['CV_statement']
    y_val = val_df['Good_Employee']
    X_test = test_df['CV_statement']
    y_test = test_df['Good_Employee']

    model_name = "microsoft/deberta-base"
    tokenizer = DebertaTokenizer.from_pretrained(model_name)
    config = DebertaConfig.from_pretrained(model_name)
    config.num_labels = 3
    config.output_hidden_states = True
    model = DebertaForSequenceClassification.from_pretrained(model_name, config=config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print("Training on:", device)

    # Tokenization and dataset preparation
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(X_val.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)

    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': y_train.tolist()
    })
    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': y_val.tolist()
    })

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        report_to=['wandb'],
        num_train_epochs=20 if mode == "overfit" else 5,
        evaluation_strategy="epoch",
        eval_steps=70,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        lr_scheduler_type='linear',
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: compute_metrics(pred)
    )

    # Train and save the model
    trainer.train()
    model.save_pretrained(des_path)
    tokenizer.save_pretrained(des_path)

    # Evaluate the model on the test set
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': y_test.tolist()
    })
    print("Evaluating on test set...")
    evaluation_results = trainer.evaluate(test_dataset)
    print("Test Evaluation results:", evaluation_results)

def compute_metrics(pred):
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    labels = pred.label_ids
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def see_confusion_matrix(model_path, test_df):
    # Load tokenizer and model
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = DebertaForSequenceClassification.from_pretrained(model_path)

    # Prepare the test data
    X_test = test_df['CV_statement'].tolist()  # Adjust the column name as per your dataset
    y_test = test_df['Good_Employee'].tolist()

    model.eval()  # Set the model to evaluation mode

    # Tokenize the test descriptions
    inputs = tokenizer(X_test, truncation=True, padding=True, max_length=512, return_tensors="pt")
    inputs = inputs.to(model.device)  # Ensure inputs are on the same device as the model

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()


    # Plot the confusion matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # Analyze prediction distribution
    unique, counts = np.unique(predictions, return_counts=True)
    prediction_distribution = dict(zip(unique, counts))
    print("Prediction Distribution:", prediction_distribution)

if __name__ == "__main__":
    # Load your datasets here
    train_df = open("Final_dataset/CV_Personal_Statements.csv", "r", encoding='utf-8-sig')
    train_df = pd.read_csv(train_df) ## size: 300
    print(f'size of train_df is: {len(train_df)}')
    print(train_df['Good_Employee'].value_counts())

    # val and test is the same dataset for now.

    val_df = open("Final_dataset/CV_Personal_Statements_w_cf.csv", "r", encoding='utf-8-sig')
    val_df = pd.read_csv(val_df)
    val_df = val_df[val_df['CF_on'] == 'ORG'] ## size: 50
    print(f'size of val_df is{len(val_df)}')
    print(val_df['Good_Employee'].value_counts())

    test_df = open("Final_dataset/CV_Personal_Statements_w_cf.csv", "r", encoding='utf-8-sig')
    test_df = pd.read_csv(test_df)
    test_df = test_df[test_df['CF_on'] == 'ORG'] ## size: 50
    print(f'size of test_df is{len(test_df)}')
    print(test_df['Good_Employee'].value_counts())

    train_cv_statement_classifier(train_df, val_df, test_df)
    model_path = "Trained_models/cv_statements_model"
    see_confusion_matrix(model_path, test_df)

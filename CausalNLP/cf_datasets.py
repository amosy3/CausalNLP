from datasets import Dataset
import pandas as pd
import numpy as np


def split_dataframe(df, frac1=0.3, frac2=0.3, random_state=42):
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate the sizes for the first and second splits
    size1 = int(len(df) * frac1)
    size2 = int(len(df) * frac2)

    # Split the DataFrame
    df1 = df_shuffled.iloc[:size1]
    df2 = df_shuffled.iloc[size1:size1 + size2]
    df3 = df_shuffled.iloc[size1 + size2:]

    return df1, df2, df3


def tokenize_dataframe(df, tokenizer, text_col='text', label_col='labels', num_labels=3):

    # Convert DataFrame to Huggingface Dataset
    dataset = Dataset.from_pandas(df[[text_col, label_col]])

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples[text_col], truncation=True, padding='longest')

    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)


    # Set the format to PyTorch tensors
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', label_col])

    return tokenized_dataset


def get_tokenized_datasets(tokenizer):
    df = pd.read_csv('../datasets/cv_w_cf.csv')
    df = df[df['CF_on']=="ORG"]
    df = df.rename(columns={'Good_Employee': 'labels'})
    df_train, df_val, df_test = split_dataframe(df, 0.1, 0.1)
    text = 'CV_statement'
    target = 'labels'

    train_dataset = tokenize_dataframe(df_train, tokenizer, text, target)
    val_dataset = tokenize_dataframe(df_val, tokenizer, text, target)
    return train_dataset, val_dataset
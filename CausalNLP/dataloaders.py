from torch.utils.data import Dataset

def tokenize_text(text, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    return inputs

class TokenizedDataset(Dataset):
    def __init__(self, dataframe, tokenizer, text_col, target_col, model='gpt2'):
        self.dataframe = dataframe
        self.text = dataframe[text_col].tolist()  # Select text columns
        if model == 't5':
            self.text = [f'classify: Rate the employee as 0 (Regular), 1 (Good), or 2 (Exceptional): {text}' for text in self.text]
        self.labels = dataframe[target_col].tolist()  # Select labels columns
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        tokenized_text = tokenize_text(self.text[idx], self.tokenizer)
        tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze(0)
        return tokenized_text, self.labels[idx]
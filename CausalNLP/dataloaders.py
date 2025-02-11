from torch.utils.data import Dataset

def tokenize_text(text, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    return inputs

class TokenizedDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.text = dataframe['CV_statement'].tolist()  # Select text columns
        self.labels = dataframe['Good_Employee'].tolist()  # Select labels columns
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        tokenized_text = tokenize_text(self.text[idx], self.tokenizer)
        tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze(0)
        return tokenized_text, self.labels[idx]
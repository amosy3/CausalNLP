from transformers import DebertaTokenizer, DebertaForSequenceClassification, DebertaConfig
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from torch import nn
class LanguageModel():
    def __init__(self, model_name='gpt2'):
        self.model_name = model_name
        if model_name == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=3)
            self.tokenizer.pad_token = self.tokenizer.eos_token # Set pad_token to eos_token to handle padding

        elif model_name == 'deberta':
            model_name = "microsoft/deberta-base"
            self.tokenizer = DebertaTokenizer.from_pretrained(model_name)
            config = DebertaConfig.from_pretrained(model_name)
            config.num_labels = 3
            config.output_hidden_states = True
            self.model = DebertaForSequenceClassification.from_pretrained(model_name, config=config)
        else:
            raise "Backbone was not found."
    def forward(self, sentences, output_type="logits"):
        # Tokenize the input batch of sentences
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

        # Get input_ids and attention mask
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        return outputs[output_type]

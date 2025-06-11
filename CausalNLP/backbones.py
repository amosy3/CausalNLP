from transformers import DebertaTokenizer, DebertaForSequenceClassification, DebertaConfig
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForSeq2SeqLM

import torch
from torch import nn



def get_model(model_name='gpt2', ckpt=None):
    if model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        if ckpt is None:
            model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=3)
        else:
            print("\n Loading pretrain model from %s \n" %ckpt)
            model = GPT2ForSequenceClassification.from_pretrained(ckpt, num_labels=3)

    elif model_name == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained("GilatToker/CV_Qwen")
        model = AutoModel.from_pretrained("GilatToker/CV_Qwen")

    elif model_name == 't5':
        tokenizer = AutoTokenizer.from_pretrained("GilatToker/CV_T5")
        model = AutoModelForSeq2SeqLM.from_pretrained("GilatToker/CV_T5")

    elif model_name == 'deberta':
        tokenizer = AutoTokenizer.from_pretrained("GilatToker/CV_Deberta")
        model = AutoModelForSequenceClassification.from_pretrained("GilatToker/CV_Deberta")

        # model_name = "microsoft/deberta-base"
        # tokenizer = DebertaTokenizer.from_pretrained(model_name)
        # config = DebertaConfig.from_pretrained(model_name)
        # config.num_labels = 3
        # config.output_hidden_states = True
        # model = DebertaForSequenceClassification.from_pretrained(model_name, config=config)
    else:
        raise "Backbone was not found."

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer



class LanguageModel():
    def __init__(self, model_name='gpt2'):
        self.model_name = model_name
        if model_name == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=3)
            # self.tokenizer.pad_token = self.tokenizer.eos_token # Set pad_token to eos_token to handle padding

        elif model_name == 'deberta':
            model_name = "microsoft/deberta-base"
            self.tokenizer = DebertaTokenizer.from_pretrained(model_name)
            config = DebertaConfig.from_pretrained(model_name)
            config.num_labels = 3
            config.output_hidden_states = True
            self.model = DebertaForSequenceClassification.from_pretrained(model_name, config=config)
        else:
            raise "Backbone was not found."

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def forward(self, sentences, output_type="logits"):
        # Tokenize the input batch of sentences
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

        # Get input_ids and attention mask
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        return outputs[output_type]

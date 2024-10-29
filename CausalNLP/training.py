import numpy as np
from transformers import Trainer, TrainingArguments
import torch
from torch import nn

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate accuracy
    correct = np.sum(predictions == labels)
    total = len(labels)
    accuracy = correct / total
    
    return {"accuracy": accuracy}


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Print input keys for debugging

        # Extract labels and remove them from inputs to avoid internal loss computation
        labels = inputs.pop("labels", None)  

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Define your custom loss function
        loss_fct = nn.CrossEntropyLoss()  # Use appropriate loss function

        # Compute loss only if labels are present
        if labels is not None:
            loss = loss_fct(logits, labels.long())
        else:
            print("Unexpected result!")
            loss = torch.tensor(0.0)  # Fallback to avoid crashes

        return (loss, outputs) if return_outputs else loss
    

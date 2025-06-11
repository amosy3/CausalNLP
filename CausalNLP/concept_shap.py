import torch
from torch import nn
from itertools import chain, combinations
import numpy as np
import math

class G(nn.Module):
    """
    Implements sup_g from the completeness definition.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(G, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        out = self.linear2(x)
        return out



def proj(concept):
    proj_matrix = (concept @ torch.inverse((concept.T @ concept))) \
                  @ concept.T  # (embedding_dim x embedding_dim)
    proj = proj_matrix @ train_embedding.T  # (embedding_dim x batch_size)

    # passing projected activations through rest of model
    return h_x(proj.T)

def powerset(iterable):
    "powerset([1,2,3]) --> [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]"
    s = list(iterable)
    pset = chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1))
    return [list(i) for i in list(pset)]

# completeness score
def n(y_gt, y_pred, y_prec_from_concepts, nconcepts):
    orig_correct = torch.sum((y_gt == y_pred).int())
    new_correct = torch.sum((y_gt == y_prec_from_concepts).int())
    return torch.div(new_correct - (1/nconcepts), orig_correct - (1/nconcepts))


# This code is based on https://github.com/agil27/TCAV_PyTorch/tree/master

import numpy as np
import os
import h5py
import torch
from torch import nn
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import save_object

class ModelWrapper(nn.Module):#object):
    def __init__(self, model, layers, backbone='gpt2'):
        super().__init__()
        # self.model = deepcopy(model)
        self.model = model
        self.intermediate_activations = {}
        self.gradients = None
        self.backbone = backbone

        def save_activation(name):
            '''create specific hook by module name'''

            def hook(module, input, output):
                # For T5, output might not be dict, but a model output object or tensor
                if self.backbone == 't5':
                    out = output[0]
                    # print(out.shape) #([2, 512, 768])
                    self.intermediate_activations[name] = out.requires_grad_(True)
                else:
                    # For GPT2 and others, output is dict with 'last_hidden_state'
                    if isinstance(output, dict) and 'last_hidden_state' in output:
                        self.intermediate_activations[name] = output['last_hidden_state'].requires_grad_(True)
                    else:
                        # fallback for normal tensor output
                        self.intermediate_activations[name] = output.requires_grad_(True)
            return hook
        
        for name, module in self.model.named_modules(): #_modules.items(): named_modules():
            # print(name)
            if name in layers:
                # register the hook
                module.register_forward_hook(save_activation(name))

    def save_gradient(self, grad):
        self.gradients = grad[:,0,:]

    def generate_gradients(self, c, layer_name):
        activation = self.intermediate_activations[layer_name]
        activation.register_hook(self.save_gradient) 
        logit = self.output['logits'][:, c]
        logit.backward(torch.ones_like(logit), retain_graph=True)
        # gradients = grad(logit, activation, retain_graph=True)[0]
        # gradients = gradients.cpu().detach().numpy()
        gradients = self.gradients.clone().detach().cpu().numpy()
        return gradients

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def __call__(self, x):
        if self.backbone == 't5':
            # Only run the encoder to get hidden states (for embeddings this is all we need)
            encoder_outputs = self.model.encoder(
                input_ids=x['input_ids'],
                attention_mask=x.get('attention_mask'),
                output_hidden_states=True,
                return_dict=True
            )
            self.output = encoder_outputs

        else:
            if 'attention_mask' in x and x['attention_mask'].dim() == 3:
                x['attention_mask'] = x['attention_mask'].squeeze(1)
            self.output = self.model(**x)
        return self.output
    


def get_activations(wmodel, output_dir, data_loader, concept_name, layer_names, max_samples, device):
    '''
    The function to generate the activations of all layers for ONE concept only
    :param model:
    :param output_dir:
    :param data_loader: the dataloader for the input of ONE concept
    :param layer_names:
    :param max_samples:
    :return:
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # wmodel = wmodel.to(device)
    # wmodel.eval()
    activations = {}
    for l in layer_names:
        activations[l] = []
    
    for i, data in enumerate(data_loader):
        if i == max_samples:
            break

        # data = data[0].to(device)
        # _ = wmodel(data)
        inputs, _ = data  # unpack batch: (batch_dict, labels)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if 'attention_mask' in inputs and inputs['attention_mask'].dim() == 3:
            inputs['attention_mask'] = inputs['attention_mask'].squeeze(1)

        _ = wmodel(inputs)

        for l in layer_names:
            z = wmodel.intermediate_activations[l][:,0,:].clone().detach().cpu().numpy() #['last_hidden_state']
            activations[l].append(z)

    for l in layer_names:
        activations[l] = np.concatenate(activations[l], axis=0)

    with h5py.File(os.path.join(output_dir, 'activations_%s.h5' % concept_name), 'w') as f:
        for l in layer_names:
            f.create_dataset(l, data=activations[l])

def load_activations(path):
    activations = {}
    with h5py.File(path, 'r') as f:
        for k, v in f.items():
            activations[k] = np.array(v)
    return activations

class TCAV(object):
    def __init__(self, wmodel, input_dataloader, concept_dataloaders, class_list, max_samples, device, logdir):
        self.model = wmodel
        self.input_dataloader = input_dataloader
        self.concept_dataloaders = concept_dataloaders
        self.concepts = list(concept_dataloaders.keys())
        self.output_dir = '%s/embeddings/' % logdir
        self.max_samples = max_samples
        self.lr = 1e-3
        self.model_type = 'logistic'
        self.class_list = class_list
        self.device = device

    def generate_activations(self, layer_names):
        for concept_name, data_loader in self.concept_dataloaders.items():
            get_activations(self.model, self.output_dir, data_loader, concept_name, layer_names, self.max_samples, self.device)

    def load_activations(self):
        self.activations = {}
        for concept_name in self.concepts:
            self.activations[concept_name] = load_activations(
                os.path.join(self.output_dir, 'activations_%s.h5' % concept_name))

    def generate_cavs(self, layer_name):
        cav_trainer = CAV(self.concepts, layer_name, self.lr, self.model_type)
        cav_trainer.train(self.activations)
        self.cavs = cav_trainer.get_cav()
        save_object(self.cavs, self.output_dir + 'cavs.pkl')
        save_object(self.concepts, self.output_dir + 'cavs_names.pkl')

    def calculate_tcav_score(self, layer_name, output_path):
        self.scores = np.zeros((self.cavs.shape[0], len(self.class_list)))
        for i, cav in enumerate(self.cavs):
            self.scores[i] = tcav_score(self.model, self.input_dataloader, cav, layer_name, self.class_list,
                                        self.concepts[i], self.device)
        # print(self.scores)
        np.save(output_path, self.scores)




def flatten_activations_and_get_labels(concepts, layer_name, activations):
    '''
    :param concepts: different name of concepts
    :param layer_name: the name of the layer to compute CAV on
    :param activations: activations with the size of num_concepts * num_layers * num_samples
    :return:
    '''
    # in case of different number of samples for each concept
    min_num_samples = np.min([activations[c][layer_name].shape[0] for c in concepts])
    # flatten the activations and mark the concept label
    data = []
    concept_labels = np.zeros(len(concepts) * min_num_samples)
    for i, c in enumerate(concepts):
        data.extend(activations[c][layer_name][:min_num_samples].reshape(min_num_samples, -1))
        concept_labels[i * min_num_samples : (i + 1) * min_num_samples] = i
    data = np.array(data)
    return data, concept_labels


class CAV(object):
    def __init__(self, concepts, layer_name, lr, model_type):
        self.concepts = concepts
        self.layer_name = layer_name
        self.lr = lr
        self.model_type = model_type

    def train(self, activations):
        data, labels = flatten_activations_and_get_labels(self.concepts, self.layer_name, activations)

        # default setting is One-Vs-All
        assert self.model_type in ['linear', 'logistic']
        if self.model_type == 'linear':
            model = SGDClassifier(alpha=self.lr)
        else:
            # model = LogisticRegression()
            model = LogisticRegression(solver='saga', max_iter=5000)

        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, stratify=labels)
        model.fit(x_train, y_train)
        '''
        The coef_ attribute is the coefficients in linear regression.
        Suppose y = w0 + w1x1 + w2x2 + ... + wnxn
        Then coef_ = (w0, w1, w2, ..., wn). 
        This is exactly the normal vector for the decision hyperplane
        '''
        if len(model.coef_) == 1:
            self.cav = np.array([-model.coef_[0], model.coef_[0]])
        else:
            self.cav = -np.array(model.coef_)

    def get_cav(self):
        return self.cav
    
def directional_derivative(model, cav, layer_name, class_name):
    gradient = model.generate_gradients(class_name, layer_name).reshape(-1)
    return np.dot(gradient, cav) < 0


def tcav_score(model, data_loader, cav, layer_name, class_list, concept, device):
    derivatives = {}
    for k in class_list:
        derivatives[k] = []

    tcav_bar = tqdm(data_loader)
    tcav_bar.set_description('Calculating tcav score for %s' % concept)
    for x, _ in tcav_bar:
        model.eval()
        if isinstance(x, dict):
            x = {k: v.to(device) for k, v in x.items()}
        
        x = x.to(device)
        outputs = model(x)
        k = int(outputs['logits'].max(dim=1)[1].clone().detach().cpu().numpy())
        if k in class_list:
            # derivatives[k].append(directional_derivative(model, cav, layer_name, k))

            ############################
            print("Activations:", model.intermediate_activations)
            print("Gradients:", model.gradients)

            for name, activation in model.intermediate_activations.items():
                activation.register_hook(model.save_gradient)

            logit = outputs['logits'][:, k]
            logit.backward(10*torch.ones_like(logit), retain_graph=True)

            print("Grads: ", model.gradients.shape, model.gradients)
            exit()
            gradients = model.gradients.cpu().detach().numpy()
            gradients = gradients.reshape(-1)
            gradient = np.dot(gradients, cav) < 0 
            derivatives[k].append(gradient)
            
            # ###########################3


    score = np.zeros(len(class_list))
    for i, k in enumerate(class_list):
        score[i] = np.array(derivatives[k]).astype(np.int64).sum(axis=0) / len(derivatives[k])
    return score
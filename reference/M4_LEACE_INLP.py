from transformers import DebertaTokenizer, DebertaForSequenceClassification, DebertaModel, DebertaPreTrainedModel, \
    pipeline
import numpy as np
import pandas as pd
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import Linear, Dropout
from transformers import DebertaTokenizer, DebertaForSequenceClassification, DebertaModel, DebertaPreTrainedModel
from concept_erasure import LeaceFitter
from scipy.spatial import distance

def pre_fit_leace_fitter_for_concept(train_df, textual_column, concept, tokenizer, model):
    # Assume dimensionality of X is the size of the model's hidden states and a single concept for simplicity
    dimensionality_of_X = model.config.hidden_size
    fitter = LeaceFitter(dimensionality_of_X, 1, dtype=torch.float32)  # Now handling a single concept

    train_df = train_df
    for _, row in train_df.iterrows():
        # print("row[Patient_id]: ", row['Patient_id'])
        inputs = tokenizer(row[textual_column], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.hidden_states[-1].mean(dim=1)  # Use mean pooling over the sequence
        # Assuming binary concepts for simplicity; adjust as needed for your dataset
        concept_labels = torch.tensor(row[concept])
        fitter.update(embeddings, concept_labels)
    print("done")
    return fitter

def train_M4(model_path, train_dataset_org, concepts_list):
    # Initialize tokenizer and model for DeBERTa
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = DebertaForSequenceClassification.from_pretrained(model_path)

    # Create a fitter for each concept and store in a dictionary
    fitters = {concept: pre_fit_leace_fitter_for_concept(train_dataset_org, "Patient_description", concept, tokenizer, model) for
               concept in concepts_list}
    return fitters

# def models_pred_1(model, tokenizer, textual_input):
#     inputs = tokenizer(textual_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     predicted_class_index = probs.argmax(dim=-1).item()
#     print("predicted_class_index: ", predicted_class_index)
#     return predicted_class_index

def models_pred(model, tokenizer, row):
    # Create a pipeline for sequence classification
    classification_pipeline = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, top_k=None,
        device=0 if torch.cuda.is_available() else -1
    )
    # print("row['Patient_description']: ", row['Patient_description'])
    # Predict the probabilities
    predictions = classification_pipeline(row['Patient_description'])
    # print("predictions: ", predictions)

    # Extract probabilities for each class
    conditions = ['Migraine', 'Sinusitis', 'Influenza']
    label_map = {condition: i for i, condition in enumerate(conditions)}
    probabilities = []

    for pred in predictions:
        # Get the scores for each condition
        condition_scores = {f'{condition}_p': pred[label_map[condition]]['score'] for condition in conditions}
        # Append the scores to the probabilities list
        probabilities.append(condition_scores)
    probabilities_list = [probabilities[0]['Migraine_p'], probabilities[0]['Sinusitis_p'], probabilities[0]['Influenza_p']]
    return probabilities_list

def run_M4(model_path, org_row, cf_row, fitters):
    # Initialize tokenizer and model for DeBERTa
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = DebertaForSequenceClassification.from_pretrained(model_path)

    # Corrected tokenization for the actual text content
    org_inputs = tokenizer(org_row["Patient_description"], return_tensors="pt", padding=True, truncation=True, max_length=512)
    org_input_ids = org_inputs["input_ids"].to(model.device)
    org_embeddings = model.get_input_embeddings()(org_input_ids)

    eraser = fitters[cf_row['CF_on']].eraser
    modified_embeddings = eraser(org_embeddings.cpu())  # Ensure embeddings are on the correct device

    # Perform prediction using the embeddings
    with torch.no_grad():
        # Ensure attention_mask is correctly passed along with inputs_embeds
        modified_embeddings_outputs = model(inputs_embeds=modified_embeddings, attention_mask=org_inputs["attention_mask"].to(model.device))
        # debug_embeddings_outputs = model(inputs_embeds=org_embeddings, attention_mask=org_inputs["attention_mask"].to(model.device))
    modified_logits = modified_embeddings_outputs.logits
    modified_distribution = torch.nn.functional.softmax(modified_logits, dim=-1)

    # debug_modified_logits= debug_embeddings_outputs.logits
    # debug_modified_distribution = torch.nn.functional.softmax(debug_modified_logits, dim=-1)
    # print("debug_modified_distribution: ", debug_modified_distribution)


    pred_M1_CF_list = modified_distribution.numpy().tolist()[0]
    pred_CF_list = models_pred(model, tokenizer, cf_row)
    pred_org_list = models_pred(model, tokenizer, org_row)
    # print("the first version of org", pred_org_list)

    # print("pred_org_list: ", pred_org_list)
    # print("pred_CF_list: ", pred_CF_list)
    # print("pred_M1_CF_list: ", pred_M1_CF_list)

    banchmark_diff = [cf - org for cf, org in zip(pred_CF_list, pred_org_list)]
    method_diff = [cf - org for cf, org in zip(pred_M1_CF_list, pred_org_list)]
    if np.all(banchmark_diff == 0) or np.all(method_diff == 0):
        return None, None, None

    cosine_distance = distance.cosine(banchmark_diff, method_diff)
    l2_dist = distance.euclidean(banchmark_diff, method_diff)
    norm_a = np.linalg.norm(banchmark_diff)
    norm_b = np.linalg.norm(method_diff)
    norm_diff = abs(norm_a - norm_b)
    return cosine_distance, l2_dist, norm_diff




























#
# # Works but not finished. return the in-correct value
# def pre_fit_leace_fitter_for_concept(train_df, textual_column, concept, tokenizer, model):
#     # Assume dimensionality of X is the size of the model's hidden states and a single concept for simplicity
#     dimensionality_of_X = model.config.hidden_size
#     fitter = LeaceFitter(dimensionality_of_X, 1, dtype=torch.float32)  # Now handling a single concept
#
#     train_df = train_df[:10]
#     for _, row in train_df.iterrows():
#         # print("row[Patient_id]: ", row['Patient_id'])
#         inputs = tokenizer(row[textual_column], return_tensors="pt", padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             embeddings = outputs.hidden_states[-1].mean(dim=1)  # Use mean pooling over the sequence
#         # Assuming binary concepts for simplicity; adjust as needed for your dataset
#         concept_labels = torch.tensor(row[concept])
#         fitter.update(embeddings, concept_labels)
#     print("done")
#     return fitter
#
# def train_M4(model_path, train_dataset_org, concepts_list):
#     # Initialize tokenizer and model for DeBERTa
#     tokenizer = DebertaTokenizer.from_pretrained(model_path)
#     model = DebertaForSequenceClassification.from_pretrained(model_path)
#
#     # Create a fitter for each concept and store in a dictionary
#     fitters = {concept: pre_fit_leace_fitter_for_concept(train_dataset_org, "Patient_description", concept, tokenizer, model) for
#                concept in concepts_list}
#     return fitters
#
# # def models_pred_1(model, tokenizer, textual_input):
# #     inputs = tokenizer(textual_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
# #     inputs = {k: v.to(model.device) for k, v in inputs.items()}
# #     with torch.no_grad():
# #         outputs = model(**inputs)
# #     logits = outputs.logits
# #     probs = torch.nn.functional.softmax(logits, dim=-1)
# #     predicted_class_index = probs.argmax(dim=-1).item()
# #     print("predicted_class_index: ", predicted_class_index)
# #     return predicted_class_index
#
# def models_pred(model, tokenizer, row):
#     # Create a pipeline for sequence classification
#     classification_pipeline = pipeline(
#         "text-classification", model=model, tokenizer=tokenizer, return_all_scores=True,
#         device=0 if torch.cuda.is_available() else -1
#     )
#     # print("row['Patient_description']: ", row['Patient_description'])
#     # Predict the probabilities
#     predictions = classification_pipeline(row['Patient_description'])
#     # print("predictions: ", predictions)
#
#     # Extract probabilities for each class
#     conditions = ['Migraine', 'Sinusitis', 'Influenza']
#     label_map = {condition: i for i, condition in enumerate(conditions)}
#     probabilities = []
#
#     for pred in predictions:
#         # Get the scores for each condition
#         condition_scores = {f'{condition}_p': pred[label_map[condition]]['score'] for condition in conditions}
#         # Append the scores to the probabilities list
#         probabilities.append(condition_scores)
#     probabilities_list = [probabilities[0]['Migraine_p'], probabilities[0]['Sinusitis_p'], probabilities[0]['Influenza_p']]
#     return probabilities_list
#
# def run_M4(model_path, org_row, cf_row, fitters):
#     # Initialize tokenizer and model for DeBERTa
#     tokenizer = DebertaTokenizer.from_pretrained(model_path)
#     model = DebertaForSequenceClassification.from_pretrained(model_path)
#
#     # Corrected tokenization for the actual text content
#     org_inputs = tokenizer(org_row["Patient_description"], return_tensors="pt", padding=True, truncation=True, max_length=512)
#     org_input_ids = org_inputs["input_ids"].to(model.device)
#     org_embeddings = model.get_input_embeddings()(org_input_ids)
#
#     eraser = fitters[cf_row['CF_on']].eraser
#     modified_embeddings = eraser(org_embeddings.cpu())  # Ensure embeddings are on the correct device
#
#     # Perform prediction using the embeddings
#     with torch.no_grad():
#         # Ensure attention_mask is correctly passed along with inputs_embeds
#         modified_embeddings_outputs = model(inputs_embeds=modified_embeddings, attention_mask=org_inputs["attention_mask"].to(model.device))
#     modified_logits = modified_embeddings_outputs.logits
#     modified_distribution = torch.nn.functional.softmax(modified_logits, dim=-1)
#     modified_distribution_list = modified_distribution.numpy().tolist()[0]
#     org_cf_distribution = models_pred(model, tokenizer, cf_row)
#
#     print("org_cf_distribution: ", org_cf_distribution)
#     print("modified_distribution_list: ", modified_distribution_list)
#
#     cosine_distance = distance.cosine(org_cf_distribution, modified_distribution_list)
#     l2_dist = distance.euclidean(org_cf_distribution, modified_distribution_list)
#     norm_a = np.linalg.norm(org_cf_distribution)
#     norm_b = np.linalg.norm(modified_distribution_list)
#     norm_diff = abs(norm_a - norm_b)
#     return cosine_distance, l2_dist, norm_diff
















































# def classify_with_and_without_concepts(train_df, test_df):
#     concepts_columns = ['Dizzy', 'Sensitivity_to_Light']
#
#     # Initialize tokenizer and model for DeBERTa
#     tokenizer = DebertaTokenizer.from_pretrained("Trained_models/disease_detection_textual_config")
#     model = DebertaForSequenceClassification.from_pretrained("Trained_models/disease_detection_textual_config")
#
#     # Create a fitter for each concept and store in a dictionary
#     fitters = {concept: pre_fit_leace_fitter_for_concept(train_df, "Patient_description", concept, tokenizer, model) for
#                concept in concepts_columns}
#
#     for index, row in test_df.iterrows():
#         # Original prediction
#         print("org model prediction: ", models_pred(model, tokenizer, row["Patient_description"]))
#
#         # Corrected tokenization for the actual text content
#         inputs = tokenizer(row["Patient_description"], return_tensors="pt", padding=True, truncation=True, max_length=512)
#         input_ids = inputs["input_ids"].to(model.device)
#
#         for concept, fitter in fitters.items():
#             eraser = fitter.eraser
#             # Convert input_ids to embeddings
#             embeddings = model.get_input_embeddings()(input_ids)
#             modified_embeddings = eraser(embeddings.cpu())  # Ensure embeddings are on the correct device
#             # print('org embeddings: ', embeddings)
#             # print('modified_embeddings: ', modified_embeddings)
#
#             # Perform prediction using the embeddings
#             with torch.no_grad():
#                 # Ensure attention_mask is correctly passed along with inputs_embeds
#                 org_outputs = model(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"].to(model.device))
#                 modified_outputs = model(inputs_embeds=modified_embeddings, attention_mask=inputs["attention_mask"].to(model.device))
#
#             org_logits = org_outputs.logits
#             org_probs = torch.nn.functional.softmax(org_logits, dim=-1)
#             org_predicted_class_index = org_probs.argmax(dim=-1).item()
#             print("org_probs", org_probs)
#
#             modified_logits = modified_outputs.logits
#             modified_probs = torch.nn.functional.softmax(modified_logits, dim=-1)
#             modified_predicted_class_index = modified_probs.argmax(dim=-1).item()
#             print("modified_probs: ", modified_probs)
#     return


# train_df = open("Prompt_experiments/cf_dataset_train.csv", "r", encoding='utf-8-sig')
# train_df = pd.read_csv(train_df).sample(n=103)
# test_df = open("Prompt_experiments/cf_dataset_test.csv", "r", encoding='utf-8-sig')
# test_df = pd.read_csv(test_df)
# test_df = test_df[test_df['ORG_or_CF'] == 'ORG'].sample(n=10)
#
# classify_with_and_without_concepts(train_df, test_df)



# return embeddings as needed
# from transformers import DebertaTokenizer, DebertaForSequenceClassification, DebertaModel, DebertaPreTrainedModel
# import pandas as pd
# import torch
# from transformers.modeling_outputs import SequenceClassifierOutput
# from torch.nn import Linear, Dropout
# from transformers import DebertaTokenizer, DebertaForSequenceClassification, DebertaModel, DebertaPreTrainedModel
# from concept_erasure import LeaceFitter
#
# def models_pred(model, tokenizer, textual_input):
#     inputs = tokenizer(textual_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     predicted_class_index = probs.argmax(dim=-1).item()
#     return predicted_class_index
#
# def classify_with_and_without_concepts(model_path, train_df, test_df, textual_column, concepts_columns):
#
#     # Initialize tokenizer and model for DeBERTa
#     tokenizer = DebertaTokenizer.from_pretrained(model_path)
#     org_model = DebertaForSequenceClassification.from_pretrained(model_path)
#
#     for index, row in test_df.iterrows():
#         # Original prediction
#         print("org model prediction: ", models_pred(org_model, tokenizer, row[textual_column]))
#
#         # Corrected tokenization for the actual text content
#         inputs = tokenizer(row[textual_column], return_tensors="pt", padding=True, truncation=True, max_length=512)
#         input_ids = inputs["input_ids"].to(org_model.device)
#
#         # Convert input_ids to embeddings
#         embeddings = org_model.get_input_embeddings()(input_ids)
#
#         # Perform prediction using the embeddings
#         with torch.no_grad():
#             # Ensure attention_mask is correctly passed along with inputs_embeds
#             outputs = org_model(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"].to(org_model.device))
#
#         logits = outputs.logits
#         probs = torch.nn.functional.softmax(logits, dim=-1)
#         predicted_class_index = probs.argmax(dim=-1).item()
#         print("predicted_class_index: ", predicted_class_index)
#     return
#
# # Assume 'model_path' is your DeBERTa model path, 'df' is your DataFrame
# model_path = "Trained_models/disease_detection_textual_config"
#
# train_df = open("Prompt_experiments/cf_dataset_train.csv", "r", encoding='utf-8-sig')
# train_df = pd.read_csv(train_df).sample(n=193)
# test_df = open("Prompt_experiments/cf_dataset_test.csv", "r", encoding='utf-8-sig')
# test_df = pd.read_csv(test_df)
# test_df = test_df[test_df['ORG_or_CF'] == 'ORG'].sample(n=100)
#
# # Identify label - It will not be needed in the future because the data will be with a column 'Label'
# conditions = ['Migraine', 'Sinusitis', 'Influenza']
# label_map = {condition: i for i, condition in enumerate(conditions)}
# train_df['Label'] = (train_df[conditions] == 1).idxmax(axis=1)
# train_df['Label'] = train_df['Label'].map(label_map)
# test_df['Label'] = (test_df[conditions] == 1).idxmax(axis=1)
# test_df['Label'] = test_df['Label'].map(label_map)
#
# concepts_columns = ['Dizzy', 'Sensitivity_to_Light', 'Headache', 'Nasal_Congestion', 'Facial_Pain_Pressure', 'Fever','General_Weakness']
# results_df = classify_with_and_without_concepts(model_path, train_df, test_df, "Patient_description", concepts_columns)
##### with the LEACE
# # Its working but the classification is allways 2 wich means I do something incorrect
# from transformers import DebertaTokenizer, DebertaForSequenceClassification, DebertaModel, DebertaPreTrainedModel, \
#     pipeline
# import pandas as pd
# import torch
# from concept_erasure import LeaceFitter
# from transformers.modeling_outputs import SequenceClassifierOutput
# from torch.nn import Linear, Dropout
#
#
# class ModifiedDebertaForSequenceClassification(DebertaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#
#         self.deberta = DebertaModel(config)
#
#         # Classifier/Pooler has to be initialized as it is in the original DebertaForSequenceClassification
#         self.classifier = Linear(config.hidden_size, self.num_labels)
#
#         self.dropout = Dropout(config.hidden_dropout_prob)
#
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             inputs_embeds=None,
#             labels=None,
#             output_attentions=None,
#             output_hidden_states=None,
#             return_dict=None,
#     ):
#         # If inputs_embeds is provided, skip embedding layer
#         if inputs_embeds is not None:
#             outputs = self.deberta(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
#         else:
#             # print("None")
#             outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
#                                    position_ids=position_ids)
#         sequence_output = outputs.last_hidden_state
#         logits = self.classifier(sequence_output[:, 0])
#         return SequenceClassifierOutput(
#             logits=logits,
#             hidden_states=outputs.hidden_states if self.config.output_hidden_states else None,
#             attentions=outputs.attentions if self.config.output_attentions else None
#         )
#
#     def get_embeddings(self, input_ids, attention_mask):
#         self.eval()
#         with torch.no_grad():
#             outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
#             # Extract the [CLS] token's embedding from the last layer
#             cls_embeddings = outputs.last_hidden_state[:, 0, :]
#         return cls_embeddings
#
#     def predict_from_embeddings(self, embeddings, attention_mask=None):
#         embeddings = embeddings.to(self.deberta.device)
#         logits = self.classifier(embeddings)  # Directly use classifier on [CLS] embeddings
#         return logits
#
# def pre_fit_leace_fitter(train_df, textual_column, concepts_columns, tokenizer, model):
#     # Assume dimensionality of X is the size of the model's hidden states and a single concept for simplicity
#     dimensionality_of_X = model.config.hidden_size
#     number_of_concepts = len(concepts_columns)
#
#     fitter = LeaceFitter(dimensionality_of_X, number_of_concepts, dtype=torch.float32)
#
#     for _, row in train_df.iterrows():
#         inputs = tokenizer(row[textual_column], return_tensors="pt", padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             embeddings = outputs.hidden_states[-1].mean(dim=1)  # Use mean pooling over the sequence
#         # Assuming binary concepts for simplicity; adjust as needed for your dataset
#         concept_labels = torch.tensor(row[concepts_columns].tolist()).unsqueeze(0)
#
#         fitter.update(embeddings, concept_labels)
#
#     return fitter
#
# def models_pred(model, tokenizer, textual_input):
#     # Tokenize the textual input
#     inputs = tokenizer(textual_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
#
#     # Move inputs to the same device as model
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#
#     # Perform inference
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     # Extract logits
#     logits = outputs.logits
#
#     # Convert logits to probabilities (softmax)
#     probs = torch.nn.functional.softmax(logits, dim=-1)
#
#     # Get the predicted class (the one with the highest probability)
#     predicted_class_index = probs.argmax(dim=-1).item()
#
#     return predicted_class_index
#
#
# def classify_with_and_without_concepts(model_path, train_df, test_df, textual_column, concepts_columns):
#     """
#     Classify textual data using the original and LEACE-modified embeddings with DeBERTa, including pre-fitting.
#     """
#
#     # Initialize tokenizer and model for DeBERTa
#     tokenizer = DebertaTokenizer.from_pretrained(model_path)
#     # model = DebertaForSequenceClassification.from_pretrained(model_path)
#     model = ModifiedDebertaForSequenceClassification.from_pretrained(model_path)
#     org_model = DebertaForSequenceClassification.from_pretrained(model_path)
#
#     # Pre-fit the LeaceFitter with embeddings and concept labels from the dataset
#     fitter = pre_fit_leace_fitter(train_df, textual_column, concepts_columns, tokenizer, model)
#
#     # Generate the eraser from fitted statistics
#     eraser = fitter.eraser
#
#     results = []
#
#     for index, row in test_df.iterrows():
#         inputs = tokenizer(row[textual_column], return_tensors="pt", padding=True, truncation=True, max_length=512)
#         attention_mask = inputs['attention_mask']
#
#         # Get embeddings for the original text
#         embeddings = model.get_embeddings(input_ids=inputs['input_ids'], attention_mask=attention_mask)
#         # print("org embeddings that will classify according to this : ", embeddings)
#
#         # Apply the pre-fitted eraser to the embeddings
#         modified_embeddings = eraser(embeddings.cpu())  # Ensure embeddings are on the correct device
#
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # Predict from the modified embeddings
#         output_logits = torch.softmax(model.predict_from_embeddings(embeddings=embeddings.to(device),
#                                                                attention_mask=attention_mask.to(device)), dim=1)
#         modified_output_logits =  torch.softmax(model.predict_from_embeddings(embeddings=modified_embeddings.to(device),
#                                                                attention_mask=attention_mask.to(device)), dim=1)
#         #
#         # print("output_logits: ", output_logits)
#         # print("modified_output_logits: ", modified_output_logits)
#         classification = torch.argmax(output_logits, dim=1)
#         modified_classification = torch.argmax(modified_output_logits, dim=1)
#         print("golden label: ", row['Label'])
#         print("org model prediction: ", models_pred(org_model, tokenizer, row['Patient_description']))
#         print("classification: ", classification)
#         print("modified_classification", modified_classification)
#
#     return pd.DataFrame(results)
#
# # Assume 'model_path' is your DeBERTa model path, 'df' is your DataFrame
# model_path = "Trained_models/disease_detection_textual_config"
#
# train_df = open("Prompt_experiments/cf_dataset_train.csv", "r", encoding='utf-8-sig')
# train_df = pd.read_csv(train_df).sample(n=193)
#
# test_df = open("Prompt_experiments/cf_dataset_test.csv", "r", encoding='utf-8-sig')
# test_df = pd.read_csv(test_df)
# test_df = test_df[test_df['ORG_or_CF'] == 'ORG'].sample(n=100)
#
# # Identify label - It will not be needed in the future because the data will be with a column 'Label'
# conditions = ['Migraine', 'Sinusitis', 'Influenza']
# label_map = {condition: i for i, condition in enumerate(conditions)}
# train_df['Label'] = (train_df[conditions] == 1).idxmax(axis=1)
# train_df['Label'] = train_df['Label'].map(label_map)
# test_df['Label'] = (test_df[conditions] == 1).idxmax(axis=1)
# test_df['Label'] = test_df['Label'].map(label_map)
#
# concepts_columns = ['Dizzy', 'Sensitivity_to_Light', 'Headache', 'Nasal_Congestion', 'Facial_Pain_Pressure', 'Fever','General_Weakness']
# results_df = classify_with_and_without_concepts(model_path, train_df, test_df, "Patient_description", concepts_columns)


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import random
import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, pipeline
from scipy.spatial import distance


def train_s_learner(train_data, aspects):
    # Load the dataset
    df = pd.read_csv(train_data)
    # Define the aspects (features) and the outcome
    models_prediction = 'models_label'

    print("Class distribution in the dataset:", df[models_prediction].value_counts())

    # Prepare the feature matrix X and the target vector y
    X = df[aspects].values
    y = df[models_prediction].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize the logistic regression model
    s_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

    # Train the model on the training data
    s_model.fit(X_train, y_train)
    # print the accuracy of the model
    y_pred = s_model.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=["0","1", "2"], columns=["0","1", "2"])
    print("Confusion Matrix:", cm_df)

    return s_model

# def aspects_vector_change(row_sample, aspects):
#     # Given a row, return 2 aspects vector. aspect_vector_org - according to the org row aspects columns values, aspect_vector_cf - choose rendomly 1 asspect and change it
#     aspect_vector_org = row_sample[aspects].values.reshape(1, -1)
#     print("aspect_vector_org: ", aspect_vector_org)
#
#     # Copy the original aspect vector to create the counterfactual vector
#     aspect_vector_cf = aspect_vector_org.copy()
#
#     # Randomly select an aspect to change
#     aspect_to_change = random.choice(aspects)
#
#     # Randomly select a new value for this aspect (which should be different from the original value)
#     possible_values = [0, 1, 2]
#     current_value = row_sample[aspect_to_change]
#     possible_values.remove(current_value)
#     new_value = random.choice(possible_values)
#
#     # Apply the change to the counterfactual vector
#     aspect_vector_cf[0][aspects.index(aspect_to_change)] = new_value
#     # print("aspect_vector_cf: ", aspect_vector_cf)
#     return aspect_vector_org, aspect_vector_cf


# Predicts the Symptom based on patient description
def predict_aspect(model, tokenizer, patient_description, aspect):
    combined_text = f"Symptom: {aspect}. Description: {patient_description}"
    inputs = tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    predicted_label = torch.argmax(predictions, dim=1).item()
    # if predicted_label ==2:
        # print("predicted_label: ", predicted_label)
    return predicted_label

def aspects_vector_change_prox(row_sample, aspects, row_cf):
    model_path = "Trained_models/disease_symptoms_detection_class"
    model = DebertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = DebertaTokenizer.from_pretrained(model_path)

    patient_description = row_sample['Patient_description']
    aspect_vector_org = {}

    # Here, we simulate the original aspect vector with predicted labels
    for i, aspect in enumerate(aspects):
        # aspect_vector_org[0][i] = predict_aspect(model, tokenizer, patient_description, aspect)  # This function should be adjusted to predict the specific aspect if possible
        aspect_vector_org[aspect] = predict_aspect(model, tokenizer, patient_description, aspect)

    # Copy the original aspect vector to create the counterfactual vector
    aspect_vector_cf = aspect_vector_org.copy()

    # # Randomly select an aspect to change
    # aspect_to_change = random.choice(aspects)
    #
    # # Randomly select a new value for this aspect (which should be different from the original value)
    # possible_values = [0, 1, 2]
    # current_value = row_sample[aspect_to_change]
    # possible_values.remove(current_value)
    # possible_values.remove(current_value)
    # new_value = random.choice(possible_values)

    # Apply the change to the counterfactual vector
    aspect_vector_cf[row_cf["CF_on"]] = row_cf[row_cf["CF_on"]]
    # print("aspect_vector_cf: ", aspect_vector_cf)
    # print("aspect_vector_org before change: ", aspect_vector_org)
    aspect_vector_org = np.array(list(aspect_vector_org.values())).reshape(1, -1)
    aspect_vector_cf = np.array(list(aspect_vector_cf.values())).reshape(1, -1)
    # print("aspect_vector_cf after change: ", aspect_vector_cf)
    return aspect_vector_org, aspect_vector_cf

def models_pred(model, tokenizer, row ):
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
    # print("probabilities: ", probabilities)
    return probabilities

def S_learner_output(s_learner_model, aspect_vector_org_cf):
    # Predict the outcome for the original and counterfactual examples
    ditribution_s_learner = s_learner_model.predict_proba(aspect_vector_org_cf)[0]
    return ditribution_s_learner

def run_M3(model_path,s_learner_model,concepts_list, org_row, cf_row):
    # First option - Based on real aspects values
    # aspect_vector_org_cf = cf_row[concepts_list].values.reshape(1, -1)
    # print("aspect_vector_org_cf for s-lerner", aspect_vector_org_cf)
    # pred_M3_CF_list = S_learner_output(s_learner_model, aspect_vector_org_cf)


    # # Second option - Based on predicted aspects values
    aspect_vector_org, aspect_vector_cf = aspects_vector_change_prox(org_row, concepts_list, cf_row)
    pred_M3_org_list = S_learner_output(s_learner_model, aspect_vector_org)
    pred_M3_CF_list = S_learner_output(s_learner_model, aspect_vector_cf)
    # print(f"Counterfactuals score for row - prox {org_row.name}: {pred_M3_CF_list}")


    model = DebertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    # Move model to the right device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Get predictions for the sample row
    pred_org = models_pred(model, tokenizer, org_row)
    pred_org_list = [pred_org[0]['Migraine_p'], pred_org[0]['Sinusitis_p'],
                             pred_org[0]['Influenza_p']]

    # Get predictions for the sample row
    pred_CF = models_pred(model, tokenizer, cf_row)
    pred_CF_list = [pred_CF[0]['Migraine_p'], pred_CF[0]['Sinusitis_p'],
                             pred_CF[0]['Influenza_p']]

    banchmark_diff = [cf - org for cf, org in zip(pred_CF_list, pred_org_list)]
    method_diff = [cf - org for cf, org in zip(pred_M3_CF_list, pred_M3_org_list)]
    if np.all(banchmark_diff == 0) or np.all(method_diff == 0):
        return None, None, None

    cosine_distance = distance.cosine(banchmark_diff, method_diff)
    l2_dist = distance.euclidean(banchmark_diff, method_diff)
    norm_a = np.linalg.norm(banchmark_diff)
    norm_b = np.linalg.norm(method_diff)
    norm_diff = abs(norm_a - norm_b)
    return cosine_distance, l2_dist, norm_diff



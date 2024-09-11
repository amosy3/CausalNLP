import numpy as np
import pandas as pd
import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, pipeline

def models_pred(model, tokenizer, row):
    # Create a pipeline for sequence classification
    classification_pipeline = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, return_all_scores=True,
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

def calculate_average_prediction(model, tokenizer, df_subset):
    # Use the models_pred function to get predictions for each row in the subset and calculate the average
    predictions = [models_pred(model, tokenizer, row) for _, row in df_subset.iterrows()]
    # print("predictions: ", predictions)
    # Convert predictions to numpy array for easy averaging
    predictions_array = np.array(predictions)
    return np.mean(predictions_array, axis=0)

def CONEXP_output(model_path, df, concept_column, value_c, value_c_cf):
    # Load the model and tokenizer
    model = DebertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Separate the dataset based on the concept values
    df_c = df[df[concept_column] == value_c]
    df_c_cf = df[df[concept_column] == value_c_cf]

    # Calculate the average predictions for each subset
    avg_pred_c = calculate_average_prediction(model, tokenizer, df_c)
    # print("avg_pred_c: ", avg_pred_c)
    avg_pred_c_cf = calculate_average_prediction(model, tokenizer, df_c_cf)
    # print("avg_pred_c_cf: ", avg_pred_c_cf)

    # Compute the difference in average predictions
    effect = avg_pred_c_cf - avg_pred_c
    return effect

def run_M2():
    model_path = "Trained_models/disease_detection_textual_config"
    from_name = "Prompt_experiments/cf_dataset_train.csv"
    df = open(from_name, "r", encoding='utf-8-sig')
    df = pd.read_csv(df)
    concepts_list = ['Dizzy', 'Sensitivity_to_Light', 'Headache', 'Nasal_Congestion', 'Facial_Pain_Pressure', 'Fever',
                     'General_Weakness']
    possible_values = [0, 1, 2]
    df = df[df['ORG_or_CF'] == 'ORG']
    for concept in concepts_list:
        for value in possible_values:
            for value_cf in possible_values:
                if value == value_cf:
                    continue
                value_c = value
                value_c_cf = value_cf
                effect = CONEXP_output(model_path, df, concept, value_c, value_c_cf)
                print(f"Effect of concept '{concept}' changing from value '{value}' to value '{value_cf}': {effect}")

run_M2()
import numpy as np
import pandas as pd
import itertools
import torch
from transformers import DebertaTokenizer, DebertaForSequenceClassification, pipeline
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def sample_counterfactual(train_df, columns, row):
    # Ensure the index is reset for proper masking and dropping
    train_df = train_df.reset_index(drop=True)
    # Create a mask for rows that match the sample_row values for the specified columns
    mask = (train_df[columns] == row[columns]).all(axis=1)
    # Filter the dataframe to only those rows that match the sample_row
    matching_rows = train_df[mask]

    if len(matching_rows) < 1:
        return None  # No counterfactuals available

    # Randomly sample a row from the matching rows to serve as the counterfactual
    cf_sample = matching_rows.sample(n=1).iloc[0]
    return cf_sample

def models_pred(model, tokenizer, row ):
    # Create a pipeline for sequence classification
    classification_pipeline = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, top_k=None,
        device=0 if torch.cuda.is_available() else -1
    )

    # # Check the length of the tokenized input
    # tokens = tokenizer(row['Patient_description'], return_tensors='pt')
    # if tokens['input_ids'].shape[1] > 512:
    #     print(f"Warning: Patient ID {row['Patient_id']} has a sequence longer than the maximum length.")

    # print("row['Patient_description']: ", row['Patient_description'])
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


def approximate_counterfactuals_output(model_path, train_df, columns, apply_on_row_org, apply_on_row_cf):
    model = DebertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Get predictions for the sample row
    pred_org = models_pred(model, tokenizer, apply_on_row_org)
    pred_org_list = [pred_org[0]['Migraine_p'], pred_org[0]['Sinusitis_p'],
                             pred_org[0]['Influenza_p']]
    pred_CF = models_pred(model, tokenizer, apply_on_row_cf)
    pred_CF_list = [pred_CF[0]['Migraine_p'], pred_CF[0]['Sinusitis_p'], pred_CF[0]['Influenza_p']]

    sample_cf = sample_counterfactual(train_df, columns, apply_on_row_cf)
    if sample_cf is None:
        return None, None, None
    pred_M1_CF = models_pred(model, tokenizer, sample_cf)
    pred_M1_CF_list = [pred_M1_CF[0]['Migraine_p'], pred_M1_CF[0]['Sinusitis_p'], pred_M1_CF[0]['Influenza_p']]
    # print("pred_M1_CF",pred_M1_CF)
    # print("pred_M1_CF_list", pred_M1_CF_list)

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

# not part of the pipeline - just helps to understand if the model pay attention to the story or just to the aspects
def calculate_similarity(vec1, vec2):
    return 1 - distance.cosine(vec1, vec2)

def calculate_mean_similarity(grouped, model_path):
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = DebertaForSequenceClassification.from_pretrained(model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    all_similarities = []

    for _, group in grouped:
        if len(group) > 1:
            predictions = []
            for _, row in group.iterrows():
                pred = models_pred(model, tokenizer, row)[0]
                pred_vector = [pred[f'{condition}_p'] for condition in ['Migraine', 'Sinusitis', 'Influenza']]
                predictions.append(pred_vector)

            similarities = []
            for vec1, vec2 in itertools.combinations(predictions, 2):
                sim = calculate_similarity(vec1, vec2)
                similarities.append(sim)

            if similarities:
                mean_similarity = np.mean(similarities)
                all_similarities.append(mean_similarity)

    if all_similarities:
        # print(f"All similarity between predictions for identical aspect vectors: {all_similarities}")
        overall_mean_similarity = np.mean(all_similarities)
        return overall_mean_similarity
    else:
        return None

def analyze_similarity_vs_scores(df, concepts_list, model_path):
    # Calculate similarity for actual groups
    grouped_actual = df.groupby(concepts_list)
    mean_similarity_actual = calculate_mean_similarity(grouped_actual, model_path)

    # Calculate similarity for random groups
    df_shuffled = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset
    df_shuffled['Random_Group'] = df_shuffled.index % len(grouped_actual)  # Create random groups
    grouped_random = df_shuffled.groupby('Random_Group')
    mean_similarity_random = calculate_mean_similarity(grouped_random, model_path)

    print(f"Mean similarity between predictions for identical aspect vectors: {mean_similarity_actual}")
    print(f"Mean similarity between predictions for random groups: {mean_similarity_random}")

    return mean_similarity_actual, mean_similarity_random

# # Example usage with your dataset
# model_path = "Trained_models/disease_detection_textual_config"
# train_dataset = pd.read_csv("Prompt_experiments/cf_dataset_train.csv", encoding='utf-8-sig')
#
# concepts_list = ['Dizzy', 'Sensitivity_to_Light', 'Headache', 'Nasal_Congestion', 'Facial_Pain_Pressure', 'Fever', 'General_Weakness']
# mean_similarity_actual, mean_similarity_random = analyze_similarity_vs_scores(train_dataset, concepts_list, model_path)


def check_same_aspects_rows():

    model_path = "Trained_models/disease_detection_textual_config"
    train_dataset = open("Prompt_experiments/cf_dataset_train.csv", "r", encoding='utf-8-sig')
    df = pd.read_csv(train_dataset)
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = DebertaForSequenceClassification.from_pretrained(model_path)
    concepts_list = ['Dizzy', 'Sensitivity_to_Light', 'Headache', 'Nasal_Congestion', 'Facial_Pain_Pressure', 'Fever',
                     'General_Weakness']

    grouped_actual = df.groupby(concepts_list)
    for _, group in grouped_actual:
        print("Group:")
        for _, row in group.iterrows():
                    pred_distribution = models_pred(model, tokenizer, row)[0]
                    print(f"ID {row['Patient_id']}: {pred_distribution}")


check_same_aspects_rows()
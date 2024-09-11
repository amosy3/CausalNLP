import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.spatial import distance
import os

from M1_Approximate_Counterfactuals import approximate_counterfactuals_output
from M3_SLEARNER import train_s_learner, run_M3
from M4_LEACE_INLP import train_M4, run_M4

FITTERS_FILE = "fitters.pkl"
S_LEARNER_FILE = "s_learner_model.pkl"

def save_fitters(fitters, filename):
    with open(filename, 'wb') as f:
        pickle.dump(fitters, f)
def load_fitters(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_s_learner_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_s_learner_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    model_path = "Trained_models/disease_detection_textual_config"
    train_dataset = pd.read_csv("Prompt_experiments/cf_dataset_train.csv", encoding='utf-8-sig')
    train_dataset = train_dataset
    train_dataset_org = train_dataset[train_dataset['ORG_or_CF'] == 'ORG']
    print("train_dataset_org size:", train_dataset_org.shape)

    test_dataset = pd.read_csv("Prompt_experiments/cf_dataset_test.csv", encoding='utf-8-sig')
    test_dataset = test_dataset
    test_dataset_wo_n = test_dataset[test_dataset['n'].isin([0, 1])][:1500]
    print("test_dataset_wo_n size:", test_dataset_wo_n.shape)

    concepts_list = ['Dizzy', 'Sensitivity_to_Light', 'Headache', 'Nasal_Congestion', 'Facial_Pain_Pressure', 'Fever','General_Weakness']
    possible_values = [0, 1, 2]

    methods = ["M1_approx", "M3_Slearner", "M4_Leace"]
    results = {method: {'cosine': [], 'l2': [], 'norm_diff': []} for method in methods}

    # pre methods training
    if os.path.exists(FITTERS_FILE):
        fitters = load_fitters(FITTERS_FILE)
        print("Fitters loaded from file.")
    else:
        fitters = train_M4(model_path, train_dataset_org, concepts_list)
        save_fitters(fitters, FITTERS_FILE)
        print("Fitters trained and saved to file.")

    if os.path.exists(S_LEARNER_FILE):
        s_learner_model = load_s_learner_model(S_LEARNER_FILE)
        print("S-learner model loaded from file.")
    else:
        s_learner_model = train_s_learner("Disease_dataset/disease_regular_model_prediction.csv", concepts_list)
        save_s_learner_model(s_learner_model, S_LEARNER_FILE)
        print("S-learner model trained and saved to file.")

    total_counter = 0
    grouped = test_dataset_wo_n.groupby('Patient_id')
    print("num of groups is: ", len(grouped))
    for patient_id, group in grouped:
        print(total_counter)
        org_row = group[group['ORG_or_CF'] == 'ORG'].iloc[0] if not group[group['ORG_or_CF'] == 'ORG'].empty else None
        if org_row is not None:
            cf_df = group[group['ORG_or_CF'] == 'CF']
            for index, row in cf_df.iterrows():
                total_counter += 1
                cf_row = row
                for method in methods:
                    if method == "M1_approx":
                        cosine_distance, l2_dist, norm_diff = approximate_counterfactuals_output(model_path, train_dataset_org, concepts_list, org_row, cf_row)
                    elif method == "M3_Slearner":
                        cosine_distance, l2_dist, norm_diff = run_M3(model_path, s_learner_model, concepts_list, org_row, cf_row)
                    elif method == "M4_Leace":
                        if cf_row[cf_row["CF_on"]] == 0 and org_row[cf_row["CF_on"]] >cf_row[cf_row["CF_on"]]:
                            cosine_distance, l2_dist, norm_diff = run_M4(model_path, org_row, cf_row, fitters)
                        else:
                            cosine_distance, l2_dist, norm_diff = None, None, None
                    else:
                        print(f'method {method} - not exist')
                        continue
                    if cosine_distance is not None:
                        results[method]['cosine'].append(cosine_distance)
                    if l2_dist is not None:
                        results[method]['l2'].append(l2_dist)
                    if norm_diff is not None:
                        results[method]['norm_diff'].append(norm_diff)
        else:
            print(f"No 'ORG' row found for Patient ID: {patient_id}")

    print(f"M1_list succeed {len(results['M1_approx']['cosine'])} times out of {total_counter}")
    print(f"M4_list succeed {len(results['M4_Leace']['cosine'])} times out of {total_counter}")

    # Calculate means and prepare data for plotting
    methods = ["M1_approx", "M3_Slearner", "M4_Leace"]
    metrics = ['cosine', 'l2', 'norm_diff']
    colors = ['blue', 'orange', 'green']  # Colors for the three metrics

    num_methods = len(methods)
    bar_width = 0.2
    index = np.arange(num_methods)  # the x locations for the groups

    # Creating the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each method's results
    for i, metric in enumerate(metrics):
        means = [np.mean(results[method][metric]) if results[method][metric] else print("problem") for method in methods]
        plt.bar(index + i * bar_width, means, bar_width, label=metric, color=colors[i])

    # Adding labels and titles
    ax.set_xlabel('Methods')
    ax.set_ylabel('Mean Values')
    ax.set_title('Performance Evaluation Across Methods')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(methods)
    ax.legend(title="Metrics")

    plt.show()


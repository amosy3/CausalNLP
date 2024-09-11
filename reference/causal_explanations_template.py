import pandas as pd
import numpy as np
from datasets import load_dataset
import torch
from typing import List, Dict, Any, Tuple, Union, Optional
import json
import os
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer, TrainingArguments


def save_json(data: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def load_cebab():
    cebab = load_dataset('CEBaB/CEBaB')
    cebab = {k: cebab[k].to_pandas() for k in ['train_inclusive', 'validation', 'test']}
    return cebab


def preprocess_cebab(cebab: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Write a function that unifies the dataframe into the following schema:
    split, is_original, original_id, cf_id, text, [concepts]
    The [concepts] are several columns, each containing a concept name and its value
     (e.g., in CEBaB, it will be food, ambiance, noise, service, and rating/review).
    “cf” means counterfactual
    original_id = f"{split}_{original_id}"
    cf_id = f"{split}_{edit_id}"
    """
    pass


def create_concepts_mapping(df_dict: Dict[str, pd.DataFrame],
                            concepts: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Create a mapping dict. Each concept is a key, the value is another dict with a mapping/enum
    between a concept value and an integer.
    Make sure the no_majority value is the last value.
    For example, {“food”: {“unknown”: 0, “Positive”: 1, “Negative”: 2, “no_majority”: 3}}.
    (tip: you can see the values of each concept with df[concept].unique())
    """
    pass


def replace_concepts(df_dict: Dict[str, pd.DataFrame],
                     concepts_mapping: Dict[str, Dict[str, int]]) -> Dict[str, pd.DataFrame]:
    """
    Replace the concepts with the corresponding integer values.
    """
    pass


def split_cebab_data(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Split the data into:
    train_inclusive → train_original (is_original == False), train_cf (is_original == False)
    train_original → train_original (60%), match_set_original (40%) (use seed=42 and df.sample() without replace).
    train_cf → train_cf (if original_id in train_original), match_set_cf (if original_id in match_set_original).
    match_set = match_set_original + match_set_cf (concat the two dataframes)
    validation → dev_original  (is_original == False), dev_cf (is_original == False)
    test → test_original (is_original == False), test_cf (is_original == False)
    At the end of this step, you should have the following:
    train_original, train_cf, match_set, dev_original, dev_cf, test_original, test_cf
    """
    pass


def train_concept_classifier(model_name: str,
                             outputs_folder: str,
                             text_col: str,
                             concept_col: str,
                             train_df: pd.DataFrame,
                             dev_df: pd.DataFrame,
                             test_df: Optional[pd.DataFrame] = None):
    """
    Write a function that trains an NLP model to predict a concept from the text.
    Use the DeBERTa model for concept predictors. You should read the tutorials on how to fine-tune models.
    Train a classifier NLP model for each concept. The input is the text, and the output is the concept.
    You do not need to achieve 100% accuracy…
    Save the checkpoints of the models, document the hyperparameters you use, document the model loss, accuracy, f1
    """
    pass


def load_model(model_path: str) -> PreTrainedModel:
    """
    Load the model from the model_path.
    """
    pass


def load_tokenizer(model_path: str) -> PreTrainedTokenizer:
    """
    Load the tokenizer from the model_path.
    """
    pass


def predict_concept(model: PreTrainedModel,
                    tokenizer: PreTrainedTokenizer,
                    texts: List[str]) -> Tuple[List[List[float]], List[int]]:
    """
    Write a function that predicts the concept values for a list of texts.
    Use the model and predict two values for each concept:
    concept_scores (which is the distribution over the labels, if the concept is food, the column will be food_scores),
    and concept_pred (which is the argmax of concept_scores).
    """
    pass


def predict_concepts(concept_models: Dict[str, str],
                     text_col: str,
                     df: pd.DataFrame) -> pd.DataFrame:
    """
    concept_models is a dict with the concept name as key and the model path as value.
    For each concept in the concept_models, predict the concept values for the df.
    Add these columns for each concept:
    concept_scores (which is the distribution over the labels, if the concept is food, the column will be food_scores),
    and concept_pred (which is the argmax of concept_scores).
    """
    pass


def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def load_df(path: str) -> pd.DataFrame:
    """
    write a function that loads the dataframes.
    Tip: use df[concept_scores] = [eval(x) for x in df[concept_scores]]
    """
    pass


def filter_df(df: pd.DataFrame, concept: str, concept_val: str) -> pd.DataFrame:
    """
    Write a function that filters a dataframe by a concept and its value (e.g., food=0).
    """
    pass


def merge_original_and_cf(original_df: pd.DataFrame, cf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Write a function that merges the original and CF dataframes into the following schema:
    split, original_id, text, [concepts], [concepts_scores/preds]
    cf_id, cf_text, cf_concept, cf_old_val, cf_new_val, [cf_concepts], [cf_concepts_scores/preds]
    The cf_concept (cf = countefactual), is the concept that we change in the example, cf_old_val is the old value, and we change it to cf_new_val.
    cf_concepts is a list of the concepts of the cf (the values should be the same as [concepts], but sometimes the cf generation makes mistakes).
    Notice that for each example_id and intervention (e.g., cf_concept: cf_old_val -> cf_new_val), there might be several rows, indicating multiple CFs.

    The function takes as input an original_df and a cf_df and merges (join) them.
    Tip: rename the columns before doing the pd.merge/pd.join.
    """
    pass


def compute_icace_scores(original_df: pd.DataFrame, cf_df: pd.DataFrame,
                         concept: str, old_val: str, new_val: str) -> pd.DataFrame:
    """
    Write a function that, given the original and CF dataframes, an intervention (concept, old_val, new_val),
    and an outcome concept, calculates the ICaCE scores.
    The ICaCE scores are the difference between the CF’s outcome_concept_scores
    and the original’s outcome_concept_scores.
    Tip: use the functions from the previous step, filter the original and CF dataframes according to the intervention,
     merge the dataframes, and calculate the ICaCE scores.
    The function should return a new dataframe with two columns: original_id, icace.
    Notice: There should be one icace score for each original_id.
    If there are multiple CFs for a given original_id, you should use the mean icace vector.
    """
    pass


def prepare_icace_df_path(outputs_folder: str, outcome_concept: str, treatment_concept: str,
                          old_val: str, new_val: str) -> str:
    """
    Write a function that saves the icace dataframe.
    The file name should indicate the intervention and the outcome concept
    (i.e., it should be easy to extract the concept, old_val, new_val, and outcome from the file name/path).
    """
    pass


def extract_concept_vars_from_path(path: str) -> Tuple[str, str, str, str]:
    """
    Write a function that extracts the outcome concept, treatment concept, old_val, new_val from the file name/path.
    """
    pass


def compute_l1_error(gold_icace: List[float], method_icace: List[float]) -> float:
    """
    Write a function that computes the L1 error between the gold ICaCE and the method ICaCE, for a single example.
    """
    pass


def compute_cos_error(gold_icace: List[float], method_icace: List[float]) -> float:
    """
    Write a function that computes the cosine similarity between the gold ICaCE and the method ICaCE, for a single example.
    """
    pass


def compute_nd_error(gold_icace: List[float], method_icace: List[float]) -> float:
    """
    Write a function that computes the normalized distance between the gold ICaCE and the method ICaCE, for a single example.
    """
    pass


def compute_order_faithfulness_score(gold_icaces: Dict[str, List[float]],
                                     method_icacess: Dict[str, List[float]]) -> float:
    """
    Write a function that computes the order faithfulness score.
    The gold_icaces and method_icacess are dictionaries with the intervention (f"{concept}:{old_val}->{new_val}")
    as keys and the ICaCE scores as values.
    ** I did not debug this function, it might contain errors - you should debug it **
    """
    interventions = list(set(gold_icaces.keys()).intersection(set(method_icacess.keys())))
    n_labels = len(gold_icaces.get(interventions[0]))
    scores = []
    for label in range(n_labels):
        correct_order, total_comp = 0, 0
        gold_order = [(gold_icaces[intervention][label], intervention) for intervention in interventions]
        method_order = [(method_icacess[intervention][label], intervention) for intervention in interventions]
        gold_order = {intervention: i
                      for i, (intervention, icace) in enumerate(sorted(gold_order, key=lambda x: x[0]))}
        method_order = {intervention: i
                        for i, (intervention, icace) in enumerate(sorted(method_order, key=lambda x: x[0]))}
        for i, inter_1 in enumerate(interventions):
            for inter_2 in interventions[i+1:]:
                if gold_order[inter_1] < gold_order[inter_2] and method_order[inter_1] < method_order[inter_2]:
                    correct_order += 1
                elif gold_order[inter_1] > gold_order[inter_2] and method_order[inter_1] > method_order[inter_2]:
                    correct_order += 1
                total_comp += 1
        scores.append(correct_order / total_comp)
    return np.mean(scores)


def evaluate_method(gold_icace_dfs: Dict[str, pd.DataFrame],
                    method_icace_dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Write a function that, given two dicts (their keys are intervention,
    and the corresponding value is an icace dataframe), one dict contains the gold icaces,
     and another dict contains the method icaces.
    The function returns the average estimation errors (average L1, average Cos, average ND) over all interventions
    and the average order-faithfulness score (implemented above).
    For example, in CEBaB, there are 24 interventions. There are 4 treatment concepts
     (food, ambiance, service, noise). Every concept has 3 values (positive, negative, unknown),
      so every concept has 6 interventions.
    The input dicts should have 24 keys.
    """
    pass


def random_match_baseline(test_df: pd.DataFrame,
                          match_df: pd.DataFrame,
                          concept: str,
                          old_val: str,
                          new_val: str,
                          top_k: int = 1,
                          seed: int = 42) -> pd.DataFrame:
    """
    Write a function that creates a random match baseline.
    The function should return a dataframe with the following columns:
    original_id, cf_id, rank (for the top_k)
    """
    pass


def approx_baseline(test_df: pd.DataFrame,
                    match_df: pd.DataFrame,
                    concept: str,
                    old_val: str,
                    new_val: str,
                    confounders: List[str],
                    top_k: int = 1) -> pd.DataFrame:
    """
    Write a function that creates an approximate baseline.
    The function should return a dataframe with the following columns:
    original_id, cf_id, rank (for the top_k)
    Notice: the approx baseline should use the concept_pred columns for the confounder concepts.
    """
    pass


def find_top_candidates(original_embeddings: Union[np.ndarray, torch.Tensor],
                        cf_embeddings: Union[np.ndarray, torch.Tensor],
                        top_k: int,
                        ranking_metric: str = 'cos') -> List[List[int]]:
    """
    This function should return the top_k candidates according to the ranking_metric.
    The ranking_metric can be 'cos' for cosine similarity or 'l1' for L1 distance (you can implement more metrics).
    The function returns a list of lists, where each list contains the indices of the top_k candidates.
    """
    # we start by converting the embeddings to tensors, the device should be the same for both tensors
    if isinstance(original_embeddings, torch.Tensor):
        device = original_embeddings.device
    elif isinstance(cf_embeddings, torch.Tensor):
        device = cf_embeddings.device
    else:
        device = 'cpu'
    if isinstance(original_embeddings, np.ndarray):
        original_embeddings = torch.tensor(original_embeddings, device=device)
    if isinstance(cf_embeddings, np.ndarray):
        cf_embeddings = torch.tensor(cf_embeddings, device=device)
    if ranking_metric == 'cos':
        # compute the cosine similarity
        similarity = torch.nn.functional.cosine_similarity(original_embeddings.unsqueeze(0),
                                                           cf_embeddings.unsqueeze(0), dim=-1)
        # sort the similarity
        _, top_k_indices = torch.topk(similarity, top_k, largest=True)
    elif ranking_metric == 'l1':
        # compute the L1 distance
        distance = torch.cdist(original_embeddings.unsqueeze(0), cf_embeddings.unsqueeze(0), p=1)
        # sort the distance
        _, top_k_indices = torch.topk(-distance, top_k, largest=True)
    else:
        raise ValueError(f"Unknown ranking metric {ranking_metric}")
    return top_k_indices.cpu().numpy().tolist()


def propensity_score_baseline(test_df: pd.DataFrame,
                              match_df: pd.DataFrame,
                              concept: str,
                              old_val: str,
                              new_val: str,
                              top_k: int = 1) -> pd.DataFrame:
    """
    Write a function that creates a propensity score baseline.
    The function should return a dataframe with the following columns:
    original_id, cf_id, rank (for the top_k)
    Notice: the propensity distance according to which the candidates are ranked
     is computed over the treatment==concept (concept_pred) score P(concept_pred | example)
     you can use the find_top_candidates function to find the top candidates.
    """
    pass


def representation_match(test_df: pd.DataFrame,
                         match_df: pd.DataFrame,
                         model: PreTrainedModel,
                         tokenizer: PreTrainedTokenizer,
                         concept: str,
                         old_val: str,
                         new_val: str,
                         text_col: str,
                         rank_metric: str = 'cos',
                         top_k: int = 1) -> pd.DataFrame:
    """
    Write a function that creates a representation match baseline.
    Tip: you should first filter the dfs according to the intervention,
     then, extract the embeddings for the original and CF examples and
     prepare a df with the example_id, embeddings.
    Then, call the find_top_candidates function to find the top candidates and fix the example_ids
    """
    pass



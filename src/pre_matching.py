import json
import pandas as pd
import numpy as np
from typing import Dict

def load_pre_match_dict(jsonl_path: str) -> Dict[str, str]:
    """
    Reads a finetuning JSONL file and returns a dictionary mapping
    prompt_occupation -> completion.

    Args:
        jsonl_path (str): Path to the finetune.jsonl file.

    Returns:
        dict: A dictionary where keys are the prompt occupations (in lowercase),
              and values are the corresponding 'completion' strings.
    """
    mapping = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            prompt_occ = entry["prompt_occupation"].strip().lower()
            completion_val = entry["completion"].strip()
            mapping[prompt_occ] = completion_val
    return mapping


def pre_match_occupation(
    df: pd.DataFrame,
    occupation_column: str,
    pre_match_dict: Dict[str, str],
    output_col: str = "final_output",
    method_col: str = "method"
) -> pd.DataFrame:
    """
    Performs a dictionary-based lookup for each row in the specified occupation column.
    If there's an exact match (case-insensitive) in pre_match_dict, sets:
        final_output = matched 'completion'
        method = 'pre_match'
    Otherwise, sets both columns to NaN.

    Args:
        df (pd.DataFrame): The input DataFrame containing at least one occupation column.
        occupation_column (str): Name of the column with occupation data.
        pre_match_dict (Dict[str, str]): Dictionary mapping occupation -> classification.
        output_col (str, optional): Name of the new column for the final classification.
        method_col (str, optional): Name of the new column for the method label.

    Returns:
        pd.DataFrame: The original DataFrame with two new columns appended.
    """
    # Create empty lists to store the results
    final_output_list = []
    method_list = []

    for _, row in df.iterrows():
        occ_value = row[occupation_column]

        # If the occupation is missing or NaN, leave results as NaN
        if pd.isna(occ_value):
            final_output_list.append(np.nan)
            method_list.append(np.nan)
            continue

        # Convert to string and lowercase
        occ_str_lower = str(occ_value).lower()

        # Attempt to match in the dictionary
        if occ_str_lower in pre_match_dict:
            final_output_list.append(pre_match_dict[occ_str_lower])
            method_list.append("pre_match")
        else:
            final_output_list.append(np.nan)
            method_list.append(np.nan)

    # Append columns to the original DataFrame
    df[output_col] = final_output_list
    df[method_col] = method_list

    return df

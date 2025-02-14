"""Utils and functions to be used by many metrics in main."""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


def plot_profile(profile_json: dict):
    """Plots the barplot with the complexity measures.

    Args:
        profile_json (dict): Metrics and values.
    """
    sns.set_theme(style="whitegrid")

    # Initialize the matplotlib figure
    _, ax = plt.subplots(figsize=(30, 5))

    # Plot the total crashes
    sns.set_color_codes("pastel")
    sns.barplot(data=profile_json, color="b")

    # Add a legend and informative axis label
    ax.set(ylabel="", xlabel="Complexity metrics")
    ax.set(ylim=(0.0, 1.0))
    sns.despine(left=True, bottom=True)


def ovo(data):
    """One-vs-one takes the data in pairs.

    Args:
        data (DataFrame): Data with class column informed

    Returns:
        list: Binary list indexing the two sub-groups
    """
    return [
        data[data["class"].isin(combo)]
        for combo in combinations(data["class"].unique(), 2)
    ]


def normalize(df: pd.DataFrame):
    """Normalization of data.

    Args:
        df (pd.DataFrame): Dataframe, all numeric.

    Returns:
        pd.DataFrame: Normalizar dataset
    """
    return (df - df.mean()) / df.std()


def adherence(adh):
    """_summary_.

    Args:
        adh (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = []
    h = []
    while adh.shape[0] > 0:
        aux = np.argmax(np.sum(adh, axis=1))
        tmp = np.where(adh[aux])[0]
        dif = np.setdiff1d(np.arange(adh.shape[0]), np.append(tmp, aux))
        adh = adh[dif][:, dif]

        if adh.shape[0] > 0:
            h.append(len(tmp))
        else:
            h.append(1)

        n.append(aux)

    return h, n


def binarize(data: pd.DataFrame):
    """Creates a binarized version for the given dataset.

    Args:
        data (pd.DataFrame): Dataset to be binarized

    Returns:
        DataFrame: _description_
    """
    categorical_cols = data.select_dtypes(
        include=["object", "category"]
    ).columns
    if not categorical_cols.empty:
        enc = OneHotEncoder(handle_unknown="ignore")
        encoded = enc.fit_transform(data[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded, columns=enc.get_feature_names_out(categorical_cols)
        )
        data = pd.concat(
            [data.drop(columns=categorical_cols), encoded_df], axis=1
        )
    return data


import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import expit
from scipy.optimize import minimize
from math import log
import re

tqdm.pandas()


STYLE_CONTROL_ELEMENTS = [
    "len_answer",
    "header_count",
    "list_count",
    "bold_count",
    "code_blocks_count"
]

DIFF_MASK = np.array([1.0, -1.0], dtype=np.float64)

def count_style_elements(markdown_text):
    def remove_pattern(answer, pattern):
        blocks = pattern.findall(answer)
        for block in blocks:
            answer = answer.replace(block, "")
        return answer

    len_answer = len(markdown_text)
    code_count = len(re.findall(r"```[^`]+```", markdown_text))
    code_pattern = re.compile("```([^`]+)```")
    markdown_text = remove_pattern(markdown_text, code_pattern)
    markdown_text = markdown_text.replace("```", "")

    mono_count = len(re.findall(r"`[^`]+`", markdown_text))
    mono_pattern = re.compile("`([^`]+)`")
    markdown_text = remove_pattern(markdown_text, mono_pattern)
    counters = {
        f"len_answer": len_answer,
        f"header_count": {
            "h1": len(re.findall(r"^#{1}\s", markdown_text, re.MULTILINE)),
            "h2": len(re.findall(r"^#{2}\s", markdown_text, re.MULTILINE)),
            "h3": len(re.findall(r"^#{3}\s", markdown_text, re.MULTILINE)),
            "h4": len(re.findall(r"^#{4}\s", markdown_text, re.MULTILINE)),
            "h5": len(re.findall(r"^#{5}\s", markdown_text, re.MULTILINE)),
            "h6": len(re.findall(r"^#{6}\s", markdown_text, re.MULTILINE)),
        },
        f"list_count": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", markdown_text, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", markdown_text, re.MULTILINE)),
        },
        f"bold_count": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", markdown_text)),
            "__": len(re.findall(r"__[^_\n]+__", markdown_text)),
        },
        f"code_blocks_count": {
            "`": mono_count,
            "```": code_count,
        },
    }
    return counters


def extract_style_feature(x, feature):
    val = x[feature]
    if isinstance(val, int):
        return val
    else:
        return sum(val.values())


def get_element_counts(text):
    style_elements = count_style_elements(text)
    el_counts = []
    for feature in style_elements:
        el_counts.append(extract_style_feature(style_elements, feature))
    return el_counts


def calculate_style(
    model_a: pd.Series,
    model_b: pd.Series,
    style_elements: list[str]=STYLE_CONTROL_ELEMENTS
):
    n_features = len(style_elements)
    n_battles = model_a.shape[0]
    style_matrix = np.zeros(shape=(2*n_features, n_battles))
    for idx, element in enumerate(style_elements):
        style_matrix[idx, :] = np.array([el[idx] for el in model_a])
    for idx, element in enumerate(style_elements):
        style_matrix[n_features + idx, :] = np.array([el[idx] for el in model_b])
    style_diff = (style_matrix[:n_features] - style_matrix[n_features]).astype(float)
    style_sum = (style_matrix[:n_features] + style_matrix[n_features]).astype(float)

    style_diff /= style_sum

    style_mean = np.mean(style_diff, axis=1)
    style_std = np.std(style_diff, axis=1)
    features = ((style_diff - style_mean[:, np.newaxis]) / style_std[:, np.newaxis]).T

    return features


def get_matchups_models(model_a: pd.Series, model_b: pd.Series):
    n_rows = len(model_a)
    assert len(model_b) == n_rows
    model_indices, models = pd.factorize(pd.concat([model_a, model_b]))
    matchups = np.column_stack([model_indices[:n_rows], model_indices[n_rows:]])
    return matchups, models.to_list()


def contextual_bt_loss_and_grad(
    params,
    n_competitors,
    matchups,
    features,
    outcomes,
    alpha=1.0,
    reg=1.0,
    half_reg=0.5,
):
    reg_loss = half_reg * np.inner(params, params)

    ratings = params[:n_competitors]
    feature_params = params[n_competitors:]

    matchup_ratings = ratings[matchups]

    bt_logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    context_logits = np.dot(features, feature_params)
    probs = expit(bt_logits + context_logits)
    loss = (
        -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes))).sum()
        + reg_loss
    )

    error = outcomes - probs
    grad = reg * params
    matchups_grads = -alpha * error
    np.add.at(
        grad[:n_competitors], matchups[:, [0, 1]], matchups_grads[:, None] * DIFF_MASK
    )
    grad[n_competitors:] -= np.dot(features.T, error)

    return loss, grad, expit(context_logits)


def fit_contextual_bt(
    matchups,
    features,
    outcomes,
    models,
    idxs=None,
    alpha=log(10.0),
    reg=0.5,
    tol=1e-6,
):
    n_features = features.shape[1]
    n_models = len(models)
    initial_params = np.zeros(n_models + n_features, dtype=np.float64)
    half_reg = reg / 2.0

    if idxs is not None:
        matchups, features, outcomes = matchups[idxs], features[idxs], outcomes[idxs]

    result = minimize(
        fun=contextual_bt_loss_and_grad,
        x0=initial_params,
        args=(n_models, matchups, features, outcomes, alpha, reg, half_reg),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    loss, grad, context_logits = contextual_bt_loss_and_grad(result["x"], n_models, matchups, features, outcomes, alpha, reg, half_reg)
    return result["x"], context_logits


def compute_style_control(
    df: pd.DataFrame,
    alpha=log(10.0), reg=0.5, tol=1e-6
):
    features = calculate_style(df.model_a_style, df.model_b_style)
    matchups, models = get_matchups_models(df.model_a, df.model_b)
    outcomes = df.winner.values
    params, context_logits = fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models=models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )
    ratings = params[: len(models)]
    weigths = params[len(models):]
    return ratings, models, context_logits

def scale_and_offset(
    ratings,
    models=[],
    baseline_model='',
    scale=400,
    init_rating=1000,
    baseline_rating=1114,
):
    """convert ratings from the natural scale to the Elo rating scale with an anchored baseline"""
    scaled_ratings = (ratings * scale) + init_rating
    if baseline_model and models and baseline_model in models:
        baseline_idx = models.index(baseline_model)
        scaled_ratings += baseline_rating - scaled_ratings[..., [baseline_idx]]
    return scaled_ratings
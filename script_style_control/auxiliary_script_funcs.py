import json
from typing import List, Dict, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from evalica import bradley_terry, Winner, pairwise_frame
from scipy.special import expit
from style_funcs import compute_style_control, scale_and_offset
tqdm.pandas()


def save_battles_to_json(filename: str, results: List[Dict]=[]):
    if not results:
        return
    if not filename:
        filename = "battles_saved_file.json"
    try:
      with open(filename, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for i, item in enumerate(results):
            line = json.dumps(item, ensure_ascii=False)
            if i < len(results) - 1:
                line += ','
            f.write(f'  {line}\n')
        f.write(']')
    except Exception as e:
        print(f"Error with file {filename}: {e}.")
    else:
        print(f"Data was stored to {filename}.")
        

def _prepare_data(
    results: List[Dict],
    len_control: bool=False,
    style_control: bool=True,
    pen_only_model: bool=False
  ) -> list:

    if style_control:
        for r in results:
            if 'styles' not in r:
                raise ValueError("If the 'style_control' mode is on, the data should contain information about the style of model answers.")

    if len_control:
        for r in results:
            if 'lens' not in r:
                if 'styles' in r:
                    r['lens'] = {'model': r['styles']['model'][0], 'reference': r['styles']['reference'][0]}
                else:
                    raise ValueError("If the 'len_control' mode is on, the data should contain information about the length of model answers.")

    full_hash =  ['|'.join([str(r['id']), r['model_name'], r['reference_model_name']]) for r in results]
    model_hash = ['|'.join([str(r['id']), r['model_name']]) for r in results]
    reference_hash = ['|'.join([str(r['id']), r['reference_model_name']]) for r in results]

    data = []

    df = pd.DataFrame()
    df['p'] = [r['p'] for r in results]
    df['full_hash'] = full_hash
    df['model_hash'] = model_hash
    df['reference_hash'] = reference_hash

    df['model_name'] = [r['model_name'] for r in results]
    df['reference_model_name'] = [r['reference_model_name'] for r in results]

    if style_control:
        df['model_style'] = [np.array(r['styles']['model']) for r in results]
        df['reference_style'] = [np.array(r['styles']['reference']) for r in results]
    elif len_control:
        df['model_len'] = [r['lens']['model'] for r in results]
        df['reference_len'] = [r['lens']['reference'] for r in results]

        answer_len_deltas = {}
        for ref_model_name, group in df.groupby('reference_model_name'):
            answer_len_deltas[ref_model_name] = (group['reference_len'] - group['model_len']).std()

    for _, group in df.groupby('model_hash'):
        for _, subgroup in group.groupby('full_hash'):
            # assert subgroup.shape[0] == 2
            if (subgroup['model_name'] == subgroup['reference_model_name']).all():
                continue
            pred = int(subgroup['p'].mean() >= 0.5)
            if style_control:
                data.append([subgroup['model_name'].tolist()[0], subgroup['reference_model_name'].tolist()[0], pred, subgroup['model_style'].tolist()[0], subgroup['reference_style'].tolist()[0]])
            else:
                normalized_answer_delta_weight = 0.5
                if len_control and (pred or not pen_only_model):
                    normalized_answer_delta_weight = 0.5
                    answers_length_deltas_std = answer_len_deltas[subgroup['reference_model_name'].iloc[0]]
                    answer_length_delta = (subgroup['reference_len'] - subgroup['model_len']).iloc[0]
                    if answer_length_delta != 0: # same model as ref
                        normalized_answer_delta_weight = expit(answer_length_delta / answers_length_deltas_std)
                data.append([subgroup['model_name'].tolist()[0], subgroup['reference_model_name'].tolist()[0], pred, normalized_answer_delta_weight])
    return data


def right_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    if df.empty:
        return False
    for c in columns:
        if c not in df.columns:
            return False
    return True


def get_models_from_data(df: pd.DataFrame) -> list[str]:
    if right_columns(df, ['model_a', 'model_b']):
        return pd.concat([df.model_a, df.model_b]).unique().tolist()
    return []


def get_results_from_file(filename) -> List[Dict]:
    with open(filename, 'r', encoding='utf-8') as file:
        results: List[Dict] = json.load(file)
        return results


def _calculate_mean_scores(df: pd.DataFrame) -> pd.DataFrame:
    # taken from https://github.com/VikhrModels/ru_llm_arena/blob/master/show_result.py
    if not right_columns(df, ['model_a', 'model_b', 'winner', 'answer_deltas']):
        print("Error in dataframe while computing mean scores.")
        return pd.DataFrame()
    df['winner'] = df['winner'].map({
            1: Winner.X,
            0: Winner.Y
    })
    result = bradley_terry(
        df['model_a'],
        df['model_b'],
        df['winner'],
        weights=df['answer_deltas'],
        tolerance=1e-8
    )
    df = pairwise_frame(result.scores)
    np.fill_diagonal(df.values, np.nan)
    return df


def _calculate_ratings(
    results: List[Dict]=[],
    style_control: bool=True,
    len_control: bool=False,
    pen_only_model: bool=False,
    get_mean_scores: bool=True,
    get_elo_ratings: bool=True,
    model: str='',
  ) -> Dict:
    '''
    На вход подаются
    results = [
        {
            'p': <model_proba_i>,
            'id': <id_i>,
            'model_name': <model_name_i>,
            'reference_model_name': <reference_model_name_i>,
            'styles': {
                'model':      [x, x, x, x, x],
                'reference':  [x, x, x, x, x], # [ len_answer, header_count, list_count, bold_count, code_blocks_count ]
            }
        } for i in range(2*n*m) # на каждый instruction было по 2 sample
    ]
    '''
    if not results:
        raise ValueError("No data for computing!")
    data = _prepare_data(results, len_control=len_control, style_control=style_control, pen_only_model=pen_only_model)
    if not data:
        raise ValueError("No data for computing!")
    scores = {'elo': {}, 'mean_scores': {}}
    if style_control:
        df = pd.DataFrame(data, columns=['model_a', 'model_b', 'winner', 'model_a_style', 'model_b_style'])
    else:
        df = pd.DataFrame(data, columns=['model_a', 'model_b', 'winner', 'answer_deltas'])
    models = get_models_from_data(df)
    if model and model not in models:
        raise ValueError(f"Model {model} is not in dataset!")
    if style_control:
        ratings, models, context_logits = compute_style_control(df)
        df['answer_deltas'] = context_logits
        scaled_ratings = scale_and_offset(ratings)
        if get_elo_ratings:
            for i in range(len(models)):
                scores['elo'][models[i]] = scaled_ratings[i]
    if get_mean_scores:
        df = _calculate_mean_scores(df)
        if df.empty:
            print(scores)
            print('Error in calculating mean scores.')
        if model:
            scores['mean_scores'][model] = df.loc[model].mean()
        else:
            for m in models:
                scores['mean_scores'][m] = df.loc[m].mean()
    return scores


def get_ratings(
    source_files: Union[str, list[str]],
    model: str='',
    save_all_battles: bool=True,
    newfilename: str='',
    answerfile: str='',
    load_results: bool=True,
    print_results: bool=True,
    style_control: bool=True,
    len_control: bool=False,
    no_control: bool=False,
    pen_only_model: bool=False,
    get_mean_scores: bool=True,
    get_elo_ratings: bool=True
):
    if not load_results and not print_results:
        raise ValueError("No task!")
    if not source_files:
        raise ValueError("No data available for analysis.")
    results = []
    if isinstance(source_files, str):
        source_files = [source_files]
    if isinstance(source_files, list):
        read_files_count = 0
        for file in source_files:
            try:
                results += get_results_from_file(file)
            except (FileNotFoundError, IsADirectoryError):
                print(f"File {file} was not found or is empty.")
            except json.JSONDecodeError:
                print(f"Error: {file} is not JSON-format.")
            except Exception as e:
                print(f"Error with file {file}: {e}")
            else:
                read_files_count += 1
        if read_files_count == len(sourse_files):
            print("All data has been received.")
    if not results:
        raise ValueError("No data for computing!")
    if save_all_battles:
        save_battles_to_json(newfilename, results)
    if not (style_control or len_control or no_control) and not (get_elo_ratings or get_mean_scores):
        raise ValueError("No option selectes.")
    if no_control:
        style_control, len_control = False, False
        get_elo_ratings = False
        get_mean_scores = True
    if style_control:
        len_control = False
    if len_control:
        get_elo_ratings = False
        get_mean_scores = True
        style_control = False
    scores = _calculate_ratings(
        results=results,
        model=model,
        style_control=style_control,
        len_control=len_control,
        pen_only_model=pen_only_model,
        get_mean_scores=get_mean_scores,
        get_elo_ratings=get_elo_ratings)
    if not get_elo_ratings:
        scores.pop("elo", None)
    if not get_mean_scores:
        scores.pop("mean_scores", None)
    if print_results:
        if get_elo_ratings:
            print('ELO RATINGS:')
            for m in scores['elo']:
                print(f"{str(m):40} | {scores['elo'][m]}")
            print()
        if get_mean_scores:
            print('MEAN SCORES:')
            for m in scores['mean_scores']:
                print(f"{str(m):40} | {scores['mean_scores'][m]}")
            print()
    if load_results:
        if not answerfile:
            answerfile = 'results.json'
        with open(answerfile, "w", encoding="utf-8") as file:
            json.dump(scores, file, ensure_ascii=False, indent=4)
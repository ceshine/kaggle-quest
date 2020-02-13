import logging
from pathlib import Path

import fire
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from scipy.special import expit
from transformers import AutoTokenizer, RobertaConfig

from .models import DualRobertaModel
from .metrics import SpearmanCorr
from .prepare_tfrecords import Preprocessor, OUTPUT_COLUMNS, INPUT_COLUMNS
from .inference import ROBERTA_CONFIG, get_batch
from .post_processing import find_best_bins


def eval_fold(
    input_path: str = "data/",
    fold_path: str = "cache/tfrecords/fold_0.jl",
    model_path: str = "cache/roberta-base-fold-0.h5",
    tokenizer_path: str = "cache/tfrecords/tokenizer_roberta-base/",
    batch_size: int = 8
):
    df_train = pd.read_csv(input_path + 'train.csv')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    processor = Preprocessor(tokenizer)
    labels = df_train.loc[:, OUTPUT_COLUMNS].values
    inputs = df_train.loc[:, INPUT_COLUMNS].values
    _, valid_idx = joblib.load(fold_path)
    # valid_idx = valid_idx[:100] # For faster debug
    labels, inputs = labels[valid_idx], inputs[valid_idx]
    tmp = []
    for i in tqdm(range(labels.shape[0]), ncols=100):
        tmp.append(processor.process_one_example(
            inputs[i, 0],
            inputs[i, 1],
            inputs[i, 2])
        )
    processed_inputs = np.array(tmp)
    del tmp, inputs

    model_name = Path(model_path).name
    if model_name.lower().startswith("roberta-base"):
        config = RobertaConfig.from_dict(
            ROBERTA_CONFIG)
        model = DualRobertaModel(
            model_name="roberta-base", config=config, pretrained=False
        )
        # build
        model(get_batch(processed_inputs[:2]), training=False)
        model.load_weights(model_path)
    else:
        raise ValueError("Unknown model.")
    spearman = SpearmanCorr()

    @tf.function
    def predict_batch(inputs):
        return model(inputs, training=False)[0]

    preds = []
    for i in tqdm(range(0, len(labels), batch_size), ncols=100):
        input_dicts = processed_inputs[i:i+batch_size]
        preds.append(predict_batch(get_batch(input_dicts)).numpy())
    preds = np.concatenate(preds)

    score = spearman(labels, preds)[0] * -1
    print(f"Raw Spearman: {score * 100 : .2f}")
    return labels, preds


def eval_folds(
    n_folds: int = 5,
    input_path: str = "data/",
    fold_pattern: str = "cache/tfrecords/fold_%d.jl",
    model_pattern: str = "cache/roberta-base-fold-%d.h5",
    tokenizer_path: str = "cache/tfrecords/tokenizer_roberta-base/",
    batch_size: int = 8
):
    if Path("cache/oof.jl").exists():
        labels, preds = joblib.load("cache/oof.jl")
    else:
        labels, preds = [], []
        for fold in range(n_folds):
            labels_tmp, preds_tmp = eval_fold(
                input_path, fold_pattern % fold,
                model_pattern % fold,
                tokenizer_path,
                batch_size
            )
            labels.append(labels_tmp)
            preds.append(preds_tmp)

        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        joblib.dump([labels, preds], "cache/oof.jl")
    spearman = SpearmanCorr()
    score = spearman(labels, preds)[0] * -1
    print(f"Raw Spearman: {score * 100 : .2f}")
    best_score, best_bins, scaler = find_best_bins(labels, expit(preds))
    print(f"Optimized Spearman: {best_score * 100 : .2f}")
    print(best_bins)
    joblib.dump([best_bins, scaler], "cache/best_bins.jl")


if __name__ == '__main__':
    fire.Fire(eval_folds)

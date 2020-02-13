from pathlib import Path

import fire
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from scipy.special import expit
from transformers import AutoTokenizer, RobertaConfig
from tf_helper_bot.utils import prepare_tpu

from .models import DualRobertaModel
from .metrics import SpearmanCorr
from .prepare_tfrecords import Preprocessor, OUTPUT_COLUMNS, INPUT_COLUMNS
from .inference import ROBERTA_CONFIG, get_batch
from .post_processing import find_best_bins
from .dataset import tfrecord_dataset


def eval_fold(
    valid_path: str,
    model_path: str = "cache/roberta-base-fold-0.h5",
    batch_size: int = 8
):
    strategy, tpu = prepare_tpu()
    if tpu:
        batch_size *= strategy.num_replicas_in_sync
    valid_ds, valid_steps = tfrecord_dataset(
        valid_path, batch_size, strategy, is_train=False)
    valid_dist_ds = strategy.experimental_distribute_dataset(
        valid_ds)

    model_name = Path(model_path).name
    if model_name.lower().startswith("roberta-base"):
        config = RobertaConfig.from_dict(
            ROBERTA_CONFIG)
        model = DualRobertaModel(
            model_name="roberta-base", config=config, pretrained=False
        )
        # build
        model(next(iter(valid_ds))[0], training=False)
        model.load_weights(model_path)
    else:
        raise ValueError("Unknown model.")
    spearman = SpearmanCorr()

    @tf.function
    def predict_batch(inputs):
        return model(inputs, training=False)[0]

    preds, labels = [], []
    for batch_, labels_ in tqdm(valid_dist_ds, total=valid_steps, ncols=100):
        tmp = strategy.experimental_run_v2(
            predict_batch,
            args=(batch_,)
        ).values
        preds.append(
            tf.concat(
                tmp, axis=0
            ).numpy()
        )
        labels.append(tf.concat(
            strategy.experimental_local_results(labels_),
            axis=0
        ).numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    score = spearman(labels, preds)[0] * -1
    print(f"Raw Spearman: {score * 100 : .2f}")
    return labels, preds


def eval_folds(
    n_folds: int = 5,
    valid_pattern: str = "gs://ceshine-colab-tmp-2/quest/valid-%d-*.tfrec",
    model_pattern: str = "cache/roberta-base-fold-%d.h5",
    batch_size: int = 8
):
    if Path("cache/oof.jl").exists():
        labels, preds = joblib.load("cache/oof.jl")
    else:
        labels, preds = [], []
        for fold in range(n_folds):
            matches = list(tf.io.gfile.glob(valid_pattern % fold))
            assert len(matches) == 1
            labels_tmp, preds_tmp = eval_fold(
                matches[0],
                model_pattern % fold,
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

import glob
import logging
from pathlib import Path

import fire
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from scipy.special import expit
from transformers import AutoTokenizer
from transformers import RobertaConfig

from .models import PoolingBertModel, DualRobertaModel
from .prepare_tfrecords import Preprocessor, INPUT_COLUMNS, OUTPUT_COLUMNS
from .post_processing import prevent_nan

ROBERTA_CONFIG = {
    "architectures": [
        "RobertaForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.1,
    "finetuning_task": None,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1"
    },
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "is_decoder": False,
    "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1
    },
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": 514,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "num_labels": 30,
    "output_attentions": False,
    "output_hidden_states": False,
    "output_past": True,
    "pruned_heads": {},
    "torchscript": False,
    "type_vocab_size": 1,
    "use_bfloat16": False,
    "vocab_size": 50265
}


def get_batch(input_dicts):
    return {
        "input_ids_question": tf.convert_to_tensor(np.stack([
            x["input_ids_question"] for x in input_dicts
        ], axis=0)),
        "attention_mask_question": tf.convert_to_tensor(np.stack([
            x["input_mask_question"] for x in input_dicts
        ], axis=0)),
        "input_ids_answer": tf.convert_to_tensor(np.stack([
            x["input_ids_answer"] for x in input_dicts
        ], axis=0)),
        "attention_mask_answer": tf.convert_to_tensor(np.stack([
            x["input_mask_answer"] for x in input_dicts
        ], axis=0)),
    }


def main(
    input_path: str = "data/",
    tokenizer_path: str = "cache/tfrecords/tokenizer_roberta-base/",
    model_path_pattern: str = "cache/roberta-base-fold-*",
    best_bins_path: str = "cache/best_bins.jl",
    batch_size: int = 8, progress_bar: bool = True,
    add_sigmoid: bool = False, rank: bool = False
):
    df_valid = pd.read_csv(input_path + 'test.csv')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    processor = Preprocessor(tokenizer)
    inputs = df_valid.loc[:, INPUT_COLUMNS].values
    tmp = []
    for i in tqdm(range(inputs.shape[0]), ncols=100, disable=not progress_bar):
        tmp.append(processor.process_one_example(
            inputs[i, 0],
            inputs[i, 1],
            inputs[i, 2])
        )
    processed_inputs = np.array(tmp)
    del tmp, inputs

    buffer = []
    for model_path in glob.glob(model_path_pattern):
        model_name = Path(model_path).name
        print(model_path, model_name)
        if model_name.lower().startswith("bert"):
            model = PoolingBertModel.from_pretrained(model_path)
        elif model_name.lower().startswith("roberta-base"):
            config = RobertaConfig.from_dict(
                ROBERTA_CONFIG)
            model = DualRobertaModel(
                model_name="roberta-base", config=config, pretrained=False)
            # build
            model(get_batch(processed_inputs[:2]), training=False)
            model.load_weights(model_path)
        else:
            raise ValueError("Unknown model.")

        @tf.function
        def predict_batch(inputs):
            return model(inputs, training=False)[0]

        preds = []
        for i in tqdm(range(
            0, len(processed_inputs), batch_size
        ), ncols=100, disable=not progress_bar):
            input_dicts = processed_inputs[i:i+batch_size]
            preds.append(predict_batch(get_batch(input_dicts)).numpy())
        if add_sigmoid and not rank:
            buffer.append(expit(np.concatenate(preds)))
        elif rank:
            tmp = np.concatenate(preds)
            buffer.append(
                tmp.argsort(axis=0).argsort(axis=0) / tmp.shape[0]
            )
        else:
            buffer.append(np.concatenate(preds))

    final_preds = np.mean(buffer, axis=0)
    if add_sigmoid and not rank:
        best_bins, scaler = joblib.load(best_bins_path)
        best_bins = np.array(best_bins)[None, :]
        # post-process
        final_preds = np.clip(scaler.transform(final_preds), 0., 1.)
        final_preds = prevent_nan(
            np.round(final_preds * best_bins) / best_bins
        )

    df_sub = pd.DataFrame(final_preds, columns=OUTPUT_COLUMNS)
    df_sub["qa_id"] = df_valid["qa_id"].values
    df_sub.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    fire.Fire(main)

import logging
from math import floor, ceil

import fire
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import GroupKFold

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    first_sep = True
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


def _trim_input(title, question, answer, max_sequence_length,
                t_max_len=30, q_max_len=239, a_max_len=239):

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:

        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len

        if a_max_len > a_len:
            a_new_len = a_len
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len

        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d"
                             % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))

        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]

    return t, q, a


def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + title + ["[SEP]"] + \
        question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows(), total=df.shape[0]):
        t, q, a = instance.question_title, instance.question_body, instance.answer

        t, q, a = _trim_input(t, q, a, max_sequence_length)

        ids, masks, segments = _convert_to_bert_inputs(
            t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


def main(
    input_path: str = "data/",
    max_sequence_length: int = 512
):
    df_train = pd.read_csv(input_path + 'train.csv')
    output_categories = list(df_train.columns[11:])
    input_categories = list(df_train.columns[[1, 2, 5]])

    gkf = GroupKFold(n_splits=5).split(
        X=df_train.question_body, groups=df_train.question_body)
    outputs = compute_output_arrays(df_train, output_categories)
    inputs = compute_input_arays(
        df_train, input_categories, tokenizer, max_sequence_length)

    for fold, (train_idx, valid_idx) in enumerate(gkf):
        joblib.dump(
            [inputs[0][train_idx], inputs[1][train_idx],
                inputs[2][train_idx], outputs[train_idx]],
            f"cache/tfrecords/train-{fold}.jl"
        )
        joblib.dump(
            [inputs[0][valid_idx], inputs[1][valid_idx],
                inputs[2][valid_idx], outputs[valid_idx]],
            f"cache/tfrecords/valid-{fold}.jl"
        )


if __name__ == '__main__':
    fire.Fire(main)

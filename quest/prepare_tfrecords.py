import math
import logging
from pathlib import Path

import fire
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from transformers import AutoTokenizer
from sklearn.model_selection import GroupKFold

QUESTION_COLUMNS = (
    'question_asker_intent_understanding', 'question_body_critical', 'question_conversational',
    'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer',
    'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent',
    'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice',
    'question_type_compare', 'question_type_consequence', 'question_type_definition',
    'question_type_entity', 'question_type_instructions', 'question_type_procedure',
    'question_type_reason_explanation', 'question_type_spelling', 'question_well_written',
)
ANSWER_COLUMNS = (
    'answer_type_instructions', 'answer_type_procedure', 'answer_type_reason_explanation',
    'answer_well_written', 'answer_level_of_information'
)
JOINT_COLUMNS = (
    'answer_helpful', 'answer_plausible', 'answer_relevance',
    'answer_satisfaction'
)
INPUT_COLUMNS = ('question_title', 'question_body', 'answer')
OUTPUT_COLUMNS = QUESTION_COLUMNS + ANSWER_COLUMNS + JOINT_COLUMNS

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


class Preprocessor:
    def __init__(self, tokenizer, title_max_len=64, question_max_len=352-7, answer_max_len=480-5):
        self.tokenizer = tokenizer
        self.title_max_len = title_max_len
        self.question_max_len = question_max_len
        self.answer_max_len = answer_max_len
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.bos_token])[0]
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.eos_token])[0]
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.pad_token])[0]
        self.question_head = self.tokenizer.encode(
            "question", add_special_tokens=False)
        self.answer_head = self.tokenizer.encode(
            "answer", add_special_tokens=False)
        self.max_q_len = (
            self.title_max_len + self.question_max_len +
            6 + len(self.question_head)
        )
        self.max_a_len = (
            self.answer_max_len + 4 + len(self.answer_head)
        )

    def _trim_input(self, title, question, answer):
        t = self.tokenizer.encode(title, add_special_tokens=False)
        q = self.tokenizer.encode(question, add_special_tokens=False)
        a = self.tokenizer.encode(answer, add_special_tokens=False)
        t = t[:self.title_max_len]
        q = q[:self.question_max_len]
        a = a[:self.answer_max_len]
        return t, q, a

    def process_one_example(self, title, question, answer):
        t_tokens, q_tokens, a_tokens = self._trim_input(
            title, question, answer)

        input_ids_question = np.zeros(
            self.max_q_len, dtype=np.int
        ) + self.pad_token_id
        input_ids_answer = np.zeros(
            self.max_a_len, dtype=np.int
        ) + self.pad_token_id
        question_tokens = np.asarray(
            [self.bos_token_id] + self.question_head +
            [self.eos_token_id, self.bos_token_id] +
            t_tokens + [self.eos_token_id, self.bos_token_id] +
            q_tokens + [self.eos_token_id]
        )
        answer_tokens = np.asarray(
            [self.bos_token_id] + self.answer_head +
            [self.eos_token_id, self.bos_token_id] +
            a_tokens + [self.eos_token_id]
        )
        assert len(question_tokens) <= len(input_ids_question)
        assert len(answer_tokens) <= len(input_ids_answer)
        input_ids_question[:len(question_tokens)] = question_tokens
        input_ids_answer[:len(answer_tokens)] = answer_tokens
        input_mask_question = np.zeros(len(input_ids_question), dtype=np.int)
        input_mask_question[:len(question_tokens)] = 1
        input_mask_answer = np.zeros(len(input_ids_answer), dtype=np.int)
        input_mask_answer[:len(answer_tokens)] = 1
        return {
            "input_ids_question": input_ids_question,
            "input_mask_question": input_mask_question,
            "input_ids_answer": input_ids_answer,
            "input_mask_answer": input_mask_answer
        }


def to_example(input_dict, labels):
    feature = {
        "input_ids_question": tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=input_dict["input_ids_question"])
        ),
        "input_mask_question": tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=input_dict["input_mask_question"])
        ),
        "input_ids_answer": tf.train.Feature(
            int64_list=tf.train.Int64List(value=input_dict["input_ids_answer"])
        ),
        "input_mask_answer": tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=input_dict["input_mask_answer"])
        ),
        "labels": tf.train.Feature(
            float_list=tf.train.FloatList(value=labels)
        )
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _write_tfrecords(inputs, labels, output_filepath):
    with tf.io.TFRecordWriter(str(output_filepath)) as writer:
        for input_dict, labels_single in zip(inputs, labels):
            example = to_example(input_dict, labels_single)
            writer.write(example.SerializeToString())
    print("Wrote file {} containing {} records".format(
        output_filepath, len(inputs)))


def _write_arrays(inputs, labels, output_filepath):
    input_ids, input_mask, token_type_ids = [], [], []
    for input_dict in inputs:
        input_ids.append(input_dict["input_ids"])
        input_mask.append(input_dict["input_mask"])
        token_type_ids.append(input_dict["token_type_ids"])
    joblib.dump(
        [np.stack(input_ids), np.stack(input_mask),
         np.stack(token_type_ids), labels],
        output_filepath)


def main(
    input_path: str = "data/", model_name: str = "roberta-base",
    output_path: str = "cache/tfrecords/", n_folds: int = 5
):
    output_path_ = Path(output_path)
    output_path_.mkdir(exist_ok=True, parents=True)
    (output_path_ / f"tokenizer_{model_name}").mkdir(exist_ok=True)

    df_train = pd.read_csv(input_path + 'train.csv')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(output_path_ / f"tokenizer_{model_name}"))
    print(tokenizer)
    processor = Preprocessor(tokenizer)
    labels = df_train.loc[
        :, OUTPUT_COLUMNS
    ].values
    inputs = df_train.loc[:, INPUT_COLUMNS].values
    tmp = []
    for i in tqdm(range(df_train.shape[0]), ncols=100):
        tmp.append(processor.process_one_example(
            inputs[i, 0],
            inputs[i, 1],
            inputs[i, 2])
        )
    processed_inputs = np.array(tmp)
    print(processed_inputs[0]["input_ids_question"])
    del tmp

    gkf = GroupKFold(n_splits=n_folds).split(
        X=df_train.question_body, groups=df_train.question_body)
    for fold, (train_idx, valid_idx) in enumerate(gkf):
        joblib.dump([train_idx, valid_idx], output_path_ / f"fold_{fold}.jl")
        filepath = (
            output_path_ /
            f"train-{fold}-{len(train_idx)}-{processor.max_q_len}-{processor.max_a_len}.tfrec"
        )
        _write_tfrecords(
            processed_inputs[train_idx], labels[train_idx], filepath)
        filepath = (
            output_path_ /
            f"valid-{fold}-{len(valid_idx)}-{processor.max_q_len}-{processor.max_a_len}.tfrec"
        )
        _write_tfrecords(
            processed_inputs[valid_idx], labels[valid_idx], filepath)


if __name__ == '__main__':
    fire.Fire(main)

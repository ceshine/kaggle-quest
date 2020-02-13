import logging
from pathlib import Path

import fire
import joblib
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
from transformers import TFBertForSequenceClassification, BertConfig, RobertaConfig
from tf_helper_bot import CosineDecayWithWarmup

from .models import PoolingBertModel, PoolingRobertaModel, tfhub_bert_model, custom_bert_model
from .dataset import tfrecord_dataset


def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        rhos.append(
            spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
    return np.mean(rhos)


class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, valid_data, batch_size=16):
        self.valid_inputs = valid_data[0]
        self.valid_outputs = valid_data[1]
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.valid_predictions = []

    def on_epoch_end(self, epoch, logs={}):
        self.valid_predictions.append(
            self.model.predict(self.valid_inputs, batch_size=self.batch_size))
        rho_val = compute_spearmanr(
            self.valid_outputs, np.average(self.valid_predictions, axis=0))
        print("\nvalidation rho: %.4f" % rho_val)


def train_model(
    train_path: str = "cache/tfrecords/train-0.jl",
    valid_path: str = "cache/tfrecords/valid-0.jl",
    model_name: str = "bert-base-uncased",
    output_path: str = "cache/keras-modl/",
    batch_size: int = 8,
    epochs: int = 5
):
    Path(output_path).mkdir(exist_ok=True, parents=True)
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        tpu = None
    strategy = tf.distribute.get_strategy()
    print(strategy)
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("REPLICAS: ", strategy.num_replicas_in_sync)

    valid_batch_size = batch_size * 2
    if strategy.num_replicas_in_sync == 8:  # single TPU
        valid_batch_size = batch_size * strategy.num_replicas_in_sync * 2
        batch_size = batch_size * strategy.num_replicas_in_sync
    logging.getLogger("tensorflow").setLevel(logging.INFO)

    train_data = joblib.load(train_path)
    valid_data = joblib.load(valid_path)

    steps = int(epochs * 1.05 * len(train_data[0]) / batch_size)
    with strategy.scope():
        if model_name.lower().startswith("bert"):
            config = BertConfig.from_pretrained(model_name, num_labels=30)
            model = PoolingBertModel.from_pretrained(
                model_name, config=config)
        elif model_name.lower().startswith("roberta"):
            config = RobertaConfig.from_pretrained(model_name, num_labels=30)
            model = PoolingRobertaModel.from_pretrained(
                model_name, config=config)
        else:
            raise ValueError("Unknown model!")
        # model = custom_bert_model()
        # model = tfhub_bert_model()
        lr_schedule = CosineDecayWithWarmup(
            initial_learning_rate=1e-6, max_learning_rate=3e-5,
            warmup_steps=int(steps * 0.2),
            decay_steps=steps - int(steps * 0.2),
            alpha=1e-4
        )
        opt = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, epsilon=1e-08)
        loss_ = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        custom_callback = CustomCallback(
            valid_data=(valid_data[:3], valid_data[3]),
            batch_size=valid_batch_size)
        model.compile(optimizer=opt, loss=loss_, metrics=[])
        print(model.summary())

    idx = np.random.choice(np.arange(len(train_data[0])), 4848, replace=False)
    for i in range(4):
        train_data[i] = train_data[i][idx]
    model.fit(
        train_data[:3], train_data[3],
        epochs=epochs, batch_size=batch_size, callbacks=[custom_callback]
    )

    model.save_pretrained(output_path)


if __name__ == '__main__':
    fire.Fire(train_model)

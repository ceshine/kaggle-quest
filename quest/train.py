import os
import logging
from pathlib import Path

import fire
import tensorflow as tf
from transformers import BertConfig, RobertaConfig
from tf_helper_bot import (
    BaseBot, BaseDistributedBot,
    MovingAverageStatsTrackerCallback, CheckpointCallback, TelegramCallback,
    CosineDecayWithWarmup
)
from tf_helper_bot.utils import prepare_tpu
from tf_helper_bot.optimizers import RAdam

from .models import PoolingBertModel, DualRobertaModel
from .dataset import tfrecord_dataset
from .metrics import SpearmanCorr

TELEGRAM_TOKEN = os.environ.get("TG_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TG_CHAT_ID", "")


class QuestBot(BaseBot):
    def _extract_prediction(self, x):
        if isinstance(x, tuple):
            # the model returns a tuple when run inside tf.function
            return x[0]
        return x


class QuestDistributedBot(BaseDistributedBot):
    def _extract_prediction(self, x):
        if isinstance(x, tuple):
            return x[0]
        return x


def loss_fn(labels, predictions):
    return tf.math.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            labels, predictions, from_logits=True
            # tf.keras.losses.mean_absolute_error(
            # tf.keras.losses.mean_squared_error(
            #     labels, predictions,
        ),
        axis=0
    )


def train_model(
    train_path: str = "cache/tfrecords/train-0-4863-288-320.tfrec",
    valid_path: str = "cache/tfrecords/valid-0-1216-288-320.tfrec",
    model_name: str = "bert-large-uncased-whole-word-masking",
    output_path: str = "cache/model",
    batch_size: int = 8, grad_accu: int = 2,
    log_interval: int = 200, steps: int = 1000,
    checkpoint_interval: int = 500,
    min_lr: float = 1e-6, max_lr: float = 3e-5,
    freeze: int = 0
):
    # Path(output_path).mkdir(exist_ok=True, parents=True)
    strategy, tpu = prepare_tpu()
    print("REPLICAS: ", strategy.num_replicas_in_sync)

    valid_batch_size = batch_size * 2
    if strategy.num_replicas_in_sync == 8:  # single TPU
        valid_batch_size = batch_size * strategy.num_replicas_in_sync * 2
        batch_size = batch_size * strategy.num_replicas_in_sync
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

    with strategy.scope():
        train_ds, train_steps = tfrecord_dataset(
            train_path, batch_size, strategy, is_train=True)
        valid_ds, valid_steps = tfrecord_dataset(
            valid_path, valid_batch_size, strategy, is_train=False)
        if model_name.lower().startswith("bert"):
            config = BertConfig.from_pretrained(model_name, num_labels=30)
            model = PoolingBertModel.from_pretrained(
                model_name, config=config)
        elif model_name.lower().startswith("roberta"):
            config = RobertaConfig.from_pretrained(model_name, num_labels=30)
            model = DualRobertaModel(
                model_name=model_name, config=config)
        else:
            raise ValueError("Unknown model!")
        lr_schedule = CosineDecayWithWarmup(
            initial_learning_rate=min_lr, max_learning_rate=max_lr,
            warmup_steps=int(steps * 0.1),
            decay_steps=steps - int(steps * 0.1),
            alpha=1e-4
        )
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-6)
        optimizer_1 = RAdam(learning_rate=1e-3, epsilon=1e-6)
        optimizer = RAdam(learning_rate=lr_schedule, epsilon=1e-6)
        # build the model
        model(next(iter(train_ds))[0])

    if freeze > 0:
        model.freeze()
        model.compile(
            optimizer=optimizer_1,
            loss=loss_fn
        )
        print(model.summary())
        model.fit(
            train_ds, epochs=1,
            steps_per_epoch=train_steps * freeze
        )
    model.unfreeze()
    model.compile(
        optimizer=optimizer,
        loss=loss_fn
    )
    print(model.summary())

    train_dist_ds = strategy.experimental_distribute_dataset(
        train_ds)
    valid_dist_ds = strategy.experimental_distribute_dataset(
        valid_ds)

    checkpoints = CheckpointCallback(
        keep_n_checkpoints=1,
        checkpoint_dir="cache/model_cache/",
        monitor_metric="spearman"
    )
    callbacks = [
        MovingAverageStatsTrackerCallback(
            avg_window=int(log_interval * 1.25),
            log_interval=log_interval,
        ),
        checkpoints
    ]
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        callbacks += [
            TelegramCallback(
                token=TELEGRAM_TOKEN,
                chat_id=TELEGRAM_CHAT_ID,
                name="QuestFinetune",
                report_evals=False
            )
        ]
    metrics = (SpearmanCorr(add_sigmoid=True),)
    if tpu:
        bot = QuestDistributedBot(
            model=model,
            criterion=loss_fn,
            optimizer=optimizer,
            train_dataset=train_dist_ds,
            valid_dataset=valid_dist_ds,
            steps_per_epoch=train_steps,
            strategy=strategy,
            gradient_accumulation_steps=1,
            callbacks=callbacks,
            metrics=metrics,
            valid_steps=valid_steps,
        )
    else:
        bot = QuestBot(
            model=model,
            criterion=loss_fn,
            optimizer=optimizer,
            train_dataset=train_dist_ds,
            valid_dataset=valid_dist_ds,
            steps_per_epoch=train_steps,
            gradient_accumulation_steps=grad_accu,
            callbacks=callbacks,
            metrics=metrics,
            valid_steps=valid_steps
        )
    print(f"Steps per epoch: {train_steps} | {valid_steps}")

    bot.train(checkpoint_interval=checkpoint_interval, n_steps=steps)
    best_score = checkpoints.best_performers[0][0]
    bot.model.load_weights(str(checkpoints.best_performers[0][1]))
    checkpoints.remove_checkpoints(keep=0)
    bot.model.save_weights(output_path + ".h5")
    return best_score


if __name__ == '__main__':
    fire.Fire(train_model)

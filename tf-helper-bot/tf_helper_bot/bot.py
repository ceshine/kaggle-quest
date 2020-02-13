import logging
from pathlib import Path
from typing import Callable, Sequence, Union, Optional

import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from tqdm.autonotebook import tqdm

from .logger import Logger


@dataclass
class BaseBot:
    """Base Interface to Model Training and Inference"""
    train_dataset: tf.data.Dataset
    valid_dataset: tf.data.Dataset
    steps_per_epoch: int
    criterion: Callable
    model: tf.keras.Model
    optimizer: tf.keras.optimizers.Optimizer
    name: str = "basebot"
    log_dir: Union[Path, str] = "./logs"
    log_level: int = logging.INFO
    loss_format: str = "%.4f"
    echo: bool = True
    pbar: bool = True
    step: int = 0
    total_steps: int = 0
    valid_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    metrics: Sequence = ()
    callbacks: Sequence = ()
    mixed_precision: bool = False

    def __post_init__(self):
        self._gradients = []
        self.logger = Logger(
            self.name, Path(self.log_dir), self.log_level,
            echo=self.echo
        )

        @tf.function
        def get_gradient(input_tensors, target):
            with tf.GradientTape() as tape:
                output = self.model(
                    input_tensors, training=True)
                loss_raw = self.criterion(
                    target, self._extract_prediction(output)
                )
                loss_ = (
                    self.optimizer.get_scaled_loss(loss_raw)
                    if self.mixed_precision else loss_raw
                )
            gradients_ = tape.gradient(
                loss_, self.model.trainable_variables)
            if self.mixed_precision:
                gradients_ = self.optimizer.get_unscaled_gradients(gradients_)
            return loss_raw, gradients_

        @tf.function
        def step_optimizer(gradients):
            self.optimizer.apply_gradients(
                zip(
                    gradients,
                    self.model.trainable_variables
                )
            )

        @tf.function
        def predict_batch(input_tensors):
            return self.model(input_tensors, training=False)

        self._get_gradient = get_gradient
        self._step_optimizer = step_optimizer
        self._predict_batch = predict_batch

    @staticmethod
    def _sum_indexed_slice(grad_1, grad_2, div_):
        values = tf.concat([grad_1.values, grad_2.values / div_], 0)
        indices = tf.concat([grad_1.indices, grad_2.indices], 0)
        return tf.IndexedSlices(values, indices)

    def train_one_step(self, input_tensor_list, target):
        loss, gradients = self._get_gradient(
            input_tensor_list[0], target)
        if self.gradient_accumulation_steps > 1:
            div_ = tf.constant(
                self.gradient_accumulation_steps,
                dtype=tf.float32
            )
            gradients = [x / div_ for x in gradients]
            loss, gradients = self._get_gradient(
                input_tensor_list[0], target)
            for i in range(1, self.gradient_accumulation_steps):
                loss_, gradients_ = self._get_gradient(
                    input_tensor_list[i], target)
                gradients = [
                    grad_1 + grad_2 / div_ if not isinstance(grad_1, tf.IndexedSlices)
                    else self._sum_indexed_slice(grad_1, grad_2, div_)
                    for grad_1, grad_2 in zip(gradients, gradients_)
                ]
                loss = loss + loss_
            loss = loss / tf.constant(
                self.gradient_accumulation_steps,
                dtype=tf.float32
            )
        self._step_optimizer(gradients)
        return loss

    @staticmethod
    def _extract_prediction(output):
        """Can be overridden to act as a shortcut to transform model outputs.

        Useful when using a pretrained model whose outputs are not in the desired format.
        """
        return output

    def train(self, *, checkpoint_interval, n_steps=None, total_steps=None):
        if total_steps:
            self.total_steps = total_steps
        if n_steps is None:
            if self.total_steps is None:
                raise ValueError("n_steps and total_steps cannot both be None")
            n_steps = self.total_steps - self.step
        elif self.total_steps is None:
            self.total_steps = n_steps
        target_step = self.step + n_steps
        input_tensor_list, cnt = [], 0
        # Train starts
        self.run_train_starts_callbacks()
        try:
            while self.step < target_step:
                for input_tensors, targets in self.train_dataset:
                    self.step += 1
                    input_tensors, targets = self.run_batch_inputs_callbacks(
                        input_tensors, targets)
                    input_tensor_list.append(input_tensors)
                    cnt += self.get_batch_size(input_tensors)
                    if len(input_tensor_list) == self.gradient_accumulation_steps:
                        loss = self.train_one_step(
                            input_tensor_list, targets
                        )
                        # Step ends
                        self.run_step_ends_callbacks(loss.numpy(), cnt)
                        input_tensor_list, cnt = [], 0
                    if (
                        (callable(checkpoint_interval) and checkpoint_interval(self.step)) or
                        (
                            not callable(checkpoint_interval) and
                            self.step % checkpoint_interval == 0
                        )
                    ):
                        # Eval starts
                        metrics = self.eval(self.valid_dataset)
                        # Eval ends
                        self.run_eval_ends_callbacks(metrics)
                    if self.step >= target_step:
                        break
                    # Epoch ends
                    if self.step % self.steps_per_epoch == 0:
                        self.run_epoch_ends_callbacks(
                            self.step // self.steps_per_epoch)
        except (KeyboardInterrupt):
            pass
        finally:
            # Train ends
            self.run_train_ends_callbacks()

    def predict_batch(self, input_tensors):
        """To be overriden in distributed modes"""
        return self._extract_prediction(
            self._predict_batch(input_tensors)
        )

    def _extract_target_for_eval(self, target):
        return target

    def predict(self, dataset, *, return_y=False):
        self.model.eval()
        outputs, y_global = [], []
        for *input_tensors, y_local in tqdm(dataset, disable=not self.pbar):
            outputs.append(self.predict_batch(input_tensors).numpy())
            if return_y:
                y_global.append(
                    self._extract_target_for_eval(y_local).numpy())
        outputs = np.concatenate(outputs, axis=0)
        if return_y:
            y_global = np.concatenate(y_global, axis=0)
            return outputs, y_global
        return outputs

    def eval(self, dataset):
        """Warning: Only support datasets whose predictions and labels together fit in memory."""
        preds, ys = [], []
        losses, weights = [], []
        self.logger.debug("Evaluating...")
        for input_tensors, y_local in tqdm(dataset, disable=not self.pbar, total=self.valid_steps, ncols=100):
            output = self.predict_batch(input_tensors)
            y_local = self._extract_target_for_eval(y_local)
            batch_loss = self.criterion(y_local, output)
            losses.append(batch_loss.numpy())
            weights.append(y_local.shape[0])
            # Save batch labels and predictions
            preds.append(output.numpy())
            ys.append(y_local.numpy())
        loss = np.average(losses, weights=weights)
        metrics = {"loss": (loss, self.loss_format % loss)}
        global_ys, global_preds = np.concatenate(ys), np.concatenate(preds)
        for metric in self.metrics:
            metric_loss, metric_string = metric(global_ys, global_preds)
            metrics[metric.name] = (metric_loss, metric_string)
        return metrics

    def get_batch_size(self, input_tensors):
        if isinstance(input_tensors, list):
            return self.get_batch_size(input_tensors[0])
        elif isinstance(input_tensors, dict):
            return self.get_batch_size(list(input_tensors.values())[0])
        return input_tensors.shape[0]

    def run_batch_inputs_callbacks(self, input_tensors, targets):
        for callback in self.callbacks:
            input_tensors, targets = callback.on_batch_inputs(
                self, input_tensors, targets)
        return input_tensors, targets

    def run_step_ends_callbacks(self, train_loss, train_weight):
        for callback in self.callbacks:
            callback.on_step_ends(self, train_loss, train_weight)

    def run_train_starts_callbacks(self):
        for callback in self.callbacks:
            callback.on_train_starts(self)

    def run_train_ends_callbacks(self):
        for callback in self.callbacks:
            callback.on_train_ends(self)

    def run_epoch_ends_callbacks(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_ends(self, epoch)

    def run_eval_ends_callbacks(self, metrics):
        for callback in self.callbacks:
            callback.on_eval_ends(self, metrics)


@dataclass
class BaseDistributedBot(BaseBot):
    """Base Interface to Model Training and Inference"""
    strategy: tf.distribute.Strategy = None

    def __post_init__(self):
        assert self.strategy is not None
        assert self.gradient_accumulation_steps == 1, (
            "Distribution mode doesn't suppoprt gradient accumulation"
        )
        super().__post_init__()
        @tf.function
        def train_one_step(input_tensor_list, target):
            loss, gradients = self._get_gradient(
                input_tensor_list[0], target)
            self._step_optimizer(gradients)
            return loss

        self._train_one_step = train_one_step

    def train_one_step(self, input_tensors, target):
        loss = self.strategy.experimental_run_v2(
            self._train_one_step,
            args=(input_tensors, target)
        )
        return self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, loss, axis=None
        )

    def get_batch_size(self, input_tensors):
        # Just use a rough estimate for speed
        return 1
        # the following can be slow (and unnecessary in most cases)
        # if isinstance(input_tensors, list):
        #     x_per_gpu_as_list = self.strategy.experimental_local_results(
        #         input_tensors[0])
        # else:
        #     x_per_gpu_as_list = self.strategy.experimental_local_results(
        #         input_tensors)
        # batch_sizes = [tf.shape(x_gpu)[0] for x_gpu in x_per_gpu_as_list]
        # return tf.reduce_sum(tf.stack(batch_sizes)).numpy()

    def _extract_target_for_eval(self, target):
        return tf.concat(
            self.strategy.experimental_local_results(target),
            axis=0
        )

    def predict_batch(self, input_tensors):
        preds = self.strategy.experimental_run_v2(
            self._predict_batch,
            args=(input_tensors,)
        )
        if isinstance(preds, tuple):
            # WARNING: This might not applicable in all situations
            preds = preds[0]
        preds_local = tf.concat(
            preds.values, axis=0
        )
        return preds_local

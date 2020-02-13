import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule


class CosineDecayWithWarmup(LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay schedule."""

    def __init__(
            self,
            initial_learning_rate,
            max_learning_rate,
            warmup_steps,
            decay_steps,
            alpha=0.0,
            name=None):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.max_learning_rate = max_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name

    @staticmethod
    def lr_warmup(steps, warmup_steps, max_learning_rate, initial_learning_rate):
        return initial_learning_rate + (
            max_learning_rate - initial_learning_rate
        ) * (steps / warmup_steps)

    @staticmethod
    def cosine_decay(steps, warmup_steps, decay_steps, max_learning_rate, alpha):
        completed_fraction = (
            steps - warmup_steps) / decay_steps
        cosine_decayed = 0.5 * (1.0 + math_ops.cos(
            constant_op.constant(math.pi) * completed_fraction))
        decayed = (1 - alpha) * cosine_decayed + alpha
        return math_ops.multiply(max_learning_rate, decayed)

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "CosineDecayWithWarmup"):
            initial_learning_rate = ops.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            max_learning_rate = ops.convert_to_tensor(
                self.max_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = math_ops.cast(self.decay_steps, dtype)
            warmup_steps = math_ops.cast(self.warmup_steps, dtype)
            total_steps = decay_steps + warmup_steps

            global_step_recomp = math_ops.cast(step, dtype)
            global_step_recomp = math_ops.minimum(
                global_step_recomp, total_steps)

            return control_flow_ops.cond(
                math_ops.less_equal(global_step_recomp, warmup_steps),
                lambda: self.lr_warmup(
                    global_step_recomp, warmup_steps, max_learning_rate,
                    initial_learning_rate
                ),
                lambda: self.cosine_decay(
                    global_step_recomp, warmup_steps, decay_steps,
                    max_learning_rate, self.alpha
                )
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name
        }

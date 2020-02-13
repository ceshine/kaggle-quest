import tensorflow as tf
import tensorflow_probability as tfp


def mixup_augment(alpha: float):
    """
       Adapted from:
       https://github.com/tensorpack/tensorpack/blob/master/examples/ResNet/cifar10-preact18-mixup.py
    """
    dist = tfp.distributions.Beta(alpha, alpha)

    def _mixup_augment(images, labels):
        batch_size = tf.shape(images)[0]
        lambd = dist.sample([batch_size])
        lambd = tf.math.reduce_max(
            tf.stack([lambd, 1-lambd]), axis=0
        )
        lambd = tf.reshape(lambd, [batch_size, 1, 1, 1])
        index = tf.random.shuffle(tf.range(batch_size))
        new_images = images * lambd + tf.gather(images, index) * (1 - lambd)
        return new_images, {"labels_1": labels, "labels_2": tf.gather(labels, index), "lambd": lambd[:, 0, 0, 0]}
    return _mixup_augment


def mixup_loss_fn(y_true, y_pred):
    if isinstance(y_true, dict):
        loss_1 = tf.keras.losses.sparse_categorical_crossentropy(
            y_true["labels_1"],
            y_pred
        )
        loss_2 = tf.keras.losses.sparse_categorical_crossentropy(
            y_true["labels_2"],
            y_pred
        )
        loss = tf.reduce_mean(
            y_true["lambd"] * loss_1 + (1 - y_true["lambd"]) * loss_2
        )
    else:
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y_true,
                y_pred
            )
        )
    return loss

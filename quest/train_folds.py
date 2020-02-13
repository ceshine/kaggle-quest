import fire
import tensorflow as tf
import numpy as np

from .train import train_model


def main(
    train_path_pattern: str = "cache/tfrecords/train-%d-*.tfrec",
    valid_path_pattern: str = "cache/tfrecords/valid-%d-*.tfrec",
    model_name: str = "bert-large-uncased-whole-word-masking",
    output_path_pattern: str = "cache/bert-fold-%d/",
    batch_size: int = 8, grad_accu: int = 2,
    log_interval: int = 200, steps: int = 1000,
    checkpoint_interval: int = 500,
    min_lr: float = 1e-6, max_lr: float = 3e-5,
    n_folds: int = 5, freeze: int = 0
):
    scores = []
    for fold in range(n_folds):
        tmp = list(tf.io.gfile.glob(train_path_pattern % fold))
        assert len(tmp) == 1
        train_path = tmp[0]
        tmp = list(tf.io.gfile.glob(valid_path_pattern % fold))
        assert len(tmp) == 1
        valid_path = tmp[0]
        output_path = output_path_pattern % fold
        print("=" * 20)
        print(f"Training Fold {fold+1}")
        print("=" * 20)
        best_score = train_model(
            train_path=train_path,
            valid_path=valid_path,
            model_name=model_name,
            output_path=output_path,
            batch_size=batch_size,
            grad_accu=grad_accu,
            log_interval=log_interval,
            steps=steps,
            checkpoint_interval=checkpoint_interval,
            min_lr=min_lr,
            max_lr=max_lr,
            freeze=freeze
        )
        scores.append(best_score)
    print(f"Scores: {np.mean(scores)} +- {np.std(scores)}")


if __name__ == '__main__':
    fire.Fire(main)

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def tfrecord_dataset(filename, batch_size, strategy, is_train: bool = True):
    opt = tf.data.Options()
    opt.experimental_deterministic = False

    name = filename.split("/")[-1]
    max_q_len = int(name.split("-")[3].split(".")[0])
    max_a_len = int(name.split("-")[4].split(".")[0])
    cnt = int(name.split("-")[2])

    features_description = {
        "input_ids_question": tf.io.FixedLenFeature([max_q_len], tf.int64),
        "input_mask_question": tf.io.FixedLenFeature([max_q_len], tf.int64),
        "input_ids_answer": tf.io.FixedLenFeature([max_a_len], tf.int64),
        "input_mask_answer": tf.io.FixedLenFeature([max_a_len], tf.int64),
        "labels": tf.io.FixedLenFeature([30], tf.float32),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        example = tf.io.parse_single_example(
            example_proto, features_description)
        return (
            {
                'input_ids_question': tf.cast(example['input_ids_question'], tf.int32),
                'attention_mask_question': tf.cast(example['input_mask_question'], tf.int32),
                'input_ids_answer': tf.cast(example['input_ids_answer'], tf.int32),
                'attention_mask_answer': tf.cast(example['input_mask_answer'], tf.int32),
            },
            example["labels"]
        )

    raw_dataset = tf.data.TFRecordDataset(
        filename, num_parallel_reads=4
    ).with_options(opt)
    dataset = raw_dataset.map(
        _parse_function, num_parallel_calls=AUTOTUNE
    ).cache()
    if is_train:
        dataset = dataset.shuffle(
            2048, reshuffle_each_iteration=True
        ).repeat()
    else:
        # usually fewer validation files than workers so disable FILE auto-sharding on validation
        # option not useful if there is no sharding (not harmful either)
        if strategy.num_replicas_in_sync > 1:
            opt = tf.data.Options()
            opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            dataset = dataset.with_options(opt)
    dataset = dataset.batch(
        batch_size
        # drop_remainder=is_train
    )
    dataset = dataset.prefetch(AUTOTUNE)
    print("cnt:", cnt, "batch size:", batch_size)
    return dataset, int(np.ceil(cnt / batch_size))

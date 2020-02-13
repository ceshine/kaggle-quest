import tensorflow as tf
import tensorflow_hub as hub
from transformers import TFBertPreTrainedModel, TFRobertaPreTrainedModel, TFBertModel
from transformers.modeling_tf_utils import get_initializer
from transformers.modeling_tf_bert import TFBertMainLayer
from transformers.modeling_tf_roberta import TFRobertaMainLayer, TFRobertaClassificationHead

from .prepare_tfrecords import QUESTION_COLUMNS, ANSWER_COLUMNS, JOINT_COLUMNS


class PoolingBertModel(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.bert = TFBertMainLayer(config, name="bert")
        # self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.classifier = tf.keras.layers.Dense(
            config.num_labels,
            # kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
            activation="linear"
        )

    def freeze(self):
        self.bert.trainable = False

    def unfreeze(self):
        self.bert.trainable = True

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)

        sequence_output = outputs[0]
        pooled_output = self.pooling(sequence_output)

        pooler_output = outputs[1]
        pooled_output = self.dropout(
            tf.concat(
                [pooled_output, pooler_output],
                axis=1
            ), training=kwargs.get("training", False))
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        return outputs


class AveragePooling(tf.keras.layers.Layer):
    def call(self, states, mask):
        mask = tf.cast(tf.expand_dims(mask, 2), tf.float32)
        pooled = tf.reduce_sum(states * mask, axis=1)
        return pooled / tf.reduce_sum(mask, axis=1)


class SELayer(tf.keras.layers.Layer):
    def __init__(self, channels, reduction):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(
            channels // reduction,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            name="fc1",
            activation="relu"
        )
        self.fc2 = tf.keras.layers.Dense(
            channels,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            name="fc2",
            activation="sigmoid"
        )

    def call(self, x):
        tmp = self.fc1(x)
        tmp = self.fc2(tmp)
        return tmp * x


class RobertaEncoder(TFRobertaPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.roberta = TFRobertaMainLayer(config, name="roberta")
        self.pooling = AveragePooling()

    def call(self, inputs, **kwargs):
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = tf.ones(
                tf.shape(inputs["input_ids"])[:2], tf.int32
            )
        outputs = self.roberta(inputs, **kwargs)[0]
        return self.pooling(outputs, inputs["attention_mask"])


class DualRobertaModel(tf.keras.Model):
    def __init__(self, config, model_name, pretrained: bool = True):
        super().__init__()
        self.num_labels = config.num_labels

        if pretrained:
            self.roberta = RobertaEncoder.from_pretrained(
                model_name, config=config, name="roberta_question")
        else:
            self.roberta = RobertaEncoder(
                config=config, name="roberta_question")
        self.dropout = tf.keras.layers.Dropout(0.5)
        # self.hidden_layer = tf.keras.layers.Dense(
        #     1024,
        #     kernel_initializer=tf.keras.initializers.he_normal(seed=None),
        #     name="hidden",
        #     activation="relu"
        # )
        self.q_classifier = tf.keras.layers.Dense(
            len(QUESTION_COLUMNS),
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            name="q_classifier",
            activation="linear"
        )
        self.a_classifier = tf.keras.layers.Dense(
            len(ANSWER_COLUMNS),
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            name="a_classifier",
            activation="linear"
        )
        self.j_classifier = tf.keras.layers.Dense(
            len(JOINT_COLUMNS),
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            name="j_classifier",
            activation="linear"
        )
        self.gating_q = SELayer(config.hidden_size, 4)
        self.gating_a = SELayer(config.hidden_size, 4)
        self.gating_j = SELayer(config.hidden_size * 3, 4)
        # self.ln1 = tf.keras.layers.LayerNormalization()

    def freeze(self):
        self.roberta.trainable = False

    def unfreeze(self):
        self.roberta.trainable = True

    def call(self, inputs, **kwargs):
        pooled_output_question = self.roberta(
            {
                "input_ids": inputs["input_ids_question"],
                "attention_mask": inputs["attention_mask_question"]
            }, **kwargs
        )
        pooled_output_answer = self.roberta(
            {
                "input_ids": inputs["input_ids_answer"],
                "attention_mask": inputs["attention_mask_answer"]
            }, **kwargs
        )
        combined = tf.concat(
            [
                pooled_output_question, pooled_output_answer,
                pooled_output_answer * pooled_output_question
            ],
            axis=1
        )
        # hidden = self.hidden_layer(self.dropout(
        #     combined, training=kwargs.get("training", False)
        # ))
        q_logit = self.q_classifier(self.dropout(
            self.gating_q(
                pooled_output_question
            ), training=kwargs.get("training", False)
        ))
        a_logit = self.a_classifier(self.dropout(
            self.gating_a(
                pooled_output_answer
            ), training=kwargs.get("training", False)
        ))
        j_logit = self.j_classifier(self.dropout(
            self.gating_j(
                combined
            ), training=kwargs.get("training", False)
        ))
        logits = tf.concat(
            [q_logit, a_logit, j_logit],
            axis=1
        )
        # add hidden states and attention if they are here
        outputs = (logits,)
        return outputs


def custom_bert_model(
        max_sequence_length=512,
        model_name="bert-base-uncased"):

    input_word_ids = tf.keras.layers.Input(
        (max_sequence_length,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (max_sequence_length,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (max_sequence_length,), dtype=tf.int32, name='input_segments')

    bert_layer = TFBertModel.from_pretrained(model_name)

    sequence_output, _ = bert_layer(
        [input_word_ids, input_masks, input_segments])

    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(
        30, activation="linear", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], outputs=out)

    return model


def tfhub_bert_model(
        max_sequence_length=512,
        model_path="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"):
        # model_path="gs://tpu-experiment-tmp/bert-model/"):

    input_word_ids = tf.keras.layers.Input(
        (max_sequence_length,), dtype=tf.int32, name='input_word_ids')
    input_masks = tf.keras.layers.Input(
        (max_sequence_length,), dtype=tf.int32, name='input_masks')
    input_segments = tf.keras.layers.Input(
        (max_sequence_length,), dtype=tf.int32, name='input_segments')

    bert_layer = hub.KerasLayer(model_path, trainable=True)

    _, sequence_output = bert_layer(
        [input_word_ids, input_masks, input_segments])

    x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(
        30, activation="linear", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], outputs=out)

    return model

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import string
import tensorflow as tf

from models import label_util
from models.base_estimator import BaseEstimator

from transformer.utils import metrics
from transformer.model.vv1_transformer import Transformer as Transformer1
from transformer.model.vc_co1_transformer import Transformer as Transformer_co1
from transformer.model.vc_co2_transformer import Transformer as Transformer_co2
from transformer.model.vc_co3_transformer import Transformer as Transformer_co3
from transformer.model.embedding_layer import EmbeddingSharedWeights
from transformer import compute_bleu

from transformer.utils.tokenizer import PAD, PAD_ID, EOS, EOS_ID, RESERVED_TOKENS
import numpy as np



class TransformerEstimator(BaseEstimator):
    """docstring for TransformerEstimator"""



    def __init__(self, params, run_config, **kwargs):
        self.label_dic = params.get('label_dic')
        self.viseme_dic = params.get('viseme_dic')
        self.pinyin_dic = params.get('pinyin_dic')
        params.update({'target_pinyin_vocab_size': len(self.pinyin_dic )})
        params.update({'target_viseme_vocab_size': len(self.viseme_dic)})
        super(TransformerEstimator, self).__init__(params, run_config, **kwargs)

    def preprocess_labels(self, labels, dic):
        """ preprocess labels to satisfy the need of model.

        Args:
            labels: 1-D Tensor. labels of shape (batch_size,). For example: [ 'ab haha', 'hha fd fd']

        Returns: 2-D int32 Tensor with shape  [batch_size, T]  . labels with numeric shape. EOS is padded to each label.

        """
        pad_tensor = tf.constant([EOS], tf.string)
        # append 'E' to label
        eos = tf.tile(pad_tensor, tf.shape(labels))  # Bx1
        labels = tf.string_join([labels, eos])

        label_char_list = label_util.string2char_list(labels)  # B x T

        numeric_label = label_util.string2indices(
            label_char_list,
            dic=dic)  # SparseTensor, dense_shape: [B x T]
        numeric_label = tf.sparse_tensor_to_dense(
            numeric_label, default_value=PAD_ID)  # [B x T]


        return label_char_list,numeric_label



    def id_to_string(self, predictions,dic):
        """convert predictions to string.

        Args:
            predictions: 3-D int64 Tensor with shape: [batch_size, T, vocab_size]

        Returns: 1-D string SparseTensor with dense shape: [batch_size,]

        """
        predictions = tf.contrib.layers.dense_to_sparse(
            predictions, eos_token=PAD_ID)  # remove PAD_ID
        predictions = tf.sparse_tensor_to_dense(predictions, EOS_ID)
        predictions = tf.contrib.layers.dense_to_sparse(
            predictions, eos_token=EOS_ID)  # remove EOS_ID
        predicted_char_list = label_util.indices2string(predictions, dic)
        predicted_string = label_util.char_list2string(
            predicted_char_list)  # ['ab', 'abc']
        return predicted_string


    def procesess_label(self,labels,dic):
        label_index = tf.contrib.layers.dense_to_sparse(
            labels, eos_token=PAD_ID)
        label_index = tf.sparse_tensor_to_dense(label_index, EOS_ID)
        label_index = tf.contrib.layers.dense_to_sparse(
            label_index, eos_token=EOS_ID)  # remove EOS_ID
        index2string_table = tf.contrib.lookup.index_to_string_table_from_tensor(
            dic, default_value='_')

        label_string = index2string_table.lookup(tf.cast(label_index, tf.int64))
        label_string = tf.sparse_tensor_to_dense(label_string, default_value="")
        label_string = label_util.string_join(label_string)

        return  label_index,label_string


    def model_fn(self, features, labels, mode, params):
        """ Model function for transformer.

        Args:
            features: float Tensor with shape [batch_size, T, H, W, C]. Input sequence.
            labels: string Tensor with shape [batch_size,]. Target labels.
            mode: Indicate train or eval or predict.
            params: dict. model parameters.

        Returns: tf.estimator.EstimatorSpec.

        """
        #learning_rate = params.get('learning_rate', 0.001)

        in_training = mode == tf.estimator.ModeKeys.TRAIN

        video = features['video']
        inputs_unpadded_length = features['unpadded_length']

        if params.get('feature_extractor') == 'early_fusion':
            from .cnn_extractor import EarlyFusion2D as CnnExtractor
        elif params.get('feature_extractor') == 'res':
            from .cnn_extractor import ResNet as CnnExtractor
        else:
            from .cnn_extractor import LipNet as CnnExtractor

        feature_extractor = CnnExtractor(
            feature_len=params.get('hidden_size'),
            training=in_training,
            scope='cnn_feature_extractor')

        inputs = feature_extractor.build(video)  # [batch_size, input_length, hidden_size]

        params.update({'pinyin_vocab_size': len(self.pinyin_dic)})
        params.update({'viseme_vocab_size': len(self.viseme_dic)})
        v_p_transformer = Transformer1(params, in_training, scope="v_p_transformer")

        label_params = params.copy()
        label_params.update({'vocab_size': len(self.label_dic)})
        label_params.update({'target_vocab_size': len(self.label_dic)})
        label_params.update({'scope': "v_c_transformer"})
        label_params.update({'dic': self.label_dic})


        if params.get('co_attention') == 1:
            v_c_transformer = Transformer_co1(label_params, in_training, scope="v_c_transformer")
        elif params.get('co_attention') == 2:
            v_c_transformer = Transformer_co2(label_params, in_training, scope="v_c_transformer")
        elif params.get('co_attention') == 3:
            v_c_transformer = Transformer_co3(label_params, in_training, scope="v_c_transformer")
        else:
            v_c_transformer = Transformer_co4(label_params, in_training, scope="v_c_transformer")

        viseme_labels = labels['viseme']
        sparse_viseme, viseme_string = self.procesess_label(viseme_labels, self.viseme_dic)
        viseme_char_list_labels = label_util.string2char_list(viseme_string)

        pinyin_labels = tf.squeeze(labels['pinyin'])  # [batch_size, ]
        pinyin_char_list_labels,pinyin_labels = self.preprocess_labels(pinyin_labels,self.pinyin_dic)  # [batch_size, target_length]
        pinyin_string = self.id_to_string(pinyin_labels,self.pinyin_dic)


        label_targets = labels['label']
        label_index, label_string = self.procesess_label(label_targets,self.label_dic)

        pinyin_logits, viseme_logits, encode, attention_bias= v_p_transformer(inputs, inputs_unpadded_length, pinyin_labels,viseme_labels)
        pinyin_sequence = tf.argmax(pinyin_logits, 2)
        viseme_sequence = tf.argmax(viseme_logits, 2)
        pinyin_embedded = tf.contrib.layers.embed_sequence(ids=pinyin_sequence, vocab_size=params["target_pinyin_vocab_size"],
                                                           embed_dim=512)
        viseme_embedded = tf.contrib.layers.embed_sequence(ids=viseme_sequence,
                                                           vocab_size=params["target_viseme_vocab_size"],
                                                           embed_dim=512)

        # Calculate model loss.
        # xentropy contains the cross entropy loss of every nonpadding token in the

        # train

        pinyin_xentropy, pinyin_weights = metrics.padded_cross_entropy_loss(
            pinyin_logits, pinyin_labels, params["label_smoothing"], params["pinyin_vocab_size"])
        pinyin_loss = tf.reduce_sum(pinyin_xentropy) / tf.reduce_sum(pinyin_weights)

        viseme_xentropy, viseme_weights = metrics.padded_cross_entropy_loss(
            viseme_logits, viseme_labels, params["label_smoothing"], params["viseme_vocab_size"])
        viseme_loss = tf.reduce_sum(viseme_xentropy) / tf.reduce_sum(viseme_weights)

        #embedded =  tf.concat([viseme_embedded, pinyin_embedded], 1)

        pinyin_unpadded_length = tf.cast(tf.count_nonzero(pinyin_sequence, 1, keepdims=True) - 1, tf.int32)
        viseme_unpadded_length = tf.cast(tf.count_nonzero(viseme_sequence, 1, keepdims=True) - 1, tf.int32)


        # label_logits = v_c_transformer(tf.cast(viseme_embedded, tf.float32), inputs_unpadded_length, label_targets)
        #label_logits = v_c_transformer(tf.cast(embedded, tf.float32), inputs_unpadded_length,  encode, attention_bias,label_targets)

        label_logits = v_c_transformer(tf.cast(pinyin_embedded, tf.float32), pinyin_unpadded_length,
                                       tf.cast(viseme_embedded, tf.float32), viseme_unpadded_length,
                                       encode, attention_bias,label_targets)

        label_xentropy, label_weights = metrics.padded_cross_entropy_loss(
            label_logits, label_targets, params["label_smoothing"], label_params["vocab_size"])
        label_loss = tf.reduce_sum(label_xentropy) / tf.reduce_sum(label_weights)

        loss = label_loss + pinyin_loss + viseme_loss



        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op, metric_dict = get_train_op_and_metrics(loss, params)

            # if params["ckpt_path"] != "":
                # print('restore from: {}'.format(params["ckpt_path"]))
                # tf.train.init_from_checkpoint(
                    # params["ckpt_path"], assignment_map={"/": "/"})

            # Epochs can be quite long. This gives some intermediate information
            # in TensorBoard.
            metric_dict["minibatch_loss"] = loss
            record_scalars(metric_dict)

            pinyin_sequence = tf.argmax(pinyin_logits, 2)
            viseme_sequence = tf.argmax(viseme_logits, 2)
            label_sequence = tf.argmax(label_logits, 2)

            sparse_pinyin_prediction, pinyin_predicted_string = self.procesess_label(pinyin_sequence, self.pinyin_dic)
            pinyin_predicted_char_list = label_util.string2char_list(pinyin_predicted_string)

            sparse_viseme_prediction, viseme_predicted_string = self.procesess_label(viseme_sequence, self.viseme_dic)
            viseme_predicted_char_list = label_util.string2char_list(viseme_predicted_string)

            label_predicted_index, label_predicted_string = self.procesess_label(label_sequence, self.label_dic)

            ver = self.cal_pinyin_metrics(viseme_char_list_labels, viseme_predicted_char_list)
            per = self.cal_pinyin_metrics(pinyin_char_list_labels, pinyin_predicted_char_list)
            cer = tf.edit_distance(label_index, tf.cast(label_predicted_index, tf.int64))

            logging_hook = tf.train.LoggingTensorHook(
                {
                    'loss': loss,
                    'ver': tf.reduce_mean(ver),
                    'per': tf.reduce_mean(per),
                    'cer': tf.reduce_mean(cer),
                },
                every_n_iter=1,
            )

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op,training_hooks=[logging_hook]
            )

            # Save loss as named tensor that will be logged with the logging hook.
            tf.identity(loss, "cross_entropy")

        # eval

        if mode == tf.estimator.ModeKeys.EVAL:
            pinyin_logits, viseme_logits, encode, attention_bias = v_p_transformer(inputs, inputs_unpadded_length, None,None)

            label_logits = v_c_transformer(tf.cast(pinyin_embedded, tf.float32), pinyin_unpadded_length,
                                           tf.cast(viseme_embedded, tf.float32), viseme_unpadded_length,
                                           encode, attention_bias, None)

            predicted_pinyin = pinyin_logits['outputs']
            sparse_pinyin_prediction, pinyin_predicted_string = self.procesess_label(predicted_pinyin, self.pinyin_dic)
            pinyin_predicted_char_list = label_util.string2char_list(pinyin_predicted_string)

            predicted_viseme = viseme_logits['outputs']
            sparse_viseme_prediction, viseme_predicted_string = self.procesess_label(predicted_viseme, self.viseme_dic)
            viseme_predicted_char_list = label_util.string2char_list(viseme_predicted_string)

            label_predictions = label_logits['outputs']
            label_predicted_index, label_predicted_string = self.procesess_label(label_predictions,self.label_dic)

            ver = self.cal_pinyin_metrics(viseme_char_list_labels,viseme_predicted_char_list)
            per = self.cal_pinyin_metrics(pinyin_char_list_labels, pinyin_predicted_char_list)
            cer = tf.edit_distance(label_index, tf.cast(label_predicted_index, tf.int64))
            tf.summary.scalar('ver', tf.reduce_mean(ver))
            tf.summary.scalar('per', tf.reduce_mean(per))
            tf.summary.scalar('cer', tf.reduce_mean(cer))


            eval_metric_ops = {
                'ver': tf.metrics.mean(ver),
                'per': tf.metrics.mean(per),
                'cer': tf.metrics.mean(cer),
            }

            def custom_formatter(tensors):
                hook_list = [
                    'predicted_sentence',
                    'sentence_label'
                ]
                ostrs = []
                for k, v in tensors.items():
                    if k in hook_list:
                        v = [str(vv, encoding='UTF8') for vv in v]
                    ostrs.append('{}: {}'.format(k, v))
                return '\n'.join(ostrs)

            logging_hook = tf.train.LoggingTensorHook(
                {
                    'loss': loss,
                    'ver': tf.reduce_mean(ver),
                    'per': tf.reduce_mean(per),
                    'cer': tf.reduce_mean(cer),
                    'viseme_labels': viseme_string[:5],
                    'pinyin_labels': pinyin_string[:5],
                    'sentence_label': label_string[:5],
                    'predicted_viseme': viseme_predicted_string[:5],
                    'predicted_pinyin': pinyin_predicted_string[:5],
                    'predicted_sentence': label_predicted_string[:5],
                },
                every_n_iter=10,
                formatter=custom_formatter
                )

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                predictions={"predictions": val[0]},
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=[logging_hook]
            )


def record_scalars(metric_dict):
    for key, value in metric_dict.items():
        tf.summary.scalar(name=key, tensor=value)


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size**-0.5)
        # Apply linear warmup
        # step /= 10.0
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        # learning_rate *= 0.1
        # Apply rsqrt decay
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))
        print (learning_rate)
        # Create a named tensor that will be logged using the logging hook.
        # The full name includes variable and names scope. In this case, the name
        # is model/get_train_op/learning_rate/learning_rate
        tf.identity(learning_rate, "learning_rate")

        return learning_rate


def get_train_op_and_metrics(loss, params):
    """Generate training op and metrics to save in TensorBoard."""
    with tf.variable_scope("get_train_op"):
        learning_rate = get_learning_rate(
           learning_rate=params["learning_rate"],
           hidden_size=params["hidden_size"],
           learning_rate_warmup_steps=params["learning_rate_warmup_steps"])
        #learning_rate = 0.000005

        # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
        # than the TF core Adam optimizer.
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
             learning_rate,
             beta1=params["optimizer_adam_beta1"],
             beta2=params["optimizer_adam_beta2"],
             epsilon=params["optimizer_adam_epsilon"])
        #optimizer = tf.train.RMSPropOptimizer(0.0005, decay=0.1)

        # Calculate and apply gradients using LazyAdamOptimizer.
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        #print (tvars)
        #tvars = [t for t in tvars if (t.name.startswith("cnn_feature_extractor") or t.name.startswith("v_v_transformer"))]
        #tvars = [t for t in tvars if
        #          (t.name.startswith("v_c_transformer") )]
        gradients = optimizer.compute_gradients(
            loss, tvars, colocate_gradients_with_ops=True)
        minimize_op = optimizer.apply_gradients(
            gradients, global_step=global_step, name="train")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

        train_metrics = {"learning_rate": learning_rate}

        # gradient norm is not included as a summary when running on TPU, as
        # it can cause instability between the TPU and the host controller.
        gradient_norm = tf.global_norm(list(zip(*gradients))[0])
        train_metrics["global_norm/gradient_norm"] = gradient_norm

        return train_op, train_metrics


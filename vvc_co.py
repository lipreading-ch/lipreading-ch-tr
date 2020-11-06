#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import string
import argparse
import tensorflow as tf
from collections import defaultdict

from vvc_dataset import tfrecord_input_fn as input_fn
from models.vvc_co_tr import TransformerEstimator
from transformer.utils.tokenizer import RESERVED_TOKENS, PAD, PAD_ID, EOS, EOS_ID, RESERVED_TOKENS
import vocab_dict


CURRENT_FILE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='either train, eval, predict',default='train')

    # dataset path
    parser.add_argument(
        '--data_dir',
        help='directory of the tfrecord files',
        default='/data/tf/')

    # train
    parser.add_argument(
        '--save_steps',
        type=int,
        default=465,
        help='steps interval to save checkpoint')
    parser.add_argument('--model_dir', help='directory to save checkpoints',
                        default='/data/ckpts')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help=
        'batch size. If train with multi_gpu, each of which will be fed with batch_size samples'
    )

    # eval
    parser.add_argument(
        '--eval_steps', type=int, default=1,help='steps to eval')
    # eval and predict
    parser.add_argument(
        '--ckpt_path', help='checkpoints to evaluate/predict', default="/data/ckpts/co2_3block/model.ckpt-117180")

    # misc
    parser.add_argument('-gpu', '--gpu', help='gpu id to use', default='0')
    parser.add_argument('-bw', '--beam_width', type=int, default=4)
    parser.add_argument('-co_num',type = int, default = 2)
    return parser.parse_args()


def transformer_params():
    """get default transformer params.
    Returns: Dict

    """
    dic, vocab_size = vocab_dict.get_dict()

    viseme_dic = RESERVED_TOKENS +\
                 ['b', 'f', 'd', 'l', 'g', 'j', 'zh', 'z',
                  'B', 'F', 'D', 'L', 'G', 'J', 'ZH', 'Z',
                  'a', 'an', 'ao', 'o', 'ou', 'e', 'en', 'er', 'i', 'u', 'v', 'i1', 'i2', ' ']

    pinyin_dic = RESERVED_TOKENS + list(string.ascii_lowercase) + [' ']

    return defaultdict(
        lambda: None,
        # Model params
        viseme_dic = viseme_dic,
        pinyin_dic = pinyin_dic,
        label_dic = dic,
        initializer_gain=1.0,  # Used in trainable variable initialization.
        hidden_size=512,  # Model dimension in the hidden layers.
        num_hidden_layers=3,  # Number of layers in the encoder and decoder stacks.
        num_heads=8,  # Number of heads to use in multi-headed attention.
        filter_size=1024,  # Inner layer dimension in the feedforward network.

        # Dropout values (only used when training)
        layer_postprocess_dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1,

        # Training params
        label_smoothing=0.1,
        learning_rate=2.0,
        learning_rate_decay_rate=1.0,
        learning_rate_warmup_steps=16000,

        # Optimizer params
        optimizer_adam_beta1=0.9,
        optimizer_adam_beta2=0.997,
        optimizer_adam_epsilon=1e-09,

        # Default prediction params
        extra_decode_length=50,
        beam_size=4,
        alpha=0.6,  # used to calculate length normalization in beam search
        allow_ffn_pad=True,

    )


def main():
    args = arg_parse()
    if args.gpu != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    tf.logging.set_verbosity(tf.logging.INFO)

    multi_gpu = len(args.gpu.split(',')) > 1
    # build estimator
    run_config = TransformerEstimator.get_runConfig(
        args.model_dir,
        args.save_steps,
        multi_gpu=multi_gpu,
        keep_checkpoint_max=100)


    model_parms = transformer_params()
    model_parms.update({
        # lipnet
        'feature_extractor': 'lip',
        'feature_len': 256,

        # learn
        'batch_size': args.batch_size,
        'ckpt_path': args.ckpt_path,
        'beam_size': args.beam_width,
        
        'co_attention':args.co_num
    })

    # ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from="/data/users/shihuima/sentence/ckpts/co2_3block/model.ckpt-99510")
    #
    # extra_params = {
    #    'warm_start_from':ws
    # }
    #
    # model = TransformerEstimator(model_parms, run_config, **extra_params)
    model = TransformerEstimator(model_parms, run_config)

    # build input
    train_file = os.path.join(args.data_dir, 'val*.tfrecord')
    if args.mode == 'eval':
        test_file = os.path.join(args.data_dir, 'test*.tfrecord' )
    else:
        test_file = os.path.join(args.data_dir, 'val*.tfrecord')

    train_input_params = {
        'num_epochs': 100000,
        'batch_size': args.batch_size,
        'num_threads': 1,
        'aug': False,
        'file_name_pattern': train_file
    }
    eval_input_params = {
        'num_epochs': 1,
        'batch_size':args.batch_size,
        'num_threads': 4,
        'aug': False,
        'file_name_pattern': train_file
    }
    print('train_input_params: {}'.format(train_input_params))
    print('eval_input_params: {}'.format(eval_input_params))
    train_input_fn = lambda: input_fn(mode=tf.estimator.ModeKeys.TRAIN, **train_input_params)
    eval_input_fn = lambda: input_fn(mode=tf.estimator.ModeKeys.EVAL, **eval_input_params)

    #begin train,eval,predict
    if args.mode == 'train':
        model.train_and_evaluate(
            train_input_fn,
            eval_input_fn,
            eval_steps=args.eval_steps,
            throttle_secs=1000)
    elif args.mode == 'eval':
        res = model.evaluate(
            eval_input_fn,
            steps=args.eval_steps,
            checkpoint_path=args.ckpt_path)
        print(res)
    elif args.mode == 'predict':
        model.predict(eval_input_fn, checkpoint_path=args.ckpt_path)
    else:
        raise ValueError(
            'arg mode should be one of "train", "eval", "predict"')


if __name__ == "__main__":
    main()


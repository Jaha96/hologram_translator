#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import os 
import sys


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    # src_shards = split_corpus(opt.src, opt.shard_size)
    # tgt_shards = split_corpus(opt.tgt, opt.shard_size)
    # shard_pairs = zip(src_shards, tgt_shards)

    # for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
    #     logger.info("Translating shard %d." % i)
    #     all_scores, all_predictions = translator.translate(
    #         src=src_shard,
    #         tgt=tgt_shard,
    #         src_dir=opt.src_dir,
    #         batch_size=opt.batch_size,
    #         batch_type=opt.batch_type,
    #         attn_debug=opt.attn_debug,
    #         align_debug=opt.align_debug
    #         )

    src_shard = [opt.src]
    all_scores, all_predictions = translator.translate(
            src=src_shard,
            tgt=None,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug
            )
    
    text = ""
    for prediction in all_predictions:
        if len(prediction) > 0:
            print(prediction[0].replace(" ", ""))
            text = prediction[0].replace(" ", "")
    return text


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


def main(text = "Translator"):
    print ("the script has the name", sys.argv[0])
    sys.argv = [sys.argv[0]]

    dir_path = os.getcwd()
    sys.argv.append("-model")
    sys.argv.append(os.path.join(dir_path, "OpenNMT", "data", "model", "en-ja-model_step_100000.pt"))
    sys.argv.append("--src")
    sys.argv.append(text.lower())
    sys.argv.append("--n_best")
    sys.argv.append("5")
    sys.argv.append("-replace_unk")
    sys.argv.append("-verbose")

    print(sys.argv)
    

    parser = _get_parser()
    opt = parser.parse_args()
    
    return translate(opt)


if __name__ == "__main__":
    main()

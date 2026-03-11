# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import sys
import logging
import functools
from termcolor import colored

_LOG_FORMAT = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
_BOUND_EVAL_PARENT_NAME = None


def _build_formatter(use_color=False):
    if use_color:
        fmt = colored('[%(asctime)s %(name)s]', 'green') + \
            colored('(%(filename)s %(lineno)d)', 'yellow') + \
            ': %(levelname)s %(message)s'
    else:
        fmt = _LOG_FORMAT
    return logging.Formatter(fmt=fmt, datefmt=_DATE_FORMAT)


def _clear_handlers(logger):
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _build_console_handler(use_color=True):
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(_build_formatter(use_color=use_color))
    return console_handler


def bind_eval_logger(parent_logger=None):
    global _BOUND_EVAL_PARENT_NAME
    if parent_logger is None:
        _BOUND_EVAL_PARENT_NAME = None
    elif isinstance(parent_logger, logging.Logger):
        _BOUND_EVAL_PARENT_NAME = parent_logger.name
    else:
        _BOUND_EVAL_PARENT_NAME = str(parent_logger)

    return get_eval_logger()


def get_eval_logger():
    if _BOUND_EVAL_PARENT_NAME:
        eval_logger = logging.getLogger(f'{_BOUND_EVAL_PARENT_NAME}.eval')
        if eval_logger.handlers:
            _clear_handlers(eval_logger)
        eval_logger.setLevel(logging.INFO)
        eval_logger.propagate = True
        return eval_logger

    eval_logger = logging.getLogger('eval')
    eval_logger.setLevel(logging.INFO)
    eval_logger.propagate = False
    if not eval_logger.handlers:
        eval_logger.addHandler(_build_console_handler(use_color=True))
    return eval_logger


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    _clear_handlers(logger)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create console handlers for master process
    if dist_rank == 0:
        logger.addHandler(_build_console_handler(use_color=True))

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(
        output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_build_formatter(use_color=False))
    logger.addHandler(file_handler)

    return logger

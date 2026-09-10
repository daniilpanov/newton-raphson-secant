import json
import logging
import os
import sys
from typing import Literal

import numpy as np

DEFAULT_LOG_FILE = 'log.ndjson'

LogKind = Literal[
    'start', 'iter', 'hess', 'direction', 'bracket_start', 'bracket',
    'sect', 'sect_denzero', 'sect_final', 'backtrack', 'update', 'stop',
]


def _to_jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v


def _json_event(name, kind, fields):
    payload = {'name': name, 'type': kind}
    for key, value in fields.items():
        payload[key] = _to_jsonable(value)
    return json.dumps(payload, separators=(',', ':'))


def log_enabled(env=None):
    env = os.environ if env is None else env
    return (env.get('LOG_MODE') or 'no').strip().lower() != 'no'


def make_logger(name, env=None):
    """
    - name: the name of run
    - LOG_MODE: no | console | file | both (no by default)
    - LOG_FILE: path to output file if LOG_MODE is file or both (trace.ndjson by default)
    """
    env = os.environ if env is None else env
    mode = (env.get('LOG_MODE') or 'no').strip().lower()
    fmt = logging.Formatter('%(message)s')
    handlers = []
    if mode in ('console', 'both'):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(fmt)
        handlers.append(handler)
    if mode in ('file', 'both'):
        handler = logging.FileHandler(
            env.get('LOG_FILE') or DEFAULT_LOG_FILE,
            mode='a', encoding='utf-8')
        handler.setFormatter(fmt)
        handlers.append(handler)
    return Logger(name, handlers)


class Logger:
    def __init__(self, name, handlers=()):
        self.name = name
        self._log = logging.Logger(name)
        self._log.propagate = False
        self._handlers = list(handlers)
        for handler in self._handlers:
            self._log.addHandler(handler)
        self._enabled = bool(self._handlers)

    def log(self, kind: LogKind, fields: dict | None = None):
        if not self._enabled:
            return
        self._log.info(_json_event(self.name, kind, fields or {}))

    def close(self):
        for handler in self._handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()
            else:
                handler.flush()
        self._log.handlers.clear()
        self._handlers.clear()
        self._enabled = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


__all__ = ['Logger', 'make_logger', 'log_enabled', 'LogKind', 'DEFAULT_LOG_FILE']
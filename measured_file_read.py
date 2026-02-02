#!/usr/bin/env python

import hashlib
import io
import time
import os
import pyarrow as pa
from datasets import Dataset

class MeasuredBytesIO(io.BytesIO):
    def __init__(self, initial, hasher, *args, **kwargs):
        self.hasher = hasher
        self.hasher.update(initial)
        super().__init__(initial, *args, **kwargs)

class MeasuredStringIO(io.StringIO):
    def __init__(self, initial, hasher, *args, **kwargs):
        self.hasher = hasher
        self.hasher.update(bytes(initial, 'utf8'))
        super().__init__(initial, *args, **kwargs)

def open_measured(path, mode, hasher=None):
    if not hasher:
        hasher = hashlib.sha256()

    if 'w' in mode or 'a' in mode or 'x' in mode or '+' in mode:
        raise IOError("Measured open can't be writable")

    if 'b' in mode:
        load_time_start = time.time()
        with open(path, mode) as fh:
            data = fh.read()
            load_time_end = time.time()
            load_time = load_time_end - load_time_start

            measure_time_start = time.time()
            buffer = MeasuredBytesIO(data, hasher)
            measure_time_end = time.time()
            measure_time = measure_time_end - measure_time_start
                
        return buffer, hasher.hexdigest(), load_time, measure_time
            
    else:
        load_time_start = time.time()
        with open(path, mode) as fh:
            data = fh.read()
            load_time_end = time.time()
            load_time = load_time_end - load_time_start
            return MeasuredStringIO(data, hasher), load_time

def open_measured_arrow(path, mode, hasher=None):
    if not hasher:
        hasher = hashlib.sha256()

    if 'w' in mode or 'a' in mode or 'x' in mode or '+' in mode:
        raise IOError("Measured open can't be writable")

    if 'b' in mode:
        load_time_start = time.time()
        in_memory_stream = pa.input_stream(path)
        # print(type(in_memory_stream))
        # opened_stream = pa.ipc.open_stream(in_memory_stream)
        # pa_table = opened_stream.read_all()
        data = in_memory_stream.read()
        measure_time_start = time.time()
        buffer = MeasuredBytesIO(data, hasher)
        measure_time_end = time.time()
        measure_time = measure_time_end - measure_time_start
        # buffer = pa.py_buffer(buffer.getvalue())
        load_time_end = time.time()
        load_time = load_time_end - load_time_start
        return buffer, hasher.hexdigest(), load_time, measure_time
            
    else:
        load_time_start = time.time()
        with open(path, mode) as fh:
            data = fh.read()
            load_time_end = time.time()
            load_time = load_time_end - load_time_start
            return MeasuredStringIO(data, hasher), load_time

class MeasuredBytesIOWrite(io.BytesIO):
    def __init__(self, hasher, *args, **kwargs):
        self.hasher = hasher
        super().__init__(*args, **kwargs)
    
    def write(self, b):
        self.hasher.update(b)
        return super().write(b)

class MeasuredStringIOWrite(io.StringIO):
    def __init__(self, hasher, *args, **kwargs):
        self.hasher = hasher
        super().__init__(*args, **kwargs)
    
    def write(self, s):
        self.hasher.update(s.encode('utf-8'))
        return super().write(s)

def open_measured_write(path, mode, hasher=None):
    if not hasher:
        hasher = hashlib.sha256()
    
    # Check for valid write modes
    if 'r' in mode:
        raise IOError("Measured open must be writable")

    if 'b' in mode:
        return MeasuredBytesIOWrite(hasher), open(path, mode)
    else:
        return MeasuredStringIOWrite(hasher), open(path, mode)
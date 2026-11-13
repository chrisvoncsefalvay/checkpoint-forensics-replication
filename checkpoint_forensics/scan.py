"""Resumable, bounded scans of public Hugging Face safetensors checkpoints."""
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse

import numpy as np
from .fetch import json_url, read_range
from .streaming import ITEMSIZE, score_rows, merge_statistics, correlation_from_statistics


class ScanLimit(Exception):
    pass


def save_state(path, state):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def validate_config(config):
    for field in ('seeds', 'sms', 'offsets'):
        if not config.get(field) or any(not isinstance(x, int) for x in config[field]):
            raise ValueError(f'{field} must be a nonempty list of integers')
    if any(x < 0 or x >= 2**64 for x in config['seeds']):
        raise ValueError('Seeds must be unsigned 64-bit integers')
    if any(x <= 0 for x in config['sms']):
        raise ValueError('SM counts must be positive')
    if any(x < 0 or x % 4 for x in config['offsets']):
        raise ValueError('Offsets must be nonnegative multiples of four')
    if config.get('mapping') not in ('pytorch-contiguous', 'single-launch'):
        raise ValueError('Choose pytorch-contiguous or single-launch mapping')
    if config.get('init_dtype') not in (None, *ITEMSIZE):
        raise ValueError('Initialisation dtype must be BF16, F16 or F32')
    if int(config.get('max_threads_per_sm', 2048)) < 256:
        raise ValueError('At least 256 threads per SM required')


def decode(raw, dtype):
    if dtype == 'BF16':
        return (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(raw, dtype={'F16': '<f2', 'F32': '<f4'}[dtype]).astype(np.float32)


def scan(args):
    """Limits are cumulative across resumptions, except wall time per invocation."""
    path = Path(args.state)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The lock is released by the OS even after an interrupted process.
    import fcntl
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process is using this scan state') from error
        return _scan_locked(args, path)


def _scan_locked(args, path):
    from .cli import ints
    defaults = dict(seeds=ints(args.seeds), sms=ints(args.sms), offsets=ints(args.offsets),
                    init_dtype=args.init_dtype, mapping=args.mapping,
                    max_threads_per_sm=args.max_threads_per_sm)
    overrides = json.loads(Path(args.tensor_config).read_text()) if args.tensor_config else {}
    validate_config(defaults)
    for override in overrides.values():
        validate_config({**defaults, **override})
    identity = dict(repo=args.repo, tensor_pattern=args.tensor, defaults=defaults,
                    tensor_overrides=overrides, columns=args.columns,
                    threshold=args.threshold, start_row=args.start_row)
    if path.exists():
        state = json.loads(path.read_text())
        if state['config'] != identity:
            raise ValueError('Scan configuration differs from saved state; use another state file')
        if args.revision not in ('main', state['revision'], state.get('requested_revision')):
            raise ValueError('Revision differs from the saved checkpoint')
        if args.keep_hits != state.get('keep_hits', 100):
            raise ValueError('--keep-hits differs from saved state; use another state file')
    else:
        info = json_url(f'https://huggingface.co/api/models/{args.repo}/revision/{args.revision}')
        revision = info['sha']
        base = f'https://huggingface.co/{args.repo}/resolve/{revision}/'
        try:
            weights = json_url(base + 'model.safetensors.index.json')['weight_map']
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            weights = None
        state = dict(version=1, config=identity, revision=revision, requested_revision=args.revision,
                     keep_hits=args.keep_hits, weight_map=weights,
                     headers={}, tensors={}, rows_scanned=0, weight_bytes_read=0,
                     metadata_bytes_read=0, status='ready')
        save_state(path, state)
    base = f"https://huggingface.co/{args.repo}/resolve/{state['revision']}/"
    started = time.monotonic()

    def check_time():
        if args.max_seconds and time.monotonic() - started >= args.max_seconds:
            raise ScanLimit('time limit')

    def header(shard):
        if shard not in state['headers']:
            import struct
            check_time()
            url = base + urllib.parse.quote(shard)
            length = struct.unpack('<Q', read_range(url, 0, 8))[0]
            if length > 100_000_000:
                raise ValueError('Safetensors header exceeds 100 MB')
            check_time()
            contents = json.loads(read_range(url, 8, length))
            state['headers'][shard] = {'length': length, 'tensors': contents}
            state['metadata_bytes_read'] += length + 8
            save_state(path, state)
        return state['headers'][shard]

    try:
        if state['weight_map'] is None:
            data = header('model.safetensors')
            state['weight_map'] = {name: 'model.safetensors' for name in data['tensors'] if name != '__metadata__'}
        selected = sorted(name for name in state['weight_map'] if re.search(args.tensor, name))
        if not selected:
            raise ValueError('No checkpoint tensors match --tensor')
        state['selected_tensor_count'] = len(selected)
        for name in selected:
            record = state['tensors'].get(name)
            if record and record['status'] in ('complete', 'skipped'):
                continue
            check_time()
            if record is None and args.max_tensors and len(state['tensors']) >= args.max_tensors:
                raise ScanLimit('tensor limit')
            shard = state['weight_map'][name]
            data = header(shard)
            meta = data['tensors'][name]
            shape, dtype = meta['shape'], meta['dtype']
            if record is None:
                record = dict(shape=shape, dtype=dtype, shard=shard, next_row=args.start_row,
                              rows_scanned=0, status='scanning', candidates={})
                state['tensors'][name] = record
            if dtype not in ITEMSIZE or not shape or math.prod(shape) == 0:
                record.update(status='skipped', reason='Only nonempty, non-scalar BF16/F16/F32 tensors are supported')
                save_state(path, state)
                continue
            logical_shape = shape if len(shape) > 1 else [1, shape[0]]
            row_count = logical_shape[0]
            record['scan_shape'] = logical_shape
            config = {**defaults, **overrides.get(name, {})}
            config['init_dtype'] = config['init_dtype'] or dtype
            record['initialisation'] = config
            width = math.prod(logical_shape[1:])
            row_bytes = width * ITEMSIZE[dtype]
            expected_bytes = math.prod(shape) * ITEMSIZE[dtype]
            if meta['data_offsets'][1] - meta['data_offsets'][0] != expected_bytes:
                raise ValueError(f'Tensor byte size mismatch: {name}')
            while record['next_row'] < row_count:
                check_time()
                count = min(args.batch_rows, args.batch_bytes // row_bytes, row_count - record['next_row'])
                if row_bytes > args.batch_bytes:
                    raise ScanLimit('one row exceeds --batch-bytes; increase the batch limit')
                if args.max_rows:
                    count = min(count, args.max_rows - state['rows_scanned'])
                if args.max_weight_bytes:
                    count = min(count, (args.max_weight_bytes - state['weight_bytes_read']) // row_bytes)
                if count <= 0:
                    raise ScanLimit('row or weight-byte limit')
                first = record['next_row']
                start = 8 + data['length'] + meta['data_offsets'][0] + first * row_bytes
                raw = read_range(base + urllib.parse.quote(shard), start, count * row_bytes)
                state['weight_bytes_read'] += len(raw)
                # Count completed downloads even if scoring is subsequently interrupted.
                save_state(path, state)
                values = decode(raw, dtype).reshape(count, width)
                rows = np.arange(first, first + count)
                scores = score_rows(values, rows, logical_shape, config,
                                    columns=args.columns, threshold=args.threshold)
                for key, result in scores.items():
                    previous = record['candidates'].setdefault(key, {'statistics': None, 'hits': [], 'hit_count': 0})
                    previous['statistics'] = merge_statistics(previous['statistics'], result['statistics'])
                    previous['r'] = correlation_from_statistics(previous['statistics'])
                    previous['hit_count'] += len(result['hits'])
                    # Keep a bounded ranking; aggregate statistics cover all scanned rows.
                    previous['hits'] = sorted(previous['hits'] + result['hits'],
                        key=lambda x: x['r'], reverse=True)[:args.keep_hits]
                record['next_row'] += count
                record['rows_scanned'] += count
                state['rows_scanned'] += count
                save_state(path, state)
                if not args.quiet:
                    print(f'{name}: {record["next_row"]}/{row_count} rows; '
                          f'{state["weight_bytes_read"]:,} checkpoint bytes read', flush=True)
            record['status'] = 'complete'
            save_state(path, state)
        state['status'] = 'complete'
        state.pop('stop_reason', None)
    except ScanLimit as error:
        state.update(status='paused', stop_reason=str(error))
    except KeyboardInterrupt:
        state = json.loads(path.read_text())
        state.update(status='paused', stop_reason='interrupted')
    except Exception as error:
        state = json.loads(path.read_text())
        state.update(status='error', stop_reason=f'{type(error).__name__}: {error}')
        save_state(path, state)
        raise
    state['last_run_seconds'] = time.monotonic() - started
    save_state(path, state)
    return {key: state[key] for key in ('revision', 'status', 'rows_scanned', 'weight_bytes_read', 'metadata_bytes_read')} | {
        'stop_reason': state.get('stop_reason'), 'state': str(path),
        'skipped_tensors': {name: record['reason'] for name, record in state['tensors'].items() if record['status'] == 'skipped'},
        'tensors': {name: {'status': record['status'], 'next_row': record['next_row'],
                          'rows_scanned': record['rows_scanned']} for name, record in state['tensors'].items()}}

"""Sparse candidates and mergeable scores for contiguous checkpoint tensors."""
import math
import numpy as np
from .analysis import corr
from .philox import normal_at, thread_count

ITEMSIZE = {'BF16': 2, 'F16': 2, 'F32': 4}


def initialise_segments(numel, sms, offset, *, itemsize=2, max_threads_per_sm=2048):
    """PyTorch distribution_nullary_kernel's contiguous 32-bit iterator launches.

    The outer call reserves RNG state before testing indexing. The iterator then
    splits into valid leaves; only the outer call and leaf calls reserve state.
    """
    def fits(count):
        return count <= 2**31 - 1 and (count - 1) * itemsize + 1 <= 2**31 - 1

    def advance(count):
        threads = thread_count(count, sms, max_threads_per_sm)
        return ((count - 1) // (4 * threads) + 1) * 4

    if fits(numel):
        return [(0, numel, offset)]
    leaves = []

    def split(start, count):
        if fits(count):
            leaves.append((start, count))
        else:
            half = count // 2
            split(start, half)
            split(start + half, count - half)

    split(0, numel)
    current = offset + advance(numel)
    segments = []
    for start, count in leaves:
        segments.append((start, count, current))
        current += advance(count)
    return segments


def candidate_values(indices, shape, seed, sms, offset, config):
    numel = math.prod(map(int, shape))
    indices = np.asarray(indices, dtype=np.int64)
    init_numel = int(config.get('init_numel', numel))
    index_shift = int(config.get('index_shift', 0))
    indices = indices + index_shift
    if np.any(indices < 0) or np.any(indices >= init_numel):
        raise ValueError('Checkpoint indices do not fit the configured initialisation tensor')
    max_threads = int(config.get('max_threads_per_sm', 2048))
    mapping = config.get('mapping', 'pytorch-contiguous')
    if mapping == 'single-launch':
        segments = [(0, init_numel, offset)]
    elif mapping == 'pytorch-contiguous':
        segments = initialise_segments(init_numel, sms, offset,
            itemsize=ITEMSIZE[config['init_dtype']], max_threads_per_sm=max_threads)
    else:
        raise ValueError(f'Unknown initialisation mapping: {mapping}')
    result = np.empty(indices.shape, dtype=np.float32)
    for start, count, rng_offset in segments:
        mask = (indices >= start) & (indices < start + count)
        if mask.any():
            result[mask] = normal_at(indices[mask] - start, seed=seed, sms=sms,
                numel=count, offset=rng_offset, max_threads_per_sm=max_threads)
    return result


def statistics(x, y):
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    dx, dy = x - x.mean(), y - y.mean()
    return dict(n=len(x), mean_x=float(x.mean()), mean_y=float(y.mean()),
                m2_x=float(dx @ dx), m2_y=float(dy @ dy), c_xy=float(dx @ dy))


def merge_statistics(a, b):
    if a is None:
        return dict(b)
    n = a['n'] + b['n']
    dx, dy = b['mean_x'] - a['mean_x'], b['mean_y'] - a['mean_y']
    weight = a['n'] * b['n'] / n
    return dict(n=n, mean_x=a['mean_x'] + dx*b['n']/n, mean_y=a['mean_y'] + dy*b['n']/n,
                m2_x=a['m2_x']+b['m2_x']+dx*dx*weight,
                m2_y=a['m2_y']+b['m2_y']+dy*dy*weight,
                c_xy=a['c_xy']+b['c_xy']+dx*dy*weight)


def correlation_from_statistics(s):
    denominator = math.sqrt(max(0, s['m2_x']) * max(0, s['m2_y']))
    return s['c_xy'] / denominator if denominator else None


def score_rows(values, rows, shape, config, *, columns=512, threshold=.99):
    width = math.prod(map(int, shape[1:])) if len(shape) > 1 else 1
    cols = np.arange(width) if columns == 0 else np.unique(np.linspace(0, width-1, min(columns, width), dtype=int))
    observed = np.asarray(values).reshape(len(rows), width)[:, cols]
    if not np.isfinite(observed).all():
        raise ValueError('Non-finite checkpoint values')
    indices = (np.asarray(rows)[:, None] * width + cols).ravel()
    scores = {}
    for seed in config['seeds']:
        for sms in config['sms']:
            for offset in config['offsets']:
                predicted = candidate_values(indices, shape, seed, sms, offset, config).reshape(observed.shape)
                stats = statistics(observed, predicted)
                hits = []
                for row, x, y in zip(rows, observed, predicted):
                    r = corr(x, y)
                    if r is not None and r >= threshold:
                        hits.append({'row': int(row), 'r': r})
                scores[f'{seed}:{sms}:{offset}'] = {
                    'statistics': stats, 'r': correlation_from_statistics(stats), 'hits': hits}
    return scores

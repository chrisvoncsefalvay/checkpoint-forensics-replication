"""Philox4x32-10 and sparse CUDA grid-stride normal candidates.

Integer stream is exact. CPU transcendental functions are approximate relative to
cuRAND/libdevice; use correlations, not bitwise vendor claims, with this backend.
"""
import numpy as np

MASK = np.uint64(0xffffffff)

def philox(seed, subsequence, block):
    sub = np.asarray(subsequence, dtype=np.uint64)
    block = np.broadcast_to(np.asarray(block, dtype=np.uint64), sub.shape)
    c0, c1 = block & MASK, block >> np.uint64(32)
    c2, c3 = sub & MASK, sub >> np.uint64(32)
    k0, k1 = np.uint64(seed & 0xffffffff), np.uint64((seed >> 32) & 0xffffffff)
    for i in range(10):
        a, b = c0 * np.uint64(0xD2511F53), c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = (b >> np.uint64(32)) ^ c1 ^ k0, b & MASK, (a >> np.uint64(32)) ^ c3 ^ k1, a & MASK
        if i != 9:
            k0 = (k0 + np.uint64(0x9E3779B9)) & MASK
            k1 = (k1 + np.uint64(0xBB67AE85)) & MASK
    return np.stack([c0,c1,c2,c3], axis=-1).astype(np.uint32)

def thread_count(numel, sms, max_threads_per_sm=2048):
    if numel <= 0 or sms <= 0 or max_threads_per_sm < 256:
        raise ValueError('Positive tensor size, SM count and >=256 threads/SM required')
    return min((numel + 255)//256, sms*(max_threads_per_sm//256))*256

def normal_at(indices, *, seed, sms, numel, offset=0, max_threads_per_sm=2048):
    if offset < 0 or offset % 4:
        raise ValueError('This backend supports nonnegative four-aligned Philox offsets')
    idx = np.asarray(indices, dtype=np.int64)
    if np.any(idx<0) or np.any(idx>=numel):
        raise ValueError('Indices outside original initialised tensor')
    threads=thread_count(numel,sms,max_threads_per_sm)
    sub=(idx%threads).astype(np.uint64)
    group=idx//(4*threads)
    lane=(idx//threads)%4
    words=philox(seed,sub,group.astype(np.uint64)+np.uint64(offset//4))
    # cuRAND Box-Muller returns sin then cos for each pair.
    u=(words[:,[0,2]].astype(np.float64)+1.0)/2**32
    v=(words[:,[1,3]].astype(np.float64)+1.0)*(2*np.pi/2**32)
    radius=np.sqrt(-2*np.log(u))
    z=np.empty(words.shape,dtype=np.float64)
    z[:,0::2]=radius*np.sin(v);z[:,1::2]=radius*np.cos(v)
    return z[np.arange(len(idx)),lane].astype(np.float32)

"""Bounded, revision-pinned reads of actual public Hugging Face tensors."""
import hashlib
import json
import struct
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import numpy as np


def json_url(url):
    with urllib.request.urlopen(url,timeout=60) as r:
        return json.load(r)


def read_range(url,start,length):
    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+length-1}'})
    with urllib.request.urlopen(req,timeout=120) as r:
        if r.status != 206:
            raise RuntimeError(f'Server ignored Range ({r.status}); refusing full shard download')
        cr=r.headers.get('Content-Range','')
        if not cr.startswith(f'bytes {start}-'):
            raise RuntimeError(f'Unexpected Content-Range: {cr}')
        data=r.read(length+1)
    if len(data)!=length: raise RuntimeError('Incorrect range response length')
    return data


def fetch_rows(repo,tensor,rows,out,revision='main'):
    info=json_url(f'https://huggingface.co/api/models/{repo}/revision/{revision}')
    revision=info['sha']
    base=f'https://huggingface.co/{repo}/resolve/{revision}/'
    try:
        index=json_url(base+'model.safetensors.index.json')
        shard=index['weight_map'][tensor]
    except urllib.error.HTTPError as e:
        if e.code!=404: raise
        shard='model.safetensors'
    url=base+urllib.parse.quote(shard)
    hlen=struct.unpack('<Q',read_range(url,0,8))[0]
    if hlen > 100_000_000: raise ValueError('Unreasonable safetensors header')
    header=json.loads(read_range(url,8,hlen))
    meta=header[tensor];shape=meta['shape'];dtype=meta['dtype']
    if len(shape)!=2: raise ValueError('Expected a rank-2 tensor')
    size={'BF16':2,'F16':2,'F32':4}[dtype]
    rows=sorted(set(rows));width=shape[1]; arrays=[];digests=[]
    for row in rows:
        if row<0 or row>=shape[0]: raise ValueError('Invalid row')
    # Coalesce adjacent rows without fetching gaps or entire tensors.
    runs=[]
    for row in rows:
        if not runs or row != runs[-1][-1]+1: runs.append([row])
        else: runs[-1].append(row)
    for run in runs:
        offset=8+hlen+meta['data_offsets'][0]+run[0]*width*size
        block=read_range(url,offset,len(run)*width*size)
        for j,row in enumerate(run):
            raw=block[j*width*size:(j+1)*width*size]
            digests.append(hashlib.sha256(raw).hexdigest())
            if dtype=='BF16': arr=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
            else: arr=np.frombuffer(raw,dtype={'F16':'<f2','F32':'<f4'}[dtype]).astype(np.float32)
            arrays.append(arr)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,values=np.stack(arrays),rows=np.array(rows),shape=np.array(shape))
    provenance={'repo':repo,'revision':revision,'tensor':tensor,'shard':shard,'shape':shape,'dtype':dtype,
        'rows':rows,'raw_row_sha256':digests,'url':url,'fixture_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'origin':'actual published Hugging Face checkpoint; no synthetic observations'}
    out.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance

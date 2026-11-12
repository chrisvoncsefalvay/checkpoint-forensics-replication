"""Select seed/layout/offset on discovery rows, score untouched holdout rows."""
import numpy as np
from .philox import normal_at


def corr(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den else None


def analyse(fixture,seeds,sms,offsets,discovery_rows=None,max_columns=512):
    d=np.load(fixture,allow_pickle=False);values=d['values'];rows=d['rows'];shape=d['shape']
    if len(rows)<4: raise ValueError('At least four actual checkpoint rows required')
    if not np.all(np.isfinite(values)): raise ValueError('Non-finite checkpoint values')
    if discovery_rows is None: discovery_rows=rows[::2].tolist()
    train=np.isin(rows,discovery_rows);hold=~train
    if not train.any() or not hold.any(): raise ValueError('Discovery and holdout must be nonempty')
    cols=np.unique(np.linspace(0,shape[1]-1,min(max_columns,shape[1]),dtype=int))
    indices=(rows[:,None]*shape[1]+cols).ravel()
    obs=values[:,cols]
    candidates=[]
    for seed in seeds:
        for sm in sms:
            for offset in offsets:
                pred=normal_at(indices,seed=int(seed),sms=int(sm),numel=int(np.prod(shape)),offset=int(offset)).reshape(obs.shape)
                candidates.append({'seed':int(seed),'sms':int(sm),'offset':int(offset),'discovery_r':corr(obs[train].ravel(),pred[train].ravel())})
    candidates.sort(key=lambda c:c['discovery_r'] if c['discovery_r'] is not None else -2,reverse=True)
    winner=candidates[0]
    # Freeze winner before touching holdout. Also report explicitly predeclared controls.
    controls=[]
    configurations=[(winner['seed'],winner['sms'],winner['offset'],'selected'),
                    (winner['seed']+1,winner['sms'],winner['offset'],'wrong_seed')]
    configurations += [(winner['seed'],int(sm),winner['offset'],'alternative_layout') for sm in sms if sm!=winner['sms']]
    for seed,sm,offset,label in configurations:
        pred=normal_at(indices,seed=seed,sms=sm,numel=int(np.prod(shape)),offset=offset).reshape(obs.shape)
        row_rs=[corr(a,b) for a,b in zip(obs[hold],pred[hold])]
        controls.append({'label':label,'seed':seed,'sms':sm,'offset':offset,'heldout_r':corr(obs[hold].ravel(),pred[hold].ravel()),'row_correlations':dict(zip(map(str,rows[hold].tolist()),row_rs))})
    return {'fixture':str(fixture),'original_tensor_shape':shape.tolist(),'discovery_rows':rows[train].tolist(),
        'heldout_rows':rows[hold].tolist(),'columns_per_row':len(cols),'candidate_count':len(candidates),
        'selected':winner,'discovery_top_10':candidates[:10],'heldout_controls':controls,
        'interpretation':'NumPy normal transform; GPU scripts provide the bitwise comparison.',
        'statistical_note':'Row-level correlations are reported because checkpoint coordinates may be dependent.'}

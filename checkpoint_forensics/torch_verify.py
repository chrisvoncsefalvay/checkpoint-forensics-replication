"""Unmodified PyTorch verification against actual checkpoint rows on host GPU.

Allocates the original initialisation tensor; refuses allocations above budget.
The original tensor shape preserves the initialisation launch layout.
"""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('fixture');p.add_argument('--seed',type=int,required=True);p.add_argument('--offset',type=int,default=0);p.add_argument('--std',type=float,default=.02);p.add_argument('--max-gib',type=float,default=4);p.add_argument('--out',required=True);a=p.parse_args()
    import torch
    from .analysis import corr
    d=np.load(a.fixture);obs=d['values'];rows=d['rows'];shape=d['shape'].tolist()
    if np.prod(shape)*2> a.max_gib*2**30:raise SystemExit('Tensor exceeds --max-gib; no allocation performed')
    g=torch.Generator(device='cuda');g.manual_seed(a.seed);g.set_offset(a.offset)
    pred=torch.empty(shape,device='cuda',dtype=torch.bfloat16).normal_(mean=0,std=a.std,generator=g)
    got=pred[torch.tensor(rows,device='cuda')].float().cpu().numpy()
    props=torch.cuda.get_device_properties(0)
    report={'fixture':a.fixture,'seed':a.seed,'offset':a.offset,'std':a.std,'torch':torch.__version__,
      'cuda':torch.version.cuda,'device':props.name,'sms':props.multi_processor_count,
      'method':'unmodified torch.empty(full_shape, dtype=bfloat16).normal_',
      'rows':[{'row':int(row),'r':corr(o,z),'equal_values':int(np.count_nonzero(o==z)),'total':len(o)} for row,o,z in zip(rows,obs,got)]}
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

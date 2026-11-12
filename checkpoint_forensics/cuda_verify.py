"""Compare real checkpoint rows against cuRAND on an NVIDIA GPU.

The sparse kernel tests candidate layouts on the available GPU. Use torch_verify
for an unmodified-PyTorch comparison on the physical target hardware.
Requires torch, a CUDA toolkit/compiler and an NVIDIA device. No training occurs.
"""
import argparse,json
from pathlib import Path
import numpy as np

CUDA = r'''
#include <torch/extension.h>
#include <curand_kernel.h>
__global__ void candidate(const int64_t* idx,float* out,int64_t n,uint64_t seed,uint64_t offset,int64_t stride) {
  int64_t q=blockIdx.x*blockDim.x+threadIdx.x;
  if(q>=n)return;
  int64_t i=idx[q],sub=i%stride,group=i/(4*stride),lane=(i/stride)%4;
  curandStatePhilox4_32_10_t state;
  curand_init(seed,sub,offset+4*group,&state);
  float4 z=curand_normal4(&state); out[q]=((&z.x)[lane]);
}
torch::Tensor generate(torch::Tensor idx,int64_t seed,int64_t offset,int64_t stride){
 auto out=torch::empty(idx.sizes(),idx.options().dtype(torch::kFloat32));
 candidate<<<(idx.numel()+255)/256,256>>>(idx.data_ptr<int64_t>(),out.data_ptr<float>(),idx.numel(),seed,offset,stride);
 return out;
}
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('fixture');p.add_argument('--seed',type=int,required=True);p.add_argument('--sms',type=int,required=True);p.add_argument('--offset',type=int,default=0);p.add_argument('--std',type=float,default=.02);p.add_argument('--out',required=True);a=p.parse_args()
    import torch
    from torch.utils.cpp_extension import load_inline
    from .philox import thread_count
    from .analysis import corr
    d=np.load(a.fixture);obs=d['values'];rows=d['rows'];shape=d['shape']
    indices=(rows[:,None]*shape[1]+np.arange(shape[1])).ravel()
    module=load_inline(name='checkpoint_philox_sparse',cpp_sources='torch::Tensor generate(torch::Tensor,int64_t,int64_t,int64_t);',cuda_sources=CUDA,functions=['generate'],verbose=False)
    ids=torch.tensor(indices,device='cuda',dtype=torch.int64)
    raw=module.generate(ids,a.seed,a.offset,thread_count(int(np.prod(shape)),a.sms));torch.cuda.synchronize()
    pred=(raw*a.std).to(torch.bfloat16).float().cpu().numpy().reshape(obs.shape)
    report={'fixture':a.fixture,'seed':a.seed,'sms':a.sms,'offset':a.offset,'std':a.std,
       'torch':torch.__version__,'cuda':torch.version.cuda,'host_device':torch.cuda.get_device_name(),
       'method':'custom sparse cuRAND kernel; not unmodified PyTorch hardware verification',
       'rows':[{'row':int(row),'r':corr(o,z),'equal_values':int(np.count_nonzero(o==z)),'total':len(o)} for row,o,z in zip(rows,obs,pred)]}
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

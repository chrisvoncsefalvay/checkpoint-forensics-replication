import argparse,json
from pathlib import Path
from .fetch import fetch_rows
from .analysis import analyse

def ints(s):
    if ':' in s:
        parts=list(map(int,s.split(':')));return list(range(*parts))
    return list(map(int,s.split(',')))

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fetch');f.add_argument('--repo',required=True);f.add_argument('--tensor',required=True);f.add_argument('--rows',required=True);f.add_argument('--revision',default='main');f.add_argument('--out',required=True)
    a=sub.add_parser('analyse');a.add_argument('fixture');a.add_argument('--seeds',required=True);a.add_argument('--sms',default='108,132,148,152');a.add_argument('--offsets',default='0');a.add_argument('--columns',type=int,default=512);a.add_argument('--out',required=True)
    scan = sub.add_parser('scan', help='Stream checkpoint tensors with resumable limits')
    scan.add_argument('--repo', required=True)
    scan.add_argument('--revision', default='main')
    scan.add_argument('--tensor', default='.*', help='Tensor-name regular expression')
    scan.add_argument('--seeds', required=True)
    scan.add_argument('--sms', default='108,132,148,152')
    scan.add_argument('--offsets', default='0')
    scan.add_argument('--init-dtype', choices=['BF16', 'F16', 'F32'])
    scan.add_argument('--mapping', choices=['pytorch-contiguous', 'single-launch'], default='pytorch-contiguous')
    scan.add_argument('--max-threads-per-sm', type=int, default=2048)
    scan.add_argument('--tensor-config', help='JSON object of exact tensor names and hypothesis overrides')
    scan.add_argument('--state', required=True)
    scan.add_argument('--start-row', type=int, default=0)
    scan.add_argument('--batch-rows', type=int, default=32)
    scan.add_argument('--batch-bytes', type=int, default=8388608, help='Maximum weight bytes in memory per batch')
    scan.add_argument('--columns', type=int, default=512, help='Columns scored per row; 0 scores every value')
    scan.add_argument('--threshold', type=float, default=.99)
    scan.add_argument('--keep-hits', type=int, default=100)
    scan.add_argument('--max-weight-bytes', type=int, default=268435456, help='Cumulative checkpoint payload limit; 0 is unlimited')
    scan.add_argument('--max-rows', type=int, default=0, help='Cumulative row limit; 0 is unlimited')
    scan.add_argument('--max-tensors', type=int, default=0, help='Cumulative tensor limit; 0 is unlimited')
    scan.add_argument('--max-seconds', type=float, default=600, help='Scanning time per invocation, checked between requests/batches; 0 is unlimited')
    scan.add_argument('--quiet', action='store_true')
    x=p.parse_args()
    if x.command == 'scan':
        if any(getattr(x, name) < 0 for name in ('start_row', 'columns', 'max_weight_bytes', 'max_rows', 'max_tensors', 'max_seconds')) or x.batch_rows < 1 or x.keep_hits < 1 or x.batch_bytes < 1:
            p.error('Limits must be nonnegative; batch rows and retained hits must be positive')
        if not -1 <= x.threshold <= 1:
            p.error('Threshold must be between -1 and 1')
        from .scan import scan as run_scan
        result = run_scan(x)
    elif x.command=='fetch': result=fetch_rows(x.repo,x.tensor,ints(x.rows),x.out,x.revision)
    else:
        result=analyse(x.fixture,ints(x.seeds),ints(x.sms),ints(x.offsets),max_columns=x.columns)
        Path(x.out).parent.mkdir(parents=True,exist_ok=True);Path(x.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

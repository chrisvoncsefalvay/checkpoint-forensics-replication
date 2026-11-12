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
    x=p.parse_args()
    if x.command=='fetch': result=fetch_rows(x.repo,x.tensor,ints(x.rows),x.out,x.revision)
    else:
        result=analyse(x.fixture,ints(x.seeds),ints(x.sms),ints(x.offsets),max_columns=x.columns)
        Path(x.out).parent.mkdir(parents=True,exist_ok=True);Path(x.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

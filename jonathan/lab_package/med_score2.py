import json,sys,importlib.util,types,re
from collections import defaultdict
ROOT="/mnt/hdd/jonathan/data/MedAgentBench"
BASE="http://localhost:8080/fhir/"
sys.path.insert(0,ROOT)
pkg=types.ModuleType("mab"); pkg.__path__=[f"{ROOT}/src/server/tasks/medagentbench"]
sys.modules["mab"]=pkg
for name in ("utils","refsol"):
    sp=importlib.util.spec_from_file_location(f"mab.{name}",
        f"{ROOT}/src/server/tasks/medagentbench/{name}.py")
    mod=importlib.util.module_from_spec(sp); sys.modules[f"mab.{name}"]=mod
    setattr(pkg,name,mod); sp.loader.exec_module(mod)
refsol=sys.modules["mab.refsol"]

class Msg:
    def __init__(s,role,content): s.role, s.content = role, content
class Res:
    def __init__(s,hist,result): s.history, s.result = hist, result

def flatten(x):
    """dict -> its 'value', else first scalar leaf. Keeps numbers and ISO strings."""
    if isinstance(x,dict):
        if 'value' in x: return flatten(x['value'])
        for v in x.values():
            if isinstance(v,(int,float,str)): return flatten(v)
        return None
    if isinstance(x,str):
        m=re.match(r'^-?\d+(\.\d+)?$',x.strip())
        return float(x) if m else x.strip()
    return x

_real=json.loads
def lenient_loads(s,*a,**k):
    v=_real(s,*a,**k)
    return [flatten(i) for i in v] if isinstance(v,list) else v

data=json.load(open(f"{ROOT}/data/medagentbench/test_data_v2.json"))
CASE={c['id']:c for c in data}

def score(f, lenient):
    out=defaultdict(lambda:[0,0])
    for l in open(f):
        try: r=_real(l)
        except Exception: continue
        c=CASE.get(r['id'])
        if not c or 'msgs' not in r: continue
        hist=[Msg('agent' if m['role']=='assistant' else 'user', m['content']) for m in r['msgs']]
        res=Res(hist, r.get('answer') or '[]')
        t=r['task']; out[t][1]+=1
        if lenient:
            refsol.json.loads = lenient_loads
            # gold must be flattened the same way
            c=dict(c); c['sol']=[flatten(x) for x in c.get('sol',[])] if c.get('sol') else c.get('sol')
        else:
            refsol.json.loads = _real
        try:
            if refsol.__dict__[t](c,res,BASE) is True: out[t][0]+=1
        except Exception: pass
    refsol.json.loads = _real
    return out

fw,fs='medq2_weak.jsonl','medq2_strong.jsonl'
if len(sys.argv)>2: fw,fs=sys.argv[1],sys.argv[2]
import io,contextlib
res={}
for mode in (False,True):
    with contextlib.redirect_stdout(io.StringIO()):
        res[mode]=(score(fw,mode),score(fs,mode))

def block(tag,W,S):
    tw=ts=to=n=0; rows=[]
    for t in sorted(set(W)|set(S), key=lambda x:int(x[4:])):
        a,na=W.get(t,[0,0]); b,nb=S.get(t,[0,0])
        pa=a/na if na else 0; pb=b/nb if nb else 0; o=max(pa,pb)
        tw+=pa; ts+=pb; to+=o; n+=1
        rows.append((t,a,na,pa,b,nb,pb,o))
    return rows,100*tw/n,100*ts/n,100*to/n

print(f"{'task':8s} | {'7B strict':>10s} {'32B strict':>11s} | {'7B lenient':>11s} {'32B lenient':>12s}")
print("-"*64)
rs,ws,ss,os_=block('strict',*res[False])
rl,wl,sl,ol=block('lenient',*res[True])
for (t,a,na,pa,b,nb,pb,_),(t2,a2,_,pa2,b2,_,pb2,_) in zip(rs,rl):
    print(f"{t:8s} | {a:3d}/{na:<3d} {100*pa:5.1f}% {b:3d}/{nb:<3d} {100*pb:5.1f}% | {a2:3d} {100*pa2:6.1f}% {b2:3d} {100*pb2:7.1f}%")
print("-"*64)
print(f"{'MEAN':8s} | {'':7s} {ws:5.1f}% {'':7s} {ss:5.1f}% | {'':4s} {wl:6.1f}% {'':4s} {sl:7.1f}%")
print()
for tag,w,s_,o in (("STRICT ",ws,ss,os_),("LENIENT",wl,sl,ol)):
    bf=max(w,s_)
    print(f"{tag}  best-fixed {bf:5.1f}%   oracle {o:5.1f}%   headroom +{o-bf:.1f}")

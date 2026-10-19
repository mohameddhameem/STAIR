import json,sys,importlib.util,types
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

data=json.load(open(f"{ROOT}/data/medagentbench/test_data_v2.json"))
CASE={c['id']:c for c in data}

def score(f):
    out=defaultdict(lambda:[0,0])
    for l in open(f):
        try: r=json.loads(l)
        except Exception: continue
        c=CASE.get(r['id'])
        if not c or 'msgs' not in r: continue
        hist=[Msg('agent' if m['role']=='assistant' else 'user', m['content']) for m in r['msgs']]
        res=Res(hist, r.get('answer') or '[]')
        t=r['task']; out[t][1]+=1
        try:
            if refsol.__dict__[t](c,res,BASE) is True: out[t][0]+=1
        except Exception: pass
    return out

fw=sys.argv[1] if len(sys.argv)>1 else 'medq2_weak.jsonl'
fs=sys.argv[2] if len(sys.argv)>2 else 'medq2_strong.jsonl'
W,S=score(fw),score(fs)
print(f"{'task':8s} {'7B':>13s} {'32B':>13s}   oracle")
print("-"*50)
tw=ts=to=n=0
for t in sorted(set(W)|set(S), key=lambda x:int(x[4:])):
    a,na=W.get(t,[0,0]); b,nb=S.get(t,[0,0])
    pa=a/na if na else 0; pb=b/nb if nb else 0; o=max(pa,pb)
    tw+=pa; ts+=pb; to+=o; n+=1
    print(f"{t:8s} {a:3d}/{na:<3d} {100*pa:5.1f}% {b:3d}/{nb:<3d} {100*pb:5.1f}%   {100*o:5.1f}%")
print("-"*50)
print(f"{'MEAN':8s} {'':7s} {100*tw/n:5.1f}% {'':7s} {100*ts/n:5.1f}%   {100*to/n:5.1f}%")
bf=max(100*tw/n,100*ts/n)
print(f"\nbest-fixed {bf:.1f}%   per-task oracle {100*to/n:.1f}%   headroom +{100*to/n-bf:.1f}")

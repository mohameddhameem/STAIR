#!/usr/bin/env python3
# make_sbranch_data.py v1.1 — data prep for the real-label S-branch QA run (18 Sep 2026)
# Goal: train the QA choice on MEASURED outcome labels (real qa_label_S), strong branch only.
# Method: copy router_data_lenient.jsonl (the REAL-label file) row-by-row and
#   - keep qa_label_S exactly as measured (the training signal under test)
#   - blank qa_label_W  -> None   (so make_examples yields no weak-branch QA lessons)
#   - KEEP ret_label unchanged: with --phases qa no ret TRAINING happens, but the
#     every-25-steps probe scores both stages — an empty ret set could crash it.
#     The ret probe will just report the untrained head (harmless).
# Everything else (question, evidence_S, em_*, splits ids) passes through unchanged.
# Output: router_data_sbranch_real.jsonl + printed label counts per split.
import json, collections
SRC='/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl'
OUT='/mnt/hdd/jonathan/runs/full_run_p100/router_data_sbranch_real.jsonl'
SPL='/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json'
sp=json.load(open(SPL)); sp=sp.get('splits',sp)
where={i:s for s in ('train','val','test') for i in sp[s]}
cnt=collections.Counter(); kept=collections.Counter()
with open(OUT,'w') as f:
    for l in open(SRC):
        r=json.loads(l)
        s=where.get(r['idx'],'?')
        lab=r.get('qa_label_S')
        cnt[(s,str(lab))]+=1
        r['qa_label_W']=None
        f.write(json.dumps(r)+'\n')
        if lab in ('W','S'): kept[s]+=1
print('wrote', OUT)
for s in ('train','val','test'):
    W=cnt[(s,'W')]; S=cnt[(s,'S')]; N=cnt[(s,'None')]
    tot=W+S
    print(f"{s:5s}: qa_label_S  W={W} S={S} unresolved={N}  -> trainable={tot}  W-share(floor)={W/tot:.4f}" if tot else f"{s}: none")

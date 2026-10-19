#!/usr/bin/env python3
# gold_coverage_aggregate.py v1.3 (adds I: body-level features; E gains strict-phrase) — Prof. Archan's analysis, aggregated (18 Sep 2026)
# CPU-only, read-only. Compares weak/strong retrieval selections against ground truth
# (title-annotated gold passages + answer-in-body) and prints the decision numbers:
#   A. coverage rates          B. pattern table (coverage x real label)
#   C. label-candidate AUCs    D. branch oracles (the skip-ret verdict)
#   E. answer-in-body rates    F. "strong matters" cell
import json, re
from collections import Counter

DATA='/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl'
SRC='/mnt/hdd/jonathan/data/tts_test.json'

def title_index(pool):
    return {m.group(2).strip(): int(m.group(1)) for m in re.finditer(r"^\[(\d+)\]\s*([^:\n]{2,80}):", pool, flags=re.M)}
def passages_full(pool):
    out={}; ms=list(re.finditer(r"^\[(\d+)\]\s*", pool, flags=re.M))
    for i,m in enumerate(ms):
        k=int(m.group(1)); end=ms[i+1].start() if i+1<len(ms) else len(pool)
        out[k]=pool[m.start():end]
    return out
def sel(evidence, titles, n_pool):
    blob=str(evidence); s=set()
    part=blob.split("Selected references",1)[-1]
    for m in re.findall(r"\[(\d+)\]", part):
        k=int(m)
        if 1<=k<=(n_pool or 10**9): s.add(k)
    for t,k in titles.items():
        if t in blob: s.add(k)
    return s
def auc(pairs):
    """pairs = [(score, is_positive)]; rank-based AUC, ties get average rank."""
    pairs=sorted(pairs, key=lambda x: x[0])
    n=len(pairs); pos=sum(1 for _,y in pairs if y); neg=n-pos
    if not pos or not neg: return None
    # average ranks with ties
    ranks={}; i=0
    while i<n:
        j=i
        while j+1<n and pairs[j+1][0]==pairs[i][0]: j+=1
        r=(i+j)/2+1
        for k in range(i,j+1): ranks[k]=r
        i=j+1
    rsum=sum(ranks[k] for k in range(n) if pairs[k][1])
    return (rsum - pos*(pos+1)/2)/(pos*neg)

src={r['_id']: r for r in json.load(open(SRC))}
R=[]
for l in open(DATA):
    r=json.loads(l); g=src.get(r.get('_id'))
    pool=r.get('pool_text',''); ti=title_index(pool); pf=passages_full(pool)
    w=sel(r.get('evidence_W',''), ti, r.get('n_pool')); s=sel(r.get('evidence_S',''), ti, r.get('n_pool'))
    un=w|s; j=len(w&s)/len(un) if un else None
    gts=g['supporting_titles'] if g else []
    gidx=set(ti[t] for t in gts if t in ti)
    gold_in_pool=bool(gidx) and len(gidx)==len(gts)
    wcov=gidx<=w if gidx else None; scov=gidx<=s if gidx else None
    wfrac=len(gidx&w)/len(gidx) if gidx else None
    a0=str(g['answers'][0]) if g and g['answers'] else ''
    ans_ok = a0.lower() not in ('yes','no') and len(a0)>=3
    def has_ans(ks):
        if not ans_ok: return None
        words=[x for x in re.findall(r'\w+', a0.lower())]
        for k in ks:
            low=pf.get(k,'').lower()
            if words and all(re.search(r'(?<!\w)'+re.escape(x)+r'(?!\w)', low) for x in words): return True
        return False
    # ---- body-level features (section I) ----
    STOP=set('the a an of in on at to for and or is are was were be been do does did what which who whom whose when where why how same both'.split())
    qwords=set(x for x in re.findall(r'[a-z0-9]+', r.get('question','').lower()) if x not in STOP and len(x)>2)
    wtext=' '.join(pf.get(k,'') for k in w).lower()
    stext=' '.join(pf.get(k,'') for k in s).lower()
    qcov=(sum(1 for x in qwords if x in wtext)/len(qwords)) if qwords else None
    wtok=set(re.findall(r'[a-z0-9]+', wtext)); stok=set(re.findall(r'[a-z0-9]+', stext))
    bj=(len(wtok&stok)/len(wtok|stok)) if (wtok|stok) else None
    gtw=set(x for t in gts for x in re.findall(r'[a-z0-9]+', t.lower()) if len(x)>2)
    nearmiss=0
    for k in w:
        if k in gidx: continue
        tw=set(re.findall(r'[a-z0-9]+', (pf.get(k,'').split(':',1)[0]).lower()))
        if tw & gtw: nearmiss+=1
    wlen=len(wtext)
    # strict phrase presence (section E addition)
    def has_ans_strict(ks):
        if not (a0.lower() not in ('yes','no') and len(a0)>=3): return None
        pat=re.compile(r'(?<!\w)'+re.escape(a0.lower())+r'(?!\w)')
        return any(pat.search(pf.get(k,'').lower()) for k in ks)
    aform = ('yes/no' if a0.lower() in ('yes','no') else ('short' if len(a0)<3 else 'entity'))
    R.append(dict(split=r['split'], j=j, gidx=gidx, gin=gold_in_pool, src_gin=(g.get('gold_in_pool') if g else None),
                  wcov=wcov, scov=scov, wfrac=wfrac, real=r.get('ret_label'),
                  wans=has_ans(w), sans=has_ans(s), em=r['em_lenient'],
                  qtype=(g.get('type') if g else None), level=(g.get('level') if g else None), aform=aform,
                  nw=len(w), ns=len(s), gmin=(min(gidx) if gidx else None), gmax=(max(gidx) if gidx else None),
                  qcov=qcov, bj=bj, nearmiss=nearmiss, wlen=wlen, wans_strict=has_ans_strict(w), sans_strict=has_ans_strict(s),
                  qlen=len(r.get('question','')), qaW=r.get('qa_label_W'), qaS=r.get('qa_label_S')))

print(f"n={len(R)}")
# cross-check our title matching vs the source's own gold_in_pool flag
agree=sum(1 for x in R if x['src_gin'] is not None and x['gin']==x['src_gin'])
tot=sum(1 for x in R if x['src_gin'] is not None)
print(f"[sanity] our gold-in-pool matches source flag: {agree}/{tot}")

print("\n0. QUESTION-TYPE PROFILE")
for t in sorted(set(x['qtype'] for x in R if x['qtype'])):
    T=[x for x in R if x['qtype']==t]
    res_t=[x for x in T if x['real'] in ('W','S')]
    yn=sum(1 for x in T if x['aform']=='yes/no')
    print(f"   {t:11s} n={len(T):4d} ({100*len(T)/len(R):.1f}%) | answer form: yes/no {100*yn/len(T):.0f}%, searchable {100*(len(T)-yn)/len(T):.0f}%"
          f" | real W-rate {100*sum(x['real']=='W' for x in res_t)/max(1,len(res_t)):.1f}% (resolved n={len(res_t)})"
          f" | em: WW {100*sum(x['em'][0] for x in T)/len(T):.1f} SW {100*sum(x['em'][2] for x in T)/len(T):.1f} SS {100*sum(x['em'][3] for x in T)/len(T):.1f}")
    a_t=auc([(x['j'], x['real']=='W') for x in res_t if x['j'] is not None])
    print(f"               overlap AUC within type: {a_t if a_t is None else round(a_t,3)}")

G=[x for x in R if x['gin']]
print(f"\nA. COVERAGE (instances with gold in pool: {len(G)}/{len(R)})")
print(f"   weak covers ALL gold:   {100*sum(x['wcov'] for x in G)/len(G):.1f}%")
print(f"   strong covers ALL gold: {100*sum(x['scov'] for x in G)/len(G):.1f}%")
print(f"   both cover: {100*sum(x['wcov'] and x['scov'] for x in G)/len(G):.1f}%   "
      f"only strong: {100*sum((not x['wcov']) and x['scov'] for x in G)/len(G):.1f}%   "
      f"only weak: {100*sum(x['wcov'] and (not x['scov']) for x in G)/len(G):.1f}%   "
      f"neither: {100*sum((not x['wcov']) and (not x['scov']) for x in G)/len(G):.1f}%")
for t in sorted(set(x['qtype'] for x in G if x['qtype'])):
    Gt=[x for x in G if x['qtype']==t]
    print(f"   [{t}] weak covers {100*sum(x['wcov'] for x in Gt)/len(Gt):.1f}%  strong covers {100*sum(x['scov'] for x in Gt)/len(Gt):.1f}%  only-strong {100*sum((not x['wcov']) and x['scov'] for x in Gt)/len(Gt):.1f}%  (n={len(Gt)})")

res=[x for x in G if x['real'] in ('W','S')]
print(f"\nB. PATTERN TABLE (resolved rows n={len(res)}): real W-rate by coverage cell")
for name,f in [('weak covers',lambda x:x['wcov']), ('weak misses',lambda x:not x['wcov']),
               ('only strong covers',lambda x:(not x['wcov']) and x['scov']),
               ('neither covers',lambda x:(not x['wcov']) and (not x['scov']))]:
    cell=[x for x in res if f(x)]
    if cell: print(f"   {name:20s} n={len(cell):4d}  W-rate {100*sum(x['real']=='W' for x in cell)/len(cell):.1f}%")

print(f"\nC. LABEL-CANDIDATE AUCs (resolved rows, per split; positive class = W)")
for split in ('train','val','test'):
    rr=[x for x in res if x['split']==split]
    a_j=auc([(x['j'], x['real']=='W') for x in rr if x['j'] is not None])
    a_g=auc([(x['wfrac'], x['real']=='W') for x in rr])
    a_c=auc([(x['j']+ (0.25 if x['wcov'] else 0), x['real']=='W') for x in rr if x['j'] is not None])
    print(f"   {split:5s}: overlap {a_j:.3f} | weak-gold-frac {a_g:.3f} | overlap+covers {a_c:.3f}")

print(f"\nD. BRANCH ORACLES (lenient, per split) — the skip-ret verdict")
for split in ('train','val','test'):
    rr=[x for x in R if x['split']==split]
    wb=sum(max(x['em'][0],x['em'][1]) for x in rr)/len(rr)   # best using weak retrieval only
    sb=sum(max(x['em'][2],x['em'][3]) for x in rr)/len(rr)   # best using strong retrieval only
    fo=sum(max(x['em']) for x in rr)/len(rr)
    ss=sum(x['em'][3] for x in rr)/len(rr)
    print(f"   {split:5s}: weak-branch oracle {100*wb:.2f} | strong-branch {100*sb:.2f} | full oracle {100*fo:.2f} | always-SS {100*ss:.2f}")

A=[x for x in G if x['wans'] is not None]
print(f"\nE. ANSWER-IN-BODY (non-yes/no rows with gold in pool: n={len(A)})")
print(f"   weak selection holds an answer-bearing passage:   {100*sum(x['wans'] for x in A)/len(A):.1f}%")
print(f"   strong selection holds an answer-bearing passage: {100*sum(x['sans'] for x in A)/len(A):.1f}%")
As=[x for x in G if x['wans_strict'] is not None]
print(f"   strict-phrase variant: weak {100*sum(x['wans_strict'] for x in As)/len(As):.1f}%  strong {100*sum(x['sans_strict'] for x in As)/len(As):.1f}%  (n={len(As)})")

O=[x for x in res if (not x['wcov']) and x['scov']]
print(f"\nF. WHERE STRONG MATTERS (only-strong-covers, resolved: n={len(O)})")
if O:
    print(f"   real W-rate there: {100*sum(x['real']=='W' for x in O)/len(O):.1f}%  "
          f"(low => these rows genuinely need strong retrieval)")
    print(f"   weak-branch success there: {100*sum(max(x['em'][0],x['em'][1]) for x in O)/len(O):.1f}%  "
          f"vs strong-branch: {100*sum(max(x['em'][2],x['em'][3]) for x in O)/len(O):.1f}%")


def prof(name, cell):
    if not cell: print(f"   {name}: n=0"); return
    n=len(cell)
    br=100*sum(1 for x in cell if x['qtype']=='bridge')/n
    j=[x['j'] for x in cell if x['j'] is not None]
    gm=[x['gmin'] for x in cell if x['gmin'] is not None]
    gx=[x['gmax'] for x in cell if x['gmax'] is not None]
    med=lambda v: sorted(v)[len(v)//2] if v else None
    print(f"   {name:24s} n={n:4d} | bridge {br:.0f}% | median jaccard {med(j):.2f} | "
          f"median n_weak {med([x['nw'] for x in cell])} n_strong {med([x['ns'] for x in cell])} | "
          f"median gold pos: first {med(gm)} last {med(gx)} | median q len {med([x['qlen'] for x in cell])}")

print("\nG. NEEDS-STRONG PROFILE (gold-in-pool rows): what characterizes each coverage cell?")
cells=[('only strong covers', [x for x in G if (not x['wcov']) and x['scov']]),
       ('both cover',         [x for x in G if x['wcov'] and x['scov']]),
       ('only weak covers',   [x for x in G if x['wcov'] and (not x['scov'])]),
       ('neither covers',     [x for x in G if (not x['wcov']) and (not x['scov'])])]
for nm,c in cells: prof(nm,c)
os_=[x for x in G if (not x['wcov']) and x['scov']]
rest=[x for x in G if not ((not x['wcov']) and x['scov'])]
print("   gold-position signal: only-strong rows with first gold index > 10:",
      f"{100*sum(1 for x in os_ if x['gmin'] and x['gmin']>10)/len(os_):.0f}%  vs other rows: "
      f"{100*sum(1 for x in rest if x['gmin'] and x['gmin']>10)/len(rest):.0f}%")
a_pos=auc([( -(x['gmin'] or 0), x['real']=='W') for x in G if x['real'] in ('W','S') and x['gmin'] is not None])
print(f"   AUC of (early gold position) predicting real W: {a_pos:.3f}   [deployment caveat: gold position unknown online]")

print("\nH. QA-SIDE PATTERNS — what predicts 'weak answerer suffices'? (real qa labels)")
S_=[x for x in R if x['qaS'] in ('W','S')]
print(f"   ON STRONG EVIDENCE (the pivot architecture's condition; n={len(S_)}): weak-QA-suffices rate {100*sum(x['qaS']=='W' for x in S_)/len(S_):.1f}%")
for t in ('bridge','comparison'):
    T=[x for x in S_ if x['qtype']==t]
    if T: print(f"      by type: {t:11s} {100*sum(x['qaS']=='W' for x in T)/len(T):.1f}% (n={len(T)})")
for nm,f in [('strong covers gold',lambda x:x['gin'] and x['scov']),('strong misses gold',lambda x:x['gin'] and not x['scov'])]:
    T=[x for x in S_ if f(x)]
    if T: print(f"      by coverage: {nm:20s} {100*sum(x['qaS']=='W' for x in T)/len(T):.1f}% (n={len(T)})")
for nm,f in [('answer text in strong sel',lambda x:x['sans'] is True),('answer text NOT in strong sel',lambda x:x['sans'] is False)]:
    T=[x for x in S_ if f(x)]
    if T: print(f"      by answer-in-body: {nm:28s} {100*sum(x['qaS']=='W' for x in T)/len(T):.1f}% (n={len(T)})")
W_=[x for x in R if x['qaW'] in ('W','S')]
print(f"   ON WEAK EVIDENCE (n={len(W_)}): weak-QA-suffices rate {100*sum(x['qaW']=='W' for x in W_)/len(W_):.1f}%")
for nm,f in [('weak covers gold',lambda x:x['gin'] and x['wcov']),('weak misses gold',lambda x:x['gin'] and not x['wcov'])]:
    T=[x for x in W_ if f(x)]
    if T: print(f"      by coverage: {nm:20s} {100*sum(x['qaW']=='W' for x in T)/len(T):.1f}% (n={len(T)})")


print("\nI. BODY-LEVEL FEATURES — can passage CONTENT (not identity) guide the retrieval choice?")
res_all=[x for x in R if x['real'] in ('W','S')]
def show(name, key, flip=False, deploy=''):
    pairs=[(( -x[key] if flip else x[key]), x['real']=='W') for x in res_all if x[key] is not None]
    a=auc(pairs)
    print(f"   {name:44s} AUC {a:.3f}   {deploy}")
show('question-term coverage in weak bodies','qcov',False,'[deployable: computed from weak\'s own output]')
show('BODY-level jaccard (weak vs strong text)','bj',False,'[offline, like index-overlap 0.744]')
show('near-miss distractors in weak selection','nearmiss',True,'[deployable if gold titles ~ question entities]')
show('weak selection text volume (chars)','wlen',False,'[deployable]')
# combo: does body-jaccard beat / add to index-overlap?
both=[x for x in res_all if x['bj'] is not None and x['j'] is not None]
a_idx=auc([(x['j'], x['real']=='W') for x in both])
a_bj=auc([(x['bj'], x['real']=='W') for x in both])
a_mix=auc([((x['j']+x['bj'])/2, x['real']=='W') for x in both])
print(f"   head-to-head on shared rows: index-overlap {a_idx:.3f} | body-jaccard {a_bj:.3f} | mean-fusion {a_mix:.3f}")
q1=[x for x in res_all if x['qcov'] is not None and x['qcov']>=0.8]
q0=[x for x in res_all if x['qcov'] is not None and x['qcov']<0.4]
print(f"   question-term coverage extremes: >=0.8 -> W-rate {100*sum(x['real']=='W' for x in q1)/max(1,len(q1)):.1f}% (n={len(q1)}) | <0.4 -> {100*sum(x['real']=='W' for x in q0)/max(1,len(q0)):.1f}% (n={len(q0)})")

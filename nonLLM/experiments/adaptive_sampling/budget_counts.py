"""Describe budget differences in a saved training-only score snapshot."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('snapshot');a=p.parse_args();path=Path(a.snapshot);z=np.load(path);ids=z['ids'];pairs=['WW','WS','SW','SS'];f=np.stack([z[p][ids,:,1] for p in pairs]);em=np.stack([z[p][ids,:,0] for p in pairs]);assert np.isfinite(f).all()
margin=.05
out={'snapshot':str(path),'questions':len(ids),'scope':'same completed training questions only','margin_f1_points':5,'definitions':{'k1_better_k40':'F1(k=1)-F1(k=40)>=0.05 within the same model pair','small_strictly_better_all_larger':'max F1(k=1..5)-max F1(k=6..40)>=0.05 within the same model pair','joint_small_strictly_better_all_larger':'max over all model pairs and k=1..5 minus max over all pairs and k=6..40 >=0.05'},'pairs':{}}
small_masks=[];first_masks=[]
for g,pair in enumerate(pairs):
 s=f[g];small=s[:,:5].max(axis=1)-s[:,5:].max(axis=1)>=margin;first=s[:,0]-s[:,39]>=margin
 small_masks.append(small);first_masks.append(first)
 maximum=s.max(axis=1);earliest=s.argmax(axis=1)+1
 out['pairs'][pair]={'k1_better_k40':int(first.sum()),'k40_better_k1':int((s[:,39]-s[:,0]>=margin).sum()),'small_strictly_better_all_larger':int(small.sum()),'larger_strictly_better_all_small':int((s[:,5:].max(axis=1)-s[:,:5].max(axis=1)>=margin).sum()),'all_k_f1_zero':int((maximum==0).sum()),'earliest_best_f1_k1to5_nonzero':int(((earliest<=5)&(maximum>0)).sum()),'em_k1_correct_k40_wrong':int(((em[g,:,0]==1)&(em[g,:,39]==0)).sum()),'em_k40_correct_k1_wrong':int(((em[g,:,39]==1)&(em[g,:,0]==0)).sum()),'ever_em_correct':int(em[g].max(axis=1).sum())}
out['unique_questions_k1_better_k40_in_atleast_one_pair']=int(np.stack(first_masks).any(axis=0).sum())
out['unique_questions_small_strictly_better_all_larger_in_atleast_one_pair']=int(np.stack(small_masks).any(axis=0).sum())
low=f[:,:,:5].max(axis=(0,2));high=f[:,:,5:].max(axis=(0,2));out['joint_small_strictly_better_all_larger']=int((low-high>=margin).sum());out['joint_larger_strictly_better_all_small']=int((high-low>=margin).sum())
q=path.with_name(path.stem+'_budget_counts.json');q.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

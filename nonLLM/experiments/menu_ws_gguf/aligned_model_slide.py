"""Train two non-LLM MLP architectures under the frozen Jonathan-aligned contract.

Render a deterministic, editable scientific reporting figure (PNG/PDF/SVG).
Linear results come from the completed reference-alignment run; no old cohorts.
"""
import argparse, copy, itertools, json, os, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import jonathan_aligned as aligned

ARCHES={'mlp64':[64],'mlp128_64':[128,64]}
METHODS={'SFT':'sft','SFT + Reward':'exact','GRPO':'grpo'}
def network(dims,hidden):
    import torch
    layers=[];last=dims
    for width in hidden:
        layers.extend([torch.nn.Linear(last,width),torch.nn.ReLU()]);last=width
    layers.append(torch.nn.Linear(last,2))
    return torch.nn.Sequential(*layers)

def prepare(source,out):
    assert not out.exists();out.mkdir(parents=True)
    p=json.loads((source/'protocol.json').read_text())
    assert json.loads((source/'status.json').read_text())['phase']=='complete'
    for name,digest in p['reference_sha256'].items():
        assert aligned.runtime.digest(source/'reference'/Path(name).name)==digest
    aligned.write(out/'protocol.json',{'source_run':str(source.resolve()),'source_protocol_sha256':aligned.runtime.digest(source/'protocol.json'),
       'source_features_sha256':aligned.runtime.digest(source/'features.joblib'),
       'source_grid_hashes':{s:aligned.runtime.digest(source/'data'/f'grid_{s}.npz') for s in ['train','val','test']},
       'architectures':{'linear':[] ,**ARCHES},'seeds':aligned.SEEDS,
       'method_contract':'Same features, labels, stage rewards, training prices, inference prices and split as jonathan_aligned_v1; two stages, three seeds averaged before validation thresholds.',
       'methods':{'SFT':'Weighted CE, 3epochs lr1e-4 batch8; best val AUC',
                  'SFT + Reward':'SFT warm start; exact E[tokenF1-.1*C/929]+.03*entropy; 3epochs lr1e-5 batch8; best val expected reward',
                  'GRPO':'SFT warm start; group8 normalized advantages+fixed-reference KL.02; no PPO clipping; 2epochs lr1e-4 batch16; best val expected reward. Stage adaptation, not verified slide GRPO run.'},
       'selection':'For each model/method choose highest validation lenient accuracy, break ties by lowest Table5 cost; thresholds 0:.05:1. Save lock before test inference.',
       'limits':'Previously inspected test; descriptive. MLP sizes are architectural variants, not random forests/SVMs. Decision compute excluded from Table5 budget, same as author readout.'})
    (out/'code').mkdir();__import__('shutil').copy2(__file__,out/'code'/Path(__file__).name)
    aligned.write(out/'status.json',{'phase':'prepared'})

def train(source,out,arch,seed):
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    p=json.loads((source/'protocol.json').read_text())
    folder=out/'training'/arch/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    for stage in ['ret','qa']:
        x={s:torch.from_numpy(np.load(source/'data'/f'{stage}_{s}.npy')).to(device) for s in ['train','val']}
        y={};reward={}
        for part in x:
            g=np.load(source/'data'/f'grid_{part}.npz');rw=g['f1']-.1*aligned.TRAIN_PRICE.reshape(1,2,2)/aligned.SS
            y[part]=torch.tensor(g['yret'] if stage=='ret' else g['yqa'].ravel(),dtype=torch.long,device=device)
            reward[part]=torch.tensor(rw.max(2) if stage=='ret' else rw.reshape(-1,2),dtype=torch.float32,device=device)
        warm=None
        for method in ['sft','exact','grpo']:
            dest=folder/f'{stage}_{method}';dest.mkdir(exist_ok=True)
            if (dest/'complete.json').exists():
                if method=='sft':warm={k:torch.tensor(v,device=device) for k,v in joblib.load(dest/'best.joblib')['state'].items()}
                continue
            torch.manual_seed(seed);net=network(p['dimensions'][stage],ARCHES[arch]).to(device)
            if method!='sft':net.load_state_dict(warm)
            reference=copy.deepcopy(net).eval()
            for v in reference.parameters():v.requires_grad_(False)
            lr=1e-5 if method=='exact' else 1e-4;batch=16 if method=='grpo' else 8;epochs=2 if method=='grpo' else 3
            optimizer=torch.optim.AdamW(net.parameters(),lr=lr)
            counts=torch.bincount(y['train'],minlength=2).float();weights=counts.sum()/(2*counts.clamp_min(1))
            best=-float('inf');curve=[];gen=torch.Generator().manual_seed(seed)
            for ep in range(1,epochs+1):
                losses=[];net.train()
                for ix in torch.randperm(len(x['train']),generator=gen).split(batch):
                    ix=ix.to(device);xx=x['train'][ix];lp=net(xx).log_softmax(-1)
                    if method=='sft':loss=-(weights[y['train'][ix]]*lp[torch.arange(len(ix),device=device),y['train'][ix]]).mean()
                    elif method=='exact':loss=-(lp.exp()*reward['train'][ix]).sum(1).mean()+.03*(lp.exp()*lp).sum(1).mean()
                    else:
                        with torch.no_grad():ref=reference(xx).log_softmax(-1)
                        loss=aligned.group_loss(lp,reward['train'][ix],ref)
                    optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
                net.eval()
                with torch.no_grad():
                    prob=net(x['val']).softmax(-1);val_reward=float((prob*reward['val']).sum(1).mean())
                    auc=float(roc_auc_score(y['val'].cpu().numpy(),prob[:,1].cpu().numpy()))
                criterion=auc if method=='sft' else val_reward
                curve.append({'epoch':ep,'train_loss':float(np.mean(losses)),'val_auc':auc,'val_expected_reward':val_reward})
                if criterion>best+1e-8:
                    best=criterion;best_epoch=ep;best_state={k:v.detach().clone() for k,v in net.state_dict().items()}
                    joblib.dump({'architecture':arch,'stage':stage,'method':method,'seed':seed,'epoch':ep,
                                 'state':{k:v.cpu().numpy() for k,v in best_state.items()}},dest/'best.joblib')
                aligned.write(dest/'curve.json',curve)
                print(arch,seed,stage,method,ep,round(criterion,6),flush=True)
            aligned.write(dest/'complete.json',{'best_epoch':best_epoch,'last_epoch':ep,'best_criterion':best,'lr':lr,'batch':batch})
            if method=='sft':warm=best_state
    aligned.write(out/f'{arch}_{seed}.json',{'phase':'complete'})

def predict(source,out,arch,method,part):
    import torch
    torch.set_num_threads(2)
    if arch=='linear':return aligned.predict(source,{'sft':'sft','exact':'sft_reward_both','grpo':'sft_grpo_stage'}[method],part)
    p=json.loads((source/'protocol.json').read_text());result=[]
    for stage in ['ret','qa']:
        x=torch.from_numpy(np.load(source/'data'/f'{stage}_{part}.npy'));probs=[]
        for seed in aligned.SEEDS:
            z=joblib.load(out/'training'/arch/f'seed{seed}'/f'{stage}_{method}'/'best.joblib')
            net=network(p['dimensions'][stage],ARCHES[arch]);net.load_state_dict({k:torch.tensor(v) for k,v in z['state'].items()});net.eval()
            with torch.no_grad():probs.append(net(x).softmax(-1)[:,0].numpy())
        result.append(np.mean(probs,axis=0))
    return result

def evaluate(source,out):
    p=json.loads((source/'protocol.json').read_text());val=np.load(source/'data/grid_val.npz')['correct'].reshape(-1,4)
    locks=[];thresholds=list(itertools.product(np.linspace(0,1,21),repeat=3))
    for arch in ['linear',*ARCHES]:
        for name,method in METHODS.items():
            pred=predict(source,out,arch,method,'val');best=None
            for threshold in thresholds:
                a=aligned.route(pred,threshold);n=int(val[np.arange(len(val)),a].sum());cost=float(aligned.PRICE[a].mean())
                key=(n,-cost)
                if best is None or key>best[0]:best=(key,threshold)
            locks.append({'architecture':arch,'training':name,'method':method,'threshold':list(best[1]),'validation_correct':best[0][0],'validation_cost':-best[0][1]})
    aligned.write(out/'validation_lock.json',locks)
    test=np.load(source/'data/grid_test.npz')['correct'].reshape(-1,4);rows=[];details=[]
    for lock in locks:
        pred=predict(source,out,lock['architecture'],lock['method'],'test');a=aligned.route(pred,lock['threshold']);correct=test[np.arange(len(test)),a]
        rows.append({**lock,'n':len(test),'test_correct':int(correct.sum()),'accuracy_pct':100*int(correct.sum())/len(test),'cost':float(aligned.PRICE[a].mean()),'cost_pct_ss':100*float(aligned.PRICE[a].mean())/aligned.SS})
        details.extend({'architecture':lock['architecture'],'method':lock['method'],'id':qid,'route':int(route),'lenient':int(v),'cost':float(aligned.PRICE[route])} for qid,route,v in zip(p['ids']['test'],a,correct))
    assert len(rows)==9
    # Independent row-level contract checks against the shared reference grid.
    idmap={qid:i for i,qid in enumerate(p['ids']['test'])}
    for row in details:
        assert row['lenient']==test[idmap[row['id']],row['route']]
        r,a=divmod(row['route'],2);assert row['cost']==77+814*r+3+115*a
    for row in rows:
        selected=[r for r in details if r['architecture']==row['architecture'] and r['method']==row['method']]
        assert len(selected)==1481 and sum(r['lenient'] for r in selected)==row['test_correct']
    fit_count=len(list((out/'training').glob('*/*/*/complete.json')));assert fit_count==36
    # The three reused linear rows must match the prior verified alignment run.
    old=json.loads((source/'results.json').read_text())
    for row in rows[:3]:
        name={'sft':'sft','exact':'sft_reward_both','grpo':'sft_grpo_stage'}[row['method']]
        prior=next(r for r in old if r['method']==name and r['objective']=='max_accuracy')
        assert row['test_correct']==prior['correct'] and abs(row['cost']-prior['cost'])<1e-9
    aligned.write(out/'results.json',rows);pd.DataFrame(rows).to_csv(out/'results.csv',index=False);pd.DataFrame(details).to_csv(out/'per_question.csv.gz',index=False)
    aligned.write(out/'verification.json',{'additional_mlp_fits':fit_count,'tested_model_recipe_pairs':9,'test_n':1481,'audited_prediction_rows':len(details),'linear_reuse_matches_prior_verified_run':True,'reference_grid_and_all_weak_call_costs_match':True})
    aligned.write(out/'status.json',{'phase':'complete'})
    print(pd.DataFrame(rows)[['architecture','training','accuracy_pct','cost_pct_ss']].to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--phase',choices=['prepare','train','evaluate'],required=True);ap.add_argument('--architecture',choices=list(ARCHES));ap.add_argument('--seed',type=int,default=17);a=ap.parse_args()
    if a.phase=='prepare':prepare(a.source,a.out)
    elif a.phase=='train':train(a.source,a.out,a.architecture,a.seed)
    else:evaluate(a.source,a.out)

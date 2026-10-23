import itertools
import unittest
import numpy as np
import torch
from extended_policy import operating_points,joint_logprob,features,tasks


class ExtensionTests(unittest.TestCase):
    def test_operating_points_against_enumeration(self):
        em=np.array([[1,0,1,1],[0,1,0,1],[0,0,1,1],[1,1,0,0],[0,1,1,1]])
        pr=np.array([.2,.2,.6,.8,.8]);qw=np.array([.3,.3,.5,.9,.5]);qs=np.array([.7,.1,.7,.4,.1])
        prices=np.array([2.,6.7,6.7,11.4]);budget=9.4;rows=[]
        for tr,tq in itertools.product(np.r_[np.unique(pr),np.inf],np.r_[np.unique(np.r_[qw,qs]),np.inf]):
            r=pr<tr;q=np.where(r,qs,qw)<tq;c=2*r.astype(int)+q.astype(int)
            rows.append((prices[c].mean(),em[np.arange(len(em)),c].mean()))
        got=operating_points(em,pr,qw,qs,prices,budget)
        for objective in ['cheap95','max_accuracy','ss_budget']:
            rs=[r for r in rows if (r[1]>=.95*em[:,3].mean()-1e-12 if objective=='cheap95' else r[0]<=budget+1e-12 if objective=='ss_budget' else True)]
            key=lambda r:(r[0],-r[1]) if objective=='cheap95' else (-r[1],r[0])
            expected=min(rs,key=key)
            self.assertTrue(np.allclose((got[objective]['cost'],got[objective]['accuracy']),expected))

    def test_stable_joint_probabilities_and_gradients(self):
        r=torch.tensor([0.,30.,-30.],requires_grad=True);q=torch.tensor([[0.,0.],[30.,-30.],[-30.,30.]],requires_grad=True)
        lp=joint_logprob(r,q);p=lp.exp()
        self.assertTrue(torch.isfinite(lp).all())
        self.assertTrue(torch.allclose(p.sum(1),torch.ones(3),atol=1e-6))
        self.assertGreater(float(torch.sigmoid(-r.detach())[1]),0.)
        (p*torch.tensor([[0.,1.,0.,1.]])).sum().backward()
        self.assertTrue(torch.isfinite(r.grad).all());self.assertTrue(torch.isfinite(q.grad).all())

    def test_document_overlap_excludes_query_and_unpaid_branch(self):
        g={'question':'Who wrote Alpha?','evidence_W':'Query: Who wrote Alpha?\nSelected references: [1] unrelated document',
           'evidence_S':'[2] secret'}
        a=features(g,'ret','W',None,None)
        self.assertEqual(a[1][-4],0.)
        g.update(evidence_S='strong information',answers=['gold'],em_lenient=[1,1,1,1])
        self.assertEqual(features(g,'ret','W',None,None),a)

    def test_training_grid(self):
        ts=tasks();self.assertEqual(len(ts),144)
        for t in ts:self.assertEqual(t['groups']==0,t['method']=='exact')


if __name__=='__main__':unittest.main()

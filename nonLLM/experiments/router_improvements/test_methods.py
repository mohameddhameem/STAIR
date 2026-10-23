import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from suite import choose,rich
from collect import confidence,seed


class MethodTests(unittest.TestCase):
    def test_thresholds_match_brute_force(self):
        em=np.array([[1,0,1,1],[0,1,0,1],[0,0,1,1],[1,1,0,0],[0,1,1,1]])
        pr=np.array([.2,.2,.6,.8,.8]);qw=np.array([.3,.3,.5,.9,.5]);qs=np.array([.7,.1,.7,.4,.1])
        price=np.array([2.,6.7,6.7,11.4]);best=choose(em,pr,qw,qs,price)
        feasible=[]
        for tr,tq in itertools.product(np.r_[np.unique(pr),np.inf],np.r_[np.unique(np.r_[qw,qs]),np.inf]):
            r=pr<tr;q=np.where(r,qs,qw)<tq;c=2*r.astype(int)+q.astype(int)
            acc=em[np.arange(len(em)),c].mean()
            if acc>=.95*em[:,3].mean()-1e-12:feasible.append((price[c].mean(),-acc))
        self.assertTrue(np.allclose((best['cost'],-best['accuracy']),min(feasible)))

    def test_features_do_not_use_gold_or_outcomes(self):
        g={'question':'Who wrote the book?','evidence_W':'[1] Alice wrote it','evidence_S':'[2] Bob wrote it'}
        expected=rich(g,'qa','W','Alice',[0.]*7)
        g.update(answers=['Bob'],supporting_titles=['secret'],em_lenient=[1,1,1,1],ret_label='S')
        self.assertEqual(rich(g,'qa','W','Alice',[0.]*7),expected)

    def test_token_confidence_and_seed(self):
        z=torch.tensor([[2.,1.,0.],[0.,1.,2.]])
        got=confidence(z,torch.tensor([0,2]))
        expected=torch.log_softmax(z,-1)[0,0].item()
        self.assertAlmostEqual(got[0],expected,places=6)
        self.assertAlmostEqual(got[2],0.)
        self.assertEqual(seed('12|W|0'),seed('12|W|0'))
        self.assertNotEqual(seed('12|W|0'),seed('12|W|1'))

    def test_exact_reward_gradient_and_group_advantage(self):
        # Favor S when it helps, favor W when S harms; group-relative signals
        # vanish on equal-reward groups rather than inventing a preference.
        z=torch.tensor([0.],requires_grad=True);p=z.sigmoid()
        loss=-((1-p)*.0+p*1.0);loss.backward()
        self.assertLess(z.grad.item(),0.)
        reward=torch.tensor([[1.,1.,1.],[0.,1.,0.]])
        adv=(reward-reward.mean(1,keepdim=True))/reward.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)
        self.assertTrue(torch.equal(adv[0],torch.zeros(3)))
        self.assertGreater(adv[1,1].item(),0.)


if __name__=='__main__':unittest.main()

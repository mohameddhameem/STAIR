import unittest
import numpy as np
import torch
import replay

class ReplayContracts(unittest.TestCase):
    def test_round2_deliver_pays_full_weak(self):
        t={'cw':20.,'cw2':4.,'cs':100.,'qw':1,'qs':0}
        q,c,r=replay.branch(t,3)
        np.testing.assert_array_equal(c,[20,104])
        np.testing.assert_allclose(r,[.98,-.144])
        self.assertEqual(replay.branch(t,1)[1][1],120.)

    def test_a1_no_standard_deviation(self):
        lp=torch.tensor([.4,.6],dtype=torch.float64).log().requires_grad_()
        rewards=torch.tensor([.1,.9],dtype=torch.float64);draws=torch.tensor([0,0,0,1])
        sampled=rewards[draws];adv=sampled-sampled.mean()
        entropy=-(lp.exp()*lp).sum()
        expected=-(adv*lp[draws]).mean()-.03*entropy
        torch.testing.assert_close(replay.a1_loss(lp,rewards,draws),expected)

    def test_zero_group_variance_only_entropy(self):
        lp=torch.tensor([.3,.7]).log().requires_grad_()
        r=torch.tensor([1.,-1.]);actions=torch.zeros(8,dtype=torch.long)
        expected=.03*(lp.exp()*lp).sum()
        torch.testing.assert_close(replay.a1_loss(lp,r,actions),expected)

    def test_weak_first_baseline_differs_from_direct_strong(self):
        rows=[{'tid':'x','cw':20.,'cw2':4.,'cs':100.,'qw':0,'qs':1}]
        m,_=replay.metric(rows,1,[0.],.5,{'x':.8})
        self.assertEqual(m['cost_pct_always_escalate'],100.)
        m,_=replay.metric(rows,3,[0.],.5,{'x':.8})
        self.assertAlmostEqual(m['cost_pct_always_escalate'],104/120*100)

if __name__=='__main__':unittest.main()

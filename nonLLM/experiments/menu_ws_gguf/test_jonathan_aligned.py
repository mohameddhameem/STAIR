"""Reward, update and weak-first cost contracts for the alignment experiment."""
import unittest
import numpy as np
import torch
import jonathan_aligned as experiment

class Contracts(unittest.TestCase):
    def test_token_f1_is_author_reward_not_lenient(self):
        self.assertAlmostEqual(experiment.token_f1('The blue whale',['blue whale']),1.)
        self.assertAlmostEqual(experiment.token_f1('blue',['blue whale']),2/3)
        # Production tokenF1 has no official yes/no special-case guard.
        self.assertAlmostEqual(experiment.token_f1('yes no',['yes']),2/3)

    def test_weak_first_and_training_cost_difference(self):
        np.testing.assert_array_equal(experiment.PRICE,[77+3,77+3+115,77+814+3,77+814+3+115])
        self.assertEqual(experiment.SS,814+115)
        self.assertEqual(experiment.TRAIN_PRICE[-1],experiment.SS)
        self.assertGreater(experiment.PRICE[-1],experiment.SS)

    def test_constant_reward_zero_gradient_at_reference(self):
        logits=torch.tensor([[.2,-.4]],requires_grad=True)
        logp=logits.log_softmax(-1)
        loss=experiment.group_loss(logp,torch.ones((1,2)),logp.detach())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(logits.grad,torch.zeros_like(logits),atol=1e-6,rtol=0)

    def test_group_update_rewards_better_action(self):
        torch.manual_seed(19)
        logits=torch.zeros((128,2),requires_grad=True)
        logp=logits.log_softmax(-1)
        loss=experiment.group_loss(logp,torch.tensor([[0.,1.]]).expand(128,-1),logp.detach())
        loss.backward()
        self.assertLess(float(logits.grad[:,1].sum()),0.)

    def test_cascade_qa_branch_is_selected_by_retrieval(self):
        pred=[np.array([.8,.2]),np.array([.9,.1,.9,.1])]
        np.testing.assert_array_equal(experiment.route(pred,(.5,.5,.5)),[0,3])

if __name__=='__main__':unittest.main()

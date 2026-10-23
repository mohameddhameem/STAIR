import copy,unittest
import numpy as np
import torch
import alternatives as a

class AlternativeContracts(unittest.TestCase):
 def test_lexical_features_ignore_gold_and_answers(self):
  q={'question':'Which city hosted the 2012 event?','ctxs_candidate':['London: The city hosted it.','Paris: Other event.'],'answers':['London']};before=a.extras(q);paid=a.extras(q,{'selected':[0]})
  q.update(answers=['SECRET'],ctxs_gt=['SECRET'],em_lenient=[1]*4,weak_answer='SECRET',strong_answer='SECRET')
  np.testing.assert_array_equal(before,a.extras(q));np.testing.assert_array_equal(paid,a.extras(q,{'selected':[0]}))
 def test_paid_features_do_not_read_unselected_evidence(self):
  q={'question':'Which city?','ctxs_candidate':['London: A city.','Paris: Other.']};before=a.extras(q,{'selected':[0]});q['ctxs_candidate'][1]='SECRET city city city'
  np.testing.assert_array_equal(before,a.extras(q,{'selected':[0]}))
  self.assertEqual(len(a.extras(q,{'selected':[]})),len(before));self.assertTrue(np.isfinite(a.extras(q,{'selected':[]})).all())
 def test_unscaled_advantage_retains_reward_magnitude(self):
  sampled=torch.tensor([[0.,1.]]);p=torch.tensor([[.5,.5]]);all_r=sampled.clone()
  for method in ['group_centered','expected_baseline']:
   adv=a.unscaled_advantages(sampled,p,all_r,method);scaled=a.unscaled_advantages(sampled*10,p,all_r*10,method);torch.testing.assert_close(scaled,adv*10)
 def test_exact_baseline_handles_a_single_sampled_action(self):
  sampled=torch.tensor([[0.,0.]]);p=torch.tensor([[.5,.5]]);all_r=torch.tensor([[0.,1.]])
  torch.testing.assert_close(a.unscaled_advantages(sampled,p,all_r,'group_centered'),torch.zeros(1,2))
  torch.testing.assert_close(a.unscaled_advantages(sampled,p,all_r,'expected_baseline'),torch.full((1,2),-.5))
 def test_retrieval_planner_does_not_use_counterfactual_qa_evidence(self):
  p={'normalizer':1.,'training_ret_mean_units':[1,10],'training_qa_mean_units':[[1,2],[1,2]]};rscore=np.array([[[.4,.5],[.7,.9]]]);qscore=np.array([[[.4,.8],[.3,.7]]]);choice={'lambda_ret':0.,'lambda_qa':0.}
  before=a.choose('outcome',rscore,qscore,choice,p);qscore[0,0]=[1,0];after=a.choose('outcome',rscore,qscore,choice,p)
  np.testing.assert_array_equal(before,after);self.assertEqual((before[0][0],before[1][0]),(1,1))
 def test_gain_router_fixes_only_qa_not_retrieval(self):
  r,q=a.choose('gain',np.array([-.1,.2]),None,{'threshold':0.},{})
  np.testing.assert_array_equal(r,[0,1]);np.testing.assert_array_equal(q,[1,1])
 def test_validation_match_ss_differs_from_95_percent(self):
  candidates=[{'accuracy':.7,'units':20.},{'accuracy':.72,'units':100.}]
  self.assertEqual(a.select(candidates,'cheap95',.72,100)['units'],20.)
  self.assertEqual(a.select(candidates,'match_ss_accuracy',.72,100)['units'],100.)
 def test_cpu_residual_matches_torch_model(self):
  x=np.array([[1.,2.]],np.float32);head={'base.weight':np.array([[.2,.3]],np.float32),'base.bias':np.array([.1],np.float32),'hidden.weight':np.eye(2,dtype=np.float32),'hidden.bias':np.zeros(2,np.float32),'output.weight':np.array([[.4,.5]],np.float32),'output.bias':np.array([.2],np.float32)}
  expected=torch.nn.functional.linear(torch.tensor(x),torch.tensor(head['base.weight']),torch.tensor(head['base.bias']))+torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(torch.tensor(x),torch.tensor(head['hidden.weight']),torch.tensor(head['hidden.bias']))),torch.tensor(head['output.weight']),torch.tensor(head['output.bias']))
  np.testing.assert_allclose(a.forward_np(head,x),expected.numpy(),rtol=1e-6)

if __name__=='__main__':unittest.main()

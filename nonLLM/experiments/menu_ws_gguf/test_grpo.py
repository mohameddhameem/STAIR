import unittest,copy
import numpy as np
import torch
import grpo

class GRPOContract(unittest.TestCase):
 def test_route_probabilities_follow_ww_ws_sw_ss(self):
  pr=torch.logit(torch.tensor([.3]));pq=torch.logit(torch.tensor([[.2,.8]]))
  p=grpo.joint_logprob(pr,pq).exp()
  torch.testing.assert_close(p,torch.tensor([[.56,.14,.06,.24]]))
  torch.testing.assert_close(p.sum(1),torch.ones(1))
 def test_normalized_advantages_are_finite_on_equal_rewards(self):
  a,std=grpo.normalized_advantages(torch.tensor([[1.,1.,1.],[0.,1.,0.]]))
  self.assertTrue(torch.isfinite(a).all());torch.testing.assert_close(a[0],torch.zeros(3));self.assertEqual(float(std[0]),0.)
 def test_policy_gradient_increases_rewarded_action(self):
  z=torch.tensor([0.],requires_grad=True);lp=torch.stack([torch.nn.functional.logsigmoid(-z),torch.nn.functional.logsigmoid(z)],1)
  adv,_=grpo.normalized_advantages(torch.tensor([[0.,1.]]));old=lp.detach();ratio=(lp-old).exp();loss=-(ratio*adv).mean();loss.backward()
  self.assertLess(float(z.grad),0.)
 def test_direct_cost_grid_has_no_sunk_weak_calls(self):
  p={'models':{'qwen7':{'params_b':7},'qwen72':{'params_b':72},'llama3':{'params_b':3},'llama70':{'params_b':70}}}
  record={'prompt_tokens':100,'generated_tokens':10,'text':'Paris'}
  rr={r:{'x':dict(record)} for r in grpo.RET};qq={a:{'x|'+r:dict(record) for r in grpo.RET} for a in grpo.QA}
  em,cost=grpo.direct_grid(['x'],{'x':{'answers':['Paris']}},rr,qq,p)
  self.assertTrue(em.all());self.assertAlmostEqual(float(cost[0,1,1]),grpo.units(72,100,10)+grpo.units(70,100,10),places=5)
 def test_supervision_masks_all_incorrect_questions(self):
  em=np.array([[[0,0],[0,0]],[[1,0],[1,1]],[[0,1],[0,0]]]);cost=np.array([[[1,2],[3,4]]]*3)
  yr,mr,yq,mq=grpo.targets(em,cost)
  np.testing.assert_array_equal(mr,[False,True,True]);self.assertEqual(int(yr[1]),0);self.assertEqual(int(yq[2,0]),1)
 def test_stage_choices_use_only_selected_branch(self):
  pr=np.array([.9]);pq=np.array([[.1,.8]])
  r,a=grpo.choices(pr,pq,0,0,[1,10],[[1,2],[1,2]],12)
  self.assertEqual((int(r[0]),int(a[0])),(1,1))
  changed=pq.copy();changed[0,0]=.99
  rr,aa=grpo.choices(pr,changed,0,0,[1,10],[[1,2],[1,2]],12)
  np.testing.assert_array_equal(a,aa)
 def test_learned_only_selection_does_not_silently_add_ss(self):
  learned=[{'type':'learned','accuracy':.65,'units':50.}];fixed={'type':'fixed','accuracy':.72,'units':100.}
  self.assertEqual(grpo.select(learned,'max_accuracy',.72,100)['type'],'learned')
  self.assertEqual(grpo.select(learned+[fixed],'max_accuracy',.72,100)['type'],'fixed')
  self.assertIsNone(grpo.select(learned,'cheap95',.72,100))
 def test_features_do_not_read_gold_or_unexecuted_answers(self):
  q={'question':'Which city?','ctxs_candidate':['CityA','CityB'],'answers':['A']};rec={'selected':[0]}
  before=grpo.classical.view(q,'answering',rec,'qwen7');q.update(answers=['secret'],strong_answer='secret',weak_answer='secret',labels=[1]*4)
  self.assertEqual(before,grpo.classical.view(q,'answering',rec,'qwen7'))

if __name__=='__main__':unittest.main()

import copy,unittest,tempfile,json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import next_studies as s

class NextStudiesContract(unittest.TestCase):
 def test_cascade_costs_charge_all_paid_calls_on_selected_branch(self):
  ret=np.array([[2.,10.]]);qa=np.array([[[1.,4.],[3.,5.]]])
  np.testing.assert_array_equal(s.cascade_costs(ret,qa),[[[3,7],[15,20]]])
  self.assertGreater(s.cascade_costs(ret,qa)[0,1,1],ret[0,1]+qa[0,1,1])
 def test_weak_retrieval_review_does_not_see_answers_or_strong_outputs(self):
  q={'question':'Which city hosted the event?','ctxs_candidate':['London: It hosted the event.','Paris: Other.']};wr={'selected':[0],'prompt_tokens':100,'generated_tokens':20,'finish_reason':'stop','text':'Selected [1]'}
  before=s.cascade_retrieval_features(q,wr);q.update(answers=['secret'],ctxs_gt=['secret'],weak_answer='secret',strong_answer='secret',strong_retrieval_output='secret')
  np.testing.assert_array_equal(before,s.cascade_retrieval_features(q,wr))
 def test_weak_answer_review_uses_only_paid_weak_answer(self):
  q={'question':'Which city?','ctxs_candidate':['London: city','Paris: city']};rec={'selected':[0]};weak={'text':'London','prompt_tokens':20,'generated_tokens':2,'finish_reason':'stop'}
  before=s.cascade_qa_features(q,rec,weak,'qwen7');q.update(strong_answer='SECRET',answers=['SECRET'])
  np.testing.assert_array_equal(before,s.cascade_qa_features(q,rec,weak,'qwen7'))
 def test_pair_predictions_handle_missing_and_constant_classes(self):
  x=np.arange(8).reshape(4,2);y=np.array([[0,1,2,1]]*4);model=ExtraTreesClassifier(n_estimators=1,random_state=0).fit(x,y)
  gain,harm=s.pair_prediction(model,x[:1]);np.testing.assert_array_equal(gain,[[-1,0,1,0]]);np.testing.assert_array_equal(harm,[[1,0,0,0]])
 def test_crc_gate_is_nested_and_charges_harm_not_signed_regret(self):
  confidence=np.array([.9,.8,.7,.6]);eligible=np.ones(4,bool);harm=np.array([False,False,True,True])
  result=s.calibrated_threshold(confidence,eligible,harm,.2);self.assertFalse(result['no_switch']);self.assertEqual(result['switches'],2);self.assertEqual(result['calibration_harm'],0);self.assertLessEqual(result['criterion_bound'],.2)
  r,a=s.apply_gate(np.zeros(4,int),np.zeros(4,int),confidence,result);np.testing.assert_array_equal(r,[0,0,1,1]);np.testing.assert_array_equal(a,[0,0,1,1])
 def test_crc_falls_back_to_exact_no_switch_for_unattainable_alpha(self):
  result=s.calibrated_threshold(np.array([.9,.8]),np.ones(2,bool),np.zeros(2,bool),.001);self.assertTrue(result['no_switch'])
  r,a=s.apply_gate(np.zeros(2,int),np.zeros(2,int),np.array([.9,.8]),result);np.testing.assert_array_equal(r,[1,1]);np.testing.assert_array_equal(a,[1,1])
 def test_retrieval_plan_does_not_use_unexecuted_qa_predictions(self):
  pred={'pair_r':np.array([[0.,.1,.2,0.]]),'pair_q':np.array([[[0.,.5],[.3,.1]]]),'cost_r':np.ones((1,6)),'cost_q':np.ones((1,2,2))}
  r,a=s.decisions(pred,'pair_staged',0);self.assertEqual((int(r[0]),int(a[0])),(1,0));pred['pair_q'][0,0]=[100.,-100.]
  rr,aa=s.decisions(pred,'pair_staged',0);np.testing.assert_array_equal(r,rr);np.testing.assert_array_equal(a,aa)
 def test_qa_fold_indices_keep_question_branches_together(self):
  np.testing.assert_array_equal(s.qa_indices(np.array([2,5])),[4,5,10,11])

if __name__=='__main__':unittest.main()

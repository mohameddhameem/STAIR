"""Check deployment information boundaries before expensive generation/training."""
import copy,unittest
import numpy as np
from sklearn.dummy import DummyClassifier
import run as runtime
import train

class Contract(unittest.TestCase):
 def setUp(self):
  self.q={'question':'Which city hosted the event?','ctxs_candidate':['A: The event was in City A.','B: Other evidence.'],'answers':['City A'],'ctxs_gt':['A: The event was in City A.']}
 def test_gold_metadata_never_changes_router_or_executor_inputs(self):
  q=copy.deepcopy(self.q);rec={'selected':[0]};base=[runtime.ret_messages(q),runtime.qa_messages(q,rec),train.view(q,'retrieval'),train.view(q,'answering',rec,'qwen7')]
  q.update(answers=['Unrelated secret answer'],ctxs_gt=['Secret gold evidence'],em_lenient=[1]*9,weak_answer='Unexecuted answer',strong_answer='Another unexecuted answer')
  self.assertEqual(base,[runtime.ret_messages(q),runtime.qa_messages(q,rec),train.view(q,'retrieval'),train.view(q,'answering',rec,'qwen7')])
 def test_evidence_contains_only_paid_selection(self):
  evidence=runtime.evidence(self.q,{'selected':[1]})
  self.assertIn('Other evidence',evidence);self.assertNotIn('City A',evidence)
 def test_selection_bounds_and_missing_selection(self):
  self.assertEqual(runtime.parse_ids('Query: [9]\nSelected references: [2] [2] [0] [1000]',2),[1])
  self.assertEqual(runtime.parse_ids('No relevant references.',2),[])
 def test_probability_classes_are_mapped_to_global_menu_indices(self):
  m=DummyClassifier(strategy='prior').fit([[1],[2],[3]],[0,2,2])
  p=train.proba({'model':m,'numeric':True},['a'],[[1]])
  self.assertEqual(p.shape,(1,3));self.assertEqual(p[0,1],0);self.assertAlmostEqual(p[0,2],2/3)

if __name__=='__main__':unittest.main()

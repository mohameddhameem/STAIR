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
  m=DummyClassifier(strategy='prior').fit([[1],[2],[3]],[0,1,1])
  p=train.proba({'model':m,'numeric':True},['a'],[[1]])
  self.assertEqual(p.shape,(1,2));self.assertAlmostEqual(p[0,0],1/3);self.assertAlmostEqual(p[0,1],2/3)

 def test_only_weak_and_strong_models_are_scheduled(self):
  self.assertEqual([m[0] for m in runtime.MODELS],['qwen7','qwen72','llama3','llama70'])
  self.assertEqual(train.RET,['qwen7','qwen72']);self.assertEqual(train.QA,['llama3','llama70'])
  m=DummyClassifier(strategy='prior').fit([[1],[2]],[1,1])
  np.testing.assert_array_equal(train.proba({'model':m,'numeric':True},['a'],[[1]]),[[0,1]])

 def test_70b_tools_are_explicitly_none_for_both_request_paths(self):
  self.assertIsNone(runtime.template_options('llama70')['chat_template_kwargs']['tools'])
  self.assertEqual(runtime.template_options('llama3'),{})
  runtime.validate_rendered_prompt('llama70','system: helpful assistant; user: QA')
  with self.assertRaises(AssertionError):runtime.validate_rendered_prompt('llama70','Given the following functions, please respond with JSON')
  with self.assertRaises(AssertionError):runtime.validate_rendered_prompt('llama70','Environment: ipython')

 def test_port_allocator_avoids_an_active_listener(self):
  import socket
  with socket.socket() as occupied:
   occupied.bind(('127.0.0.1',0));occupied.listen()
   self.assertNotEqual(runtime.allocate_port(),occupied.getsockname()[1])

 def test_concurrent_writer_receives_each_job_once_and_bounds_inflight(self):
  import threading
  lock=threading.Lock();barrier=threading.Barrier(2);active=0;peak=0
  def work(job):
   nonlocal active,peak
   with lock:active+=1;peak=max(peak,active)
   barrier.wait(timeout=2)
   with lock:active-=1
   return job
  self.assertEqual(sorted(runtime.bounded_results(work,range(8),2)),list(range(8)))
  self.assertEqual(peak,2)

 def test_shared_kv_budget_waits_and_releases_on_error(self):
  import threading
  from concurrent.futures import ThreadPoolExecutor
  budget=runtime.TokenBudget(10);started=threading.Event();entered=threading.Event()
  def work():
   started.set()
   with budget.reserve(5,lambda:False):entered.set()
  with ThreadPoolExecutor(max_workers=1) as pool:
   with budget.reserve(6,lambda:False):
    f=pool.submit(work);self.assertTrue(started.wait(2));self.assertFalse(entered.wait(.02));self.assertEqual(budget.used,6)
   f.result(timeout=2)
  self.assertEqual(budget.used,0)
  with self.assertRaises(ValueError):
   with budget.reserve(11,lambda:False):pass
  with self.assertRaises(RuntimeError):
   with budget.reserve(5,lambda:False):raise RuntimeError('request failed')
  self.assertEqual(budget.used,0)

if __name__=='__main__':unittest.main()

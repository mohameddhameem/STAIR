import unittest
import numpy as np
from architecture import pre_features,qa_features,chooser_features,policy_options,lock_options
from architecture_combine import combined_cells

class ArchitectureTests(unittest.TestCase):
    def test_combiner_only_changes_paid_ss_answers(self):
        cells=np.array([0,1,2,3,3])
        out=combined_cells(cells,np.array([.5,-.5]),0.)
        np.testing.assert_equal(out,[0,1,2,2,3])
        np.testing.assert_equal(cells,[0,1,2,3,3])
    def test_pre_features_ignore_unavailable_information(self):
        inst={'question':'Who built the bridge?','ctxs_candidate':['Bridge: Built by Smith.'],
              'answers':['SECRET'],'supporting_titles':['SECRET'],'level':'SECRET'}
        a=pre_features(inst)
        inst.update(answers=['CHANGED'],supporting_titles=['CHANGED'],level='CHANGED',
                    evidence_W='CHANGED',em_lenient=[1,1,1,1])
        self.assertEqual(a,pre_features(inst))
        self.assertNotIn('SECRET',str(a))

    def test_evidence_statistics_do_not_count_copied_query(self):
        g={'question':'Who built the bridge?','evidence_W':'Query: Who built the bridge?\nSelected references: [1] Unrelated text'}
        r=qa_features(g,'W','Smith',[0]*7)
        self.assertEqual(r['num'][4],0.)

    def test_paid_weak_and_strong_calls(self):
        direct=np.array([12.,17.,41.,47.]);review=np.array([12.,19.,41.,48.])
        fixed=np.array([np.nan,np.nan,41.,48.]);post=np.array([np.nan,np.nan,48.,48.])
        p=np.array([[0.,0.,0.,1.,1.],[1.,0.,0.,0.,-1.]])
        qa=np.array([[1.,1.],[0.,0.]])
        opts=policy_options(p,qa,np.array([1.,-1.]),(direct,review,fixed,post))
        d={(f,k):(cells,cost) for f,k,cells,cost in opts}
        cells,cost=d['pre_ret_qa','ret_0.0_qa_0.0']
        np.testing.assert_equal(cells,[3,0]);self.assertEqual(cost[cells[0]],48.)
        cells,cost=d['post_answer_chooser','threshold_0.0']
        np.testing.assert_equal(cells,[2,3]);np.testing.assert_equal(cost[cells],[48.,48.])
        cells,cost=d['fixed_S_qa','direct_SS_fallback']
        np.testing.assert_equal(cost[cells],[47.,47.])

    def test_ss_fallback_makes_budget_feasible(self):
        direct=np.array([12.,17.,41.,47.]);paid=direct+1
        p=np.zeros((2,5));q=np.zeros((2,2));ch=np.zeros(2)
        em=np.array([[0,0,0,1],[0,0,0,1]])
        opts=policy_options(p,q,ch,(direct,paid,paid,paid))
        locks=lock_options(opts,em,47.)
        for name,l in locks.items():
            if name.endswith('ss_budget'):
                self.assertLessEqual(l['val_cost'],47.)
                self.assertEqual(l['val_accuracy'],1.)

if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import classical

class ClassicalContracts(unittest.TestCase):
    def test_weighted_action_error_equals_reward_regret(self):
        r=np.array([[.9,.1],[-.1,.4],[.2,.2],[0.,-.3]])
        y,w=classical.target(r)
        np.testing.assert_array_equal(y,[0,1,0,0])
        for action in [np.zeros(4,dtype=int),np.ones(4,dtype=int)]:
            regret=r.max(1)-r[np.arange(4),action]
            expected=np.abs(r[:,0]-r[:,1])*(action!=y)
            np.testing.assert_allclose(regret,expected)
        self.assertTrue((w>0).all());self.assertAlmostEqual(float(w.mean()),1.)

    def test_svm_calibration_groups_and_probabilities(self):
        text=[f'group{g} '+('deliver success' if j%2==0 else 'escalate failure') for g in range(6) for j in range(4)]
        groups=np.repeat(np.arange(6),4);y=np.tile([0,1,0,1],6);w=np.ones(24)
        pipe=Pipeline([('features',TfidfVectorizer()),('model',LinearSVC(C=1.,dual='auto',random_state=17))])
        bundle,audits=classical.fit_candidate(pipe,'linear_svm',text,y,w,groups,17)
        self.assertEqual(len(audits),3)
        held=[]
        for a in audits:
            self.assertFalse(set(groups[a['fit_indices']])&set(groups[a['held_indices']]))
            held.extend(a['held_indices'])
        self.assertEqual(sorted(held),list(range(24)))
        p=classical.predict(bundle,['deliver success unseen','escalate failure unseen'])
        self.assertTrue(np.isfinite(p).all());self.assertTrue(((p>=0)&(p<=1)).all())
        self.assertGreater(p[0],p[1])

if __name__=='__main__':unittest.main()

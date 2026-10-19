import io
src = open("/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/final_signal_tests.py").read()
old = """def auc(ps, ys):
    pairs = sorted(zip(ps, ys)); pos = [i+1 for i,(p,y) in enumerate(pairs) if y==1]
    nP, nN = len(pos), len(pairs)-len(pos)
    return (sum(pos)-nP*(nP+1)/2)/(nP*nN) if nP and nN else float("nan")"""
new = """def auc(ps, ys):
    # tie-corrected (midrank) AUC -- equal feature values share their average rank
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    ranks = [0.0]*len(ps); i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and ps[order[j+1]] == ps[order[i]]: j += 1
        mid = (i + j)/2 + 1
        for k in range(i, j+1): ranks[order[k]] = mid
        i = j + 1
    pos = [ranks[k] for k, y in enumerate(ys) if y == 1]
    nP = len(pos); nN = len(ys) - nP
    return (sum(pos) - nP*(nP+1)/2)/(nP*nN) if nP and nN else float("nan")"""
assert src.count(old) == 1
open("/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/final_signal_tests.py","w").write(src.replace(old, new))
print("auc fixed (midrank)")

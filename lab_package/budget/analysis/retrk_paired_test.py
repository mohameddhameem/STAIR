import json, sys, random, math
grid = {g["idx"]: g for g in map(json.loads, (l for l in open(sys.argv[1]) if l.strip()))}
rows = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
S = {}
for r in rows:
    if r["ans"] == "S": S.setdefault(r["ctx"], {})[r["i"]] = r["correct"]
qids = sorted(S["u5"]); n = len(qids)
ws1 = {i: grid[i]["em_standard"][1] for i in qids}; ss1 = {i: grid[i]["em_standard"][3] for i in qids}
print("n =", n, " WS(k=1) =", round(100*sum(ws1.values())/n,2), " SS =", round(100*sum(ss1.values())/n,2))
random.seed(20261124)
def test(name, a, b):
    """a = candidate, b = reference; paired."""
    d = [a[i]-b[i] for i in qids]; win = sum(1 for x in d if x>0); loss = sum(1 for x in d if x<0)
    # exact McNemar (binomial, two-sided)
    m = win+loss; k = min(win,loss)
    p = sum(math.comb(m,j) for j in range(0,k+1))*2/2**m if m else 1.0; p = min(1.0,p)
    boots = []
    for _ in range(10000):
        idx = [random.randrange(n) for _ in range(n)]; boots.append(sum(d[j] for j in idx)/n*100)
    boots.sort(); lo, hi = boots[249], boots[9749]
    print(f"{name:28s} Δ = {sum(d)/n*100:+.2f} pts  wins {win} / losses {loss} / ties {n-m}   McNemar exact p = {p:.4f}   95% CI [{lo:+.2f}, {hi:+.2f}]")
for ctx in ("u5","u10","v5","v10"):
    test(f"{ctx} vs WS(k=1)", S[ctx], ws1)
test("u10 vs u5", S["u10"], S["u5"])
test("u10 vs SS", S["u10"], ss1)
# where do the union gains come from? questions SS gets right but WS(k=1) gets wrong
gap = [i for i in qids if ss1[i]==1 and ws1[i]==0]
print(f"\nSS-right/WS-wrong questions: {len(gap)}; u5 fixes {sum(S['u5'][i] for i in gap)}, u10 fixes {sum(S['u10'][i] for i in gap)}")
brk = [i for i in qids if ws1[i]==1 and S['u10'][i]==0]
print(f"WS-right questions broken by u10: {len(brk)}  (of {sum(ws1.values())})")

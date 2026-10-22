import re, string, sys, json

# STANDARD ruler -- certified 13 Oct 2026: reproduces the published AgentTTS 0.746
def base(s): return ''.join(c for c in s.lower() if c not in set(string.punctuation))
def norm(s): return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', base(s)).split())
def words(s): return norm(s).split()
QUANT = {"more","less","fewer","over","under","about","around","approximately",
         "least","most","nearly","almost","than","up","between","roughly"}
def _seq(nd, hy):
    N, H = words(nd), words(hy)
    if not N: return False
    for i in range(len(H) - len(N) + 1):
        if H[i:i+len(N)] == N:
            if i > 0 and H[i-1] in QUANT: continue
            return True
    return False
def LEN(a, gs): return max(int(norm(a) == norm(x) or _seq(x, a) or _seq(a, x)) for x in gs)

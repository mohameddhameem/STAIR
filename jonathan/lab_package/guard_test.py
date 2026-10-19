# guard_test.py -- exercises compose_eval.py's test-mode guards in a sandbox.
#
# SAFE BY CONSTRUCTION:
#   * LOCK and OUT are redirected to a throwaway /tmp folder; nothing under
#     /mnt/hdd/jonathan/runs is written or deleted.
#   * score_split is replaced by a stub that raises immediately, so even when
#     all guards pass, no model loads and no test instance is scored.
#   * The grid cache and dataset are opened read-only (compose_eval's main()
#     loads them before the guards; that is harmless).
#
# Expected output: five lines, every one ending PASS.
import os, sys, json, tempfile, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose_eval as CE

SB = tempfile.mkdtemp(prefix="guard_test_")
CE.LOCK = os.path.join(SB, "theta_lock.json")
CE.OUT = os.path.join(SB, "out")
os.makedirs(CE.OUT)

def stub(*a, **k):
    raise RuntimeError("ALL GUARDS PASSED -- control reached the scoring step "
                       "(stubbed; nothing was scored, nothing was read).")
CE.score_split = stub

def run_case(name, expect):
    sys.argv = ["compose_eval.py", "--mode", "test"]
    try:
        CE.main()
        print(f"[{name}] FAIL: main() returned normally (no guard fired)")
    except SystemExit as e:
        verdict = "PASS" if expect in str(e) else "FAIL"
        print(f"[{name}] {verdict}: {e}")
    except RuntimeError as e:
        verdict = "PASS" if expect == "GUARDS_PASS" else "FAIL"
        print(f"[{name}] {verdict}: {e}")

LOCK_OK = {"primary": "theta_bar",
           "ss_val_accuracy": 53.5, "bar": 50.83,   # bar = 0.95 x SS val accuracy
           "theta_bar": {"theta_ret": 0.5, "theta_qa": 0.5,
                         "val_accuracy": 52.0, "cost": 5.0}}

# Case 1: theta_lock.json does not exist at all -> clean refusal
run_case("1 missing lock file   ", "does not exist")

# Case 2: lock exists but holds no theta -> refusal
json.dump({}, open(CE.LOCK, "w"))
run_case("2 no locked theta     ", "no locked theta")

# Case 3: locked val accuracy below the lock's bar (95% of SS val) -> refusal
low = json.loads(json.dumps(LOCK_OK))
low["theta_bar"]["val_accuracy"] = 50.0          # below bar 50.83
json.dump(low, open(CE.LOCK, "w"))
run_case("3 below 95%-of-SS bar ", "iterate on validation")

# Case 4: good lock, but test_result.json already exists -> vault refusal
json.dump(LOCK_OK, open(CE.LOCK, "w"))
open(os.path.join(CE.OUT, "test_result.json"), "w").write("{}")
run_case("4 vault already read  ", "already")

# Case 5: good lock, empty vault -> all guards pass; scoring is stubbed
os.remove(os.path.join(CE.OUT, "test_result.json"))
run_case("5 all guards pass     ", "GUARDS_PASS")

shutil.rmtree(SB)
print("sandbox removed, real runs/ untouched")

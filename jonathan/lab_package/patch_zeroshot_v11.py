#!/usr/bin/env python3
# =============================================================================
# patch_zeroshot_v11.py — VERSION: patch_zeroshot_v11 v1.0 2026-09-15
#
# Upgrades zeroshot_router.py v1.0 -> v1.1 with ONE new capability:
#   --only_splits  (comma list; default "val,test" preserves v1.0 behavior)
# so the 7B judge can score the TRAIN split — the raw material every fork
# option (A distill / D stage-matched labels) needs.
#
# SAFETY: exact-match edits with drift guards (refuses if the source is not
# byte-wise what we reviewed on 15 Sep); original backed up to
# zeroshot_router_v10.bak before writing.
#
# USAGE (lab, in lab_package/):  python3 patch_zeroshot_v11.py
# =============================================================================
import shutil, sys

VERSION = "patch_zeroshot_v11 v1.0 2026-09-15"
PATH = "zeroshot_router.py"

OLD_ARG = '''    ap.add_argument("--limit", type=int, default=0, help="0 = all of val+test; N = first N (smoke)")'''
NEW_ARG = OLD_ARG + '''
    ap.add_argument("--only_splits", default="val,test",
                    help="comma list of splits to score (v1.1; default preserves v1.0 behavior)")'''

OLD_SEL = '''    ids = [i for i in sp["val"] + sp["test"] if i in rows]'''
NEW_SEL = '''    want = [w.strip() for w in a.only_splits.split(",") if w.strip()]
    ids = [i for w in want for i in sp[w] if i in rows]'''


def main():
    print(VERSION)
    s = open(PATH, encoding="utf-8").read()
    for tag, old in (("arg-block", OLD_ARG), ("selection-line", OLD_SEL)):
        n = s.count(old)
        assert n == 1, f"drift guard: {tag} found {n} times (expected 1) — source differs from the reviewed 15-Sep read; NOT patching"
    assert "zeroshot_router v1.0 2026-09-06" in s, "drift guard: version string not found — NOT patching"
    shutil.copyfile(PATH, "zeroshot_router_v10.bak")
    s = s.replace(OLD_ARG, NEW_ARG)
    s = s.replace(OLD_SEL, NEW_SEL)
    s = s.replace("zeroshot_router v1.0 2026-09-06", "zeroshot_router v1.1 2026-09-15 (--only_splits)")
    open(PATH, "w", encoding="utf-8").write(s)
    print("patched -> zeroshot_router.py is now v1.1 (backup: zeroshot_router_v10.bak)")
    print("verify:  head -3 zeroshot_router.py | grep VERSION   (must say v1.1)")


if __name__ == "__main__":
    main()

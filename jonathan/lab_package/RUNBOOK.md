# Lab GPU Runbook — Full HotpotQA Label Run (7,405 instances)

Machine: `jonathan@10.4.8.30` · data dir `/mnt/hdd/jonathan` · expected GPU: RTX 6000 Ada 48GB
Prereq: VPN (or campus network) working; password already changed via `passwd`.

## 0. One-time setup (first session only)

```bash
# --- from your LAPTOP: copy the package and the pool file to the lab machine ---
scp lab_smoketest.py run_labels.py mint_labels.py jonathan@10.4.8.30:/mnt/hdd/jonathan/
scp tts_test.json jonathan@10.4.8.30:/mnt/hdd/jonathan/data/
#   (tts_test.json is 422 MB: download it once from Google Drive > AgentTTS/data/HotpotQA/
#    to your laptop first. On home upload speeds this can take a while — start it early.
#    If the lab machine has internet, downloading directly there is faster: ask me for the gdown recipe.)

# --- on the LAB MACHINE (ssh in) ---
cd /mnt/hdd/jonathan
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install torch transformers accelerate bitsandbytes
python lab_smoketest.py            # Phase 1 gate
python lab_smoketest.py --model    # Phase 2: downloads weak model, one generation
```

## 1. Smoke slice (ALWAYS before the full run) — ~15 min

```bash
source /mnt/hdd/jonathan/venv/bin/activate
python run_labels.py --data /mnt/hdd/jonathan/data/tts_test.json \
                     --out  /mnt/hdd/jonathan/slr_run --end 20
python mint_labels.py --data /mnt/hdd/jonathan/data/tts_test.json \
                      --out  /mnt/hdd/jonathan/slr_run
```
PASS check: `mint_summary.json` shows solved ≈ 8–12 of 20 (Colab mini-pilot got 10/20).
If it matches, the port is faithful — proceed.

## 2. The full run — inside tmux (NEVER a bare terminal)

```bash
tmux new -s slr          # create a session that survives disconnects
source /mnt/hdd/jonathan/venv/bin/activate
python run_labels.py --data /mnt/hdd/jonathan/data/tts_test.json \
                     --out  /mnt/hdd/jonathan/slr_run 2>&1 | tee -a slr_run.log
```
- Detach and leave it running: press `Ctrl+B`, then `D`. Close your laptop. Go live your life.
- Reattach later to check: `ssh` in, then `tmux attach -t slr`
- Quick status without attaching: `tail -20 /mnt/hdd/jonathan/slr_run.log` and `nvidia-smi`
- Expected wall time: ~15–30 h (the ETA prints in the log after the first 25 instances).
- Crash/reboot/anything: just rerun the SAME command — checkpoints resume automatically.
- Courtesy on a shared machine: check `nvidia-smi` before starting; if a labmate's job is
  using the GPU, coordinate with Dhanuja rather than double-loading the card.

## 3. Score + mint labels v4-full (CPU, minutes)

```bash
python mint_labels.py --data /mnt/hdd/jonathan/data/tts_test.json \
                      --out  /mnt/hdd/jonathan/slr_run
```
Expected: ~2,300–3,700 solved (41% ± 10 from the n=100 pilot), labels + full combo outcomes written.

## 4. Bring results home

```bash
# from your LAPTOP:
scp jonathan@10.4.8.30:/mnt/hdd/jonathan/slr_run/mint_summary.json .
scp jonathan@10.4.8.30:/mnt/hdd/jonathan/slr_run/sft_labels_hotpot_v4_alpha0.6.jsonl .
scp jonathan@10.4.8.30:/mnt/hdd/jonathan/slr_run/combo_results_full.jsonl .
```
Paste `mint_summary.json` to Claude → next phase: SFT at scale (can run on the same lab GPU).

## Troubleshooting quick table

| Symptom | Meaning | Fix |
|---|---|---|
| `no CUDA device visible` | driver/venv issue | `nvidia-smi` works? If not, ask Dhanuja |
| `GPU not clean` assert | someone else's job on the card | `nvidia-smi` shows whose; coordinate |
| OOM fail-fast raise | unexpected on 48GB | send Claude the log tail |
| `retrieval incomplete` assert | QA started before retrieval done | run phases in order / rerun `--phase all` |
| ETA looks wrong early | first instances include model download | judge after ~100 instances |

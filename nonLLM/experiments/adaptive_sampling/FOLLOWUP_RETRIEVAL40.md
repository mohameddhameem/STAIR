# Follow-up requested on 2026-10-15

Status: recorded, waiting for the current experiment to finish. This file is a
request record, not an automatic launcher. No running profile, worker or
controller has been changed.

The current experiment is `/data1/zmcheng/cs707_sampling40_instruct_gguf` on
serverA. It keeps one W retrieval and one S retrieval per question, followed by
40 answer samples for each model pair and fusion of each prefix.

After that run completes, run a separate experiment with 40 retrieval samples
per question for **each** retrieval model, W and S. Retain ordered samples so
retrieval prefix budgets can be compared, as with answer prefixes. Preserve
the public AgentTTS HotpotQA prompts and temperatures (retrieval 0.9, answer
0.7, fusion 0.3), the existing Instruct GGUF checkpoints and frozen data split.
Use serverA only. Keep current caches/results separate and immutable.

Before launching, specify the retrieval aggregation and downstream answer
budget grid explicitly. Forty retrieval candidates do not by themselves
specify whether to evaluate all 40 retrieval prefixes crossed with all 40
answer prefixes. That much larger grid must not be silently substituted for
the existing experiment. Do not launch the follow-up concurrently with the
current run. Check the current run's COMPLETE marker and successful analysis;
failure or controller exit alone is not completion.

## GPU 2 diagnosis, 2026-10-15 15:51 +08

Read-only inspection on serverA:

- NVIDIA L40S, UUID `GPU-b32c400f-6936-def0-5abc-21a38010e91c`, PCI
  `0000:81:00.0`, driver 580.178.04.
- Important device mapping: nvidia-smi index **2** has minor number **1**, so
  its device node is `/dev/nvidia1`, not `/dev/nvidia2`.
- Kernel log at 05:12:59 reports Xid 48 and 171 (uncorrectable framebuffer
  DRAM error), Xid 63 (new row marked for remapping, reset to activate), and
  Xid 154 (recovery action changed to Drain and Reset).
- Recovery action: Drain and Reset. Volatile DRAM uncorrectable errors: 2;
  aggregate: 6. Uncorrectable remapped rows: 4; pending: Yes; remapping
  failure: No. Spare-bank histogram: Max 191, Partial 1, Low 0, None 0.
- GPU 2 has no compute process and 0 MiB application memory. Its actual device
  node is held by nvidia-persistenced (PID 2313) and dcgm-exporter (PID 697346).
  PID 2173095 holds `/dev/nvidia2`, which is a different GPU's node.
- The active experiment continues on physical indices 0, 1 and 3. No reset,
  monitoring shutdown, service restart, reboot or ECC-mode change was made.

Likely recovery path after the current experiment: verify GPU identity by UUID
and all device handles again, release blocking monitoring clients with their
original state recorded, reset only the affected GPU, restore monitoring, then
check that pending remapping/recovery flags clear. Validate CUDA allocation,
compute and a small inference workload before reusing it. Historical aggregate
error counts need not disappear. Recovery is not yet verified; recurring new
uncorrectable errors require further hardware diagnosis. If other jobs block
reset, defer rather than stop those jobs.

References:

- https://docs.nvidia.com/deploy/gpu-debug-guidelines/gpu-node-triage.html
- https://docs.nvidia.com/deploy/nvidia-smi/

Update at 18:49 +08: the user subsequently authorized pausing the current run.
GPU2 was successfully reset after temporarily unloading idle NVIDIA graphics
modules as well as releasing monitoring handles. Modules and system monitoring
were restored. Row remapping is applied, recovery action is None and volatile
DRAM uncorrectable errors are zero. A 36 GiB two-pattern memory check and
matrix-compute checks passed. The same experiment resumed on all four GPUs;
see GGUF_RUN.md and the run-root maintenance audit. Retrieval-40 remains a
separate follow-up and was not introduced into the current run.

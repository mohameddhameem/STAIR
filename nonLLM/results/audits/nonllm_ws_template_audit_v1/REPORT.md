# Llama70 chat-template root cause and repair

The verified bartowski Llama-3.1-70B-Instruct Q4_K_M GGUF contains a legacy template with conditions `tools is not none`. The backend provides an empty list when no tools are requested. The template treats this as enabled tool mode, injects `Environment: ipython` plus instructions to respond with function-call JSON, and the model follows these instructions. A rendered-prompt comparison reproduces the issue: default and `tools=[]`/`tool_choice=none` retain those instructions; `chat_template_kwargs.tools=null` removes them.

Function-call JSON signatures in original Llama70 outputs: 12416/12810 train-extra, 969/1000 validation, 972/1000 test. Llama3 has zero such signatures across all splits. This is a prompt-rendering defect; it is not evidence that Llama70 is weaker than Llama3.

Repair: explicit `chat_template_kwargs={tools: null, date_string: "22 Oct 2026"}` in both actual generation and preflight rendering. A guard rejects unrequested tool instructions/environment. The date is frozen for reproducibility. Model weights, task messages, evidence, decoding and scorer remain unchanged.

Diagnostic verification uses only the first 12 train-extra IDs and Qwen7 evidence. All 12 corrected responses are ordinary text, with no function-call JSON. Lenient correct increases from 2/12 to 5/12; this small diagnostic sample is not a test-set estimate. See cases.json, ab_results.json and summary.json. Rendered default/none/empty prompts are saved as rendered_*.json.

Old results and router policies are invalidated because their labels and validation decisions include the broken Llama70 branch. The replacement run `runs/nonllm_common_ws_gguf_v2` reuses only Qwen7/Qwen72 retrieval and Llama3 answering, regenerates all 14810 Llama70 answers, then rebuilds router labels/fits and evaluates on the unchanged common test split. Old artifacts remain preserved for audit.

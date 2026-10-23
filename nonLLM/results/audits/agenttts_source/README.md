# AgentTTS source interpretation

The source audit is historical; a later source inspection additionally located `best config` comments in `archon_hotpotqa.py` recording Qwen72 retrieval1 and Llama3 answering40. The two adjacent numeric values in those comments are not fully explained, so this is not proof of an independently reproduced paper score. Official HotpotQA search has three retrieval and three answering candidates; the largest answerer is Llama3.1-70B. The appendix72B wording conflicts with the code/model tables.

The completed local experiment is a weak/strong four-model subset with K=1 and no fusion, rather than that original sampled recipe. Original Qwen AWQ/Llama loading is not numerically identical to third-party GGUF.

[Paper](https://arxiv.org/html/2508.00890v2) · [Official candidate menu](https://github.com/FairyFali/AgentTTS/blob/3f0c2df814c059d0df57a71056c3266ff02c8559/code_others/AgentTTS.py#L403-L411) · [Best-config comments](https://github.com/FairyFali/AgentTTS/blob/3f0c2df814c059d0df57a71056c3266ff02c8559/code_others/archon_hotpotqa.py#L657-L670).

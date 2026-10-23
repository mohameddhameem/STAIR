> INVALIDATED: 70B chat-template bug. See INVALID_RESULTS.md; corrected run is nonllm_common_ws_gguf_v2.

# Four-model WS2x2 GGUF direct non-LLM routing

Four-model WS2x2 subset of menu A; GGUF exploratory; NOT eligible for pooled bf16/AWQ headline tables without team adopting the identical GGUF copies

Standard common test IDs and author lenient ruler; Q4_K_M weights, q8_0 KV, K1/greedy/no fusion. Cached generation seed0 is shared by all router seeds; only router fitting varies. No bf16/AWQ pooling claim.

Strongest: 25.00% lenient, 1579.703 AgentTTS units.

| Policy | Router seed | Lenient % | Units | Cost % strongest | ms/router |
|---|---:|---:|---:|---:|---:|
| gguf-fixed-qwen7-llama3 | 0 | 54.00 | 151.481 | 9.59 | 0.000 |
| gguf-fixed-qwen7-llama70 | 0 | 23.80 | 200.589 | 12.70 | 0.000 |
| gguf-fixed-qwen72-llama3 | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-fixed-qwen72-llama70 | 0 | 25.00 | 1579.703 | 100.00 | 0.000 |
| gguf-majority-max_accuracy | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-majority-ss_budget | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-majority-cheap95 | 0 | 54.00 | 151.481 | 9.59 | 9.365 |
| gguf-logreg-max_accuracy | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-logreg-ss_budget | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-logreg-cheap95 | 0 | 54.00 | 151.481 | 9.59 | 9.786 |
| gguf-linear_svm-max_accuracy | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-linear_svm-ss_budget | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-linear_svm-cheap95 | 0 | 54.00 | 151.481 | 9.59 | 10.616 |
| gguf-multinomial_nb-max_accuracy | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-multinomial_nb-ss_budget | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-multinomial_nb-cheap95 | 0 | 54.00 | 154.471 | 9.78 | 9.878 |
| gguf-random_forest-max_accuracy | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-random_forest-ss_budget | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-random_forest-cheap95 | 0 | 54.00 | 151.481 | 9.59 | 0.000 |
| gguf-extra_trees-max_accuracy | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-extra_trees-ss_budget | 0 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-extra_trees-cheap95 | 0 | 54.00 | 151.481 | 9.59 | 0.000 |
| gguf-majority-max_accuracy | 1 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-majority-ss_budget | 1 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-majority-cheap95 | 1 | 54.00 | 151.481 | 9.59 | 9.421 |
| gguf-logreg-max_accuracy | 1 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-logreg-ss_budget | 1 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-logreg-cheap95 | 1 | 54.00 | 151.481 | 9.59 | 9.724 |
| gguf-linear_svm-max_accuracy | 1 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-linear_svm-ss_budget | 1 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-linear_svm-cheap95 | 1 | 54.00 | 151.481 | 9.59 | 10.654 |
| gguf-multinomial_nb-max_accuracy | 1 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-multinomial_nb-ss_budget | 1 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-multinomial_nb-cheap95 | 1 | 54.00 | 154.471 | 9.78 | 9.955 |
| gguf-random_forest-max_accuracy | 1 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-random_forest-ss_budget | 1 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-random_forest-cheap95 | 1 | 54.00 | 151.481 | 9.59 | 0.000 |
| gguf-extra_trees-max_accuracy | 1 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-extra_trees-ss_budget | 1 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-extra_trees-cheap95 | 1 | 54.00 | 151.481 | 9.59 | 0.000 |
| gguf-majority-max_accuracy | 2 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-majority-ss_budget | 2 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-majority-cheap95 | 2 | 54.00 | 151.481 | 9.59 | 9.262 |
| gguf-logreg-max_accuracy | 2 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-logreg-ss_budget | 2 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-logreg-cheap95 | 2 | 54.00 | 151.481 | 9.59 | 9.795 |
| gguf-linear_svm-max_accuracy | 2 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-linear_svm-ss_budget | 2 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-linear_svm-cheap95 | 2 | 54.00 | 151.481 | 9.59 | 10.625 |
| gguf-multinomial_nb-max_accuracy | 2 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-multinomial_nb-ss_budget | 2 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-multinomial_nb-cheap95 | 2 | 54.00 | 154.471 | 9.78 | 9.907 |
| gguf-random_forest-max_accuracy | 2 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-random_forest-ss_budget | 2 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-random_forest-cheap95 | 2 | 54.00 | 151.481 | 9.59 | 0.000 |
| gguf-extra_trees-max_accuracy | 2 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-extra_trees-ss_budget | 2 | 65.80 | 1531.793 | 96.97 | 0.000 |
| gguf-extra_trees-cheap95 | 2 | 54.00 | 151.481 | 9.59 | 0.000 |

Router cost reports CPU time separately; the parameter-FLOP proxy excludes classical CPU feature/model work. Classical fits are selected by validation log loss, not a fabricated neural epoch. Hyperparameters and all operating points were locked before test features/scoring. Previously examined test content means results are descriptive.

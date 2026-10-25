## Nano-Jev — `runs/nanojev-pretrain/nohotpot/model` on `runs/nanojev-pretrain/nohotpot/data`

| decision | n | T | acc | macro-F1 | NLL raw → cal | Brier raw → cal | ECE raw → cal | ms/decision |
|---|---|---|---|---|---|---|---|---|
| grounded | 500 | 1.52 | 0.858 | 0.858 | 0.354 → 0.340 | 0.212 → 0.206 | 0.052 → 0.033 | 0.3 |
| relevance | 2000 | 1.39 | 0.897 | 0.885 | 0.284 → 0.260 | 0.160 → 0.153 | 0.050 → 0.018 | 1.7 |
| sufficient | 500 | 3.42 | 0.710 | 0.708 | 0.823 → 0.591 | 0.437 → 0.401 | 0.149 → 0.065 | 1.1 |

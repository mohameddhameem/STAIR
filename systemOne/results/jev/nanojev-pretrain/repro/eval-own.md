## Nano-Jev — `runs/nanojev-pretrain/repro/model` on `runs/nanojev-pretrain/nano-jev-src/data`

| decision | n | T | acc | macro-F1 | NLL raw → cal | Brier raw → cal | ECE raw → cal | ms/decision |
|---|---|---|---|---|---|---|---|---|
| grounded | 500 | 1.56 | 0.858 | 0.858 | 0.338 → 0.327 | 0.208 → 0.202 | 0.062 → 0.049 | 0.3 |
| relevance | 1984 | 1.40 | 0.812 | 0.783 | 0.457 → 0.435 | 0.264 → 0.256 | 0.058 → 0.021 | 0.8 |
| sufficient | 1492 | 1.54 | 0.854 | 0.854 | 0.356 → 0.323 | 0.216 → 0.206 | 0.058 → 0.023 | 1.2 |

# MedAgentBench rulebooks (test 150, read once; learnt on val 60)

| Policy | Accuracy % | Without task 9 % | Cost (% of always-strong) | To a person | High-risk failures no one sees |
|---|---|---|---|---|---|
| untrained Qwen 7B (always-weak) | 12.67 | 14.07 | 23.4 | 0 | 30 |
| untrained Qwen 72B (always-strong) | 54.67 | 60.74 | 100.0 | 0 | 21 |
| fine-tuned Qwen 7B alone | 76.67 | 74.81 | 7.6 | 0 | 3 |
| fine-tuned Qwen 7B with expert order examples, alone | 76.00 | 81.48 | 7.9 | 0 | 19 |
| fine-tuned Qwen 32B alone | 80.00 | 88.89 | 49.4 | 0 | 17 |
| fine-tuned Qwen 72B alone | 68.67 | 76.30 | 106.1 | 0 | 17 |
| AL-16 (revision 1) W -> S: learnt rules (0 changes) | 76.67 | 74.81 | 7.6 | 0 | 3 |
| AL-17 (revision 1) W -> S -> person: learnt rules (0 changes) | 76.67 | 74.81 | 7.6 | 0 | 3 |
| AL-18 (revision 1) W -> M -> S -> person: learnt rules (1 changes) | 86.00 | 85.19 | 10.9 | 0 | 3 |
| AL-19 (revision 1) W searches -> W / M / S answers, or a person: learnt rules (0 changes) | 76.67 | 74.81 | 7.6 | 0 | 3 |
| AL-26 (revision 1) approach 3, W with expert orders (person 2 x): learnt rules (4 changes) | 94.00 | 93.33 | 100.0 | 15 | 2 |
| ... same, a person priced at 1 x always-strong | 94.00 | 93.33 | 68.3 | 15 | 2 |
| AL-27 (revision 1) approach 3, W with expert orders (person 1 x): learnt rules (4 changes) | 94.00 | 93.33 | 68.3 | 15 | 2 |
| AL-28 (revision 1) approach 3, W with expert orders (person 2 x, 4 x penalty): learnt rules (4 changes) | 94.00 | 93.33 | 104.2 | 15 | 2 |
| ... same, cost without the missed-high-risk penalty (AI + person only) | | | 100.0 | | |
| ... same, a person priced at 1 x always-strong | 94.00 | 93.33 | 72.4 | 15 | 2 |
| AL-29 (revision 1) approach 3, W with expert orders (person 1 x, 4 x penalty): learnt rules (4 changes) | 94.00 | 93.33 | 72.4 | 15 | 2 |
| ... same, cost without the missed-high-risk penalty (AI + person only) | | | 68.3 | | |
| AL-16 W -> S: learnt rules (0 changes) | 76.67 | 74.81 | 7.6 | 0 | 3 |
| AL-17 W -> S -> person: learnt rules (0 changes) | 76.67 | 74.81 | 7.6 | 0 | 3 |
| AL-18 W -> M -> S -> person: learnt rules (1 changes) | 86.00 | 85.19 | 11.0 | 0 | 3 |
| AL-19 W searches -> W / M / S answers, or a person: learnt rules (2 changes) | 86.00 | 85.19 | 9.4 | 0 | 3 |
| AL-20 AL-18 with a person at 1 x always-strong (Jonathan's price): learnt rules (1 changes) | 86.00 | 85.19 | 11.0 | 0 | 3 |
| AL-21 AL-19 with a person at 1 x always-strong (Jonathan's price): learnt rules (2 changes) | 86.00 | 85.19 | 9.4 | 0 | 3 |
| AL-22 AL-18, plus 4 x always-strong for a high-risk task answered wrong without a person: learnt rules (1 changes) | 86.00 | 85.19 | 25.5 | 0 | 3 |
| ... same, cost without the missed-high-risk penalty (AI + person only) | | | 11.0 | | |
| AL-23 AL-19, plus 4 x always-strong for a high-risk task answered wrong without a person: learnt rules (2 changes) | 86.00 | 85.19 | 23.8 | 0 | 3 |
| ... same, cost without the missed-high-risk penalty (AI + person only) | | | 9.4 | | |
| AL-24 AL-20 (person at 1 x), plus 4 x always-strong for a high-risk task answered wrong without a person: learnt rules (1 changes) | 86.00 | 85.19 | 25.5 | 0 | 3 |
| ... same, cost without the missed-high-risk penalty (AI + person only) | | | 11.0 | | |
| AL-25 AL-21 (person at 1 x), plus 4 x always-strong for a high-risk task answered wrong without a person: learnt rules (2 changes) | 86.00 | 85.19 | 23.8 | 0 | 3 |
| ... same, cost without the missed-high-risk penalty (AI + person only) | | | 9.4 | | |
| AL-26 approach 3, W with expert orders (person 2 x): learnt rules (3 changes) | 96.00 | 95.56 | 104.1 | 18 | 2 |
| ... same, a person priced at 1 x always-strong | 96.00 | 95.56 | 70.8 | 18 | 2 |
| AL-27 approach 3, W with expert orders (person 1 x): learnt rules (2 changes) | 97.33 | 97.04 | 73.2 | 30 | 0 |
| AL-28 approach 3, W with expert orders (person 2 x, 4 x penalty): learnt rules (3 changes) | 96.00 | 95.56 | 108.3 | 18 | 2 |
| ... same, cost without the missed-high-risk penalty (AI + person only) | | | 104.1 | | |
| ... same, a person priced at 1 x always-strong | 96.00 | 95.56 | 75.0 | 18 | 2 |
| AL-29 approach 3, W with expert orders (person 1 x, 4 x penalty): learnt rules (2 changes) | 97.33 | 97.04 | 73.2 | 30 | 0 |
| ... same, cost without the missed-high-risk penalty (AI + person only) | | | 73.2 | | |
| AL-30 approach 4, W with expert orders (person 2 x): learnt rules (1 changes) | 94.67 | 94.07 | 74.0 | 13 | 2 |
| ... same, a person priced at 1 x always-strong | 94.67 | 94.07 | 50.6 | 13 | 2 |
| AL-31 approach 4, W with expert orders (person 1 x): learnt rules (1 changes) | 94.67 | 94.07 | 50.6 | 13 | 2 |
| AL-32 approach 4, W with expert orders (person 2 x, 4 x penalty): learnt rules (1 changes) | 94.67 | 94.07 | 78.2 | 13 | 2 |
| ... same, cost without the missed-high-risk penalty (AI + person only) | | | 74.0 | | |
| ... same, a person priced at 1 x always-strong | 94.67 | 94.07 | 54.8 | 13 | 2 |
| AL-33 approach 4, W with expert orders (person 1 x, 4 x penalty): learnt rules (1 changes) | 94.67 | 94.07 | 54.8 | 13 | 2 |
| ... same, cost without the missed-high-risk penalty (AI + person only) | | | 50.6 | | |
| Jonathan: always-weak (untrained 7B) | 12.00 | | 18.7 | | |
| Jonathan: always-strong (untrained 72B) | 54.00 | | 100.0 | | |
| Jonathan: oracle | 63.33 | | - | | |
| Jonathan: Training 1 router | 61.33 | | 54.1 | | |
| Jonathan: Training 3 router (round 2) | 63.33 | | 81.5 | | |
| Jonathan: Training 5b (person on high risk) | 76.00 | | 94.5 | | |

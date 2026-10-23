# Non-LLM Config A cascade

Full weak-answer input. Six mean token-aware call costs from token_costs.json; excludes routing overhead.

Selected on validation: random_forest.

Test n=1481: lenient 50.03%, strict 44.16%.

Cost: 79.36% of direct SS; 65.35% of cascade SS.

Direct SS test accuracy: 52.94%.

Picks: {'WS': 658, 'SS': 811, 'SW': 12}.

This replays recorded Config A outcomes; it does not generate new model answers. Test was used in earlier experiments; this run locks choices before its new test evaluation.

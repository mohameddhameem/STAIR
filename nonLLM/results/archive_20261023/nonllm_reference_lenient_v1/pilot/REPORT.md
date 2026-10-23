# Non-LLM Config A cascade

ABLATION: weak-answer input absent. Six mean token-aware call costs from token_costs.json; excludes routing overhead.

Selected on validation: random_forest.

Test n=1481: lenient 49.90%, strict 43.96%.

Cost: 77.67% of direct SS; 63.96% of cascade SS.

Direct SS test accuracy: 52.94%.

Picks: {'WS': 677, 'SS': 743, 'SW': 55, 'WW': 6}.

This replays recorded Config A outcomes; it does not generate new model answers. Test was used in earlier experiments; this run locks choices before its new test evaluation.

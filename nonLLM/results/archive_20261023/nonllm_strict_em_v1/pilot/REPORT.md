> Corrected run: both legacy accuracy columns represent official normalized EM. See root protocol.json.

# Non-LLM Config A cascade

ABLATION: weak-answer input absent. Six mean token-aware call costs from token_costs.json; excludes routing overhead.

Selected on validation: random_forest.

Test n=1481: exact-match 43.82%, strict 43.82%.

Cost: 77.12% of direct SS; 63.51% of cascade SS.

Direct SS test accuracy: 47.20%.

Picks: {'WS': 683, 'SS': 720, 'SW': 70, 'WW': 8}.

This replays recorded Config A outcomes; it does not generate new model answers. Test was used in earlier experiments; this run locks choices before its new test evaluation.

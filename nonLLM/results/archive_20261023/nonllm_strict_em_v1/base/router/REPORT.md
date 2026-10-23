> Corrected run: both legacy accuracy columns represent official normalized EM. See root protocol.json.

# Non-LLM Config A cascade

Full weak-answer input. Six mean token-aware call costs from token_costs.json; excludes routing overhead.

Selected on validation: random_forest.

Test n=1481: exact-match 43.62%, strict 43.62%.

Cost: 77.16% of direct SS; 63.53% of cascade SS.

Direct SS test accuracy: 47.20%.

Picks: {'WS': 685, 'SS': 735, 'SW': 55, 'WW': 6}.

This replays recorded Config A outcomes; it does not generate new model answers. Test was used in earlier experiments; this run locks choices before its new test evaluation.

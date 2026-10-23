# Non-LLM Config A cascade

Full weak-answer input. Six mean token-aware call costs from token_costs.json; excludes routing overhead.

Selected on validation: multinomial_nb.

Test n=1481: lenient 55.98%, strict 44.56%.

Cost: 77.53% of direct SS; 63.84% of cascade SS.

Direct SS test accuracy: 59.08%.

Picks: {'WS': 686, 'SS': 794, 'WW': 1}.

This replays recorded Config A outcomes; it does not generate new model answers. Test was used in earlier experiments; this run locks choices before its new test evaluation.

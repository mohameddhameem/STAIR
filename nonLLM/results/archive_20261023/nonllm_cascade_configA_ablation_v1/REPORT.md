# Non-LLM Config A cascade

ABLATION: weak-answer input absent. Config A flat parameter-ratio proxy, NOT measured token costs; excludes routing overhead.

Selected on validation: random_forest.

Test n=1481: lenient 50.30%, strict 45.64%.

Cost: 89.00% of direct SS; 73.39% of cascade SS.

Direct SS test accuracy: 53.54%.

Picks: {np.str_('SS'): 629, np.str_('WW'): 104, np.str_('WS'): 472, np.str_('SW'): 276}.

This replays recorded Config A outcomes; it does not generate new model answers. Test was used in earlier experiments; this run locks choices before its new test evaluation.

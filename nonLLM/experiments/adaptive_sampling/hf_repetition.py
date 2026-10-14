"""Match HF repetition penalty over prompt and output using SGLang's hook.

The server must enable custom logit processors and have this module on PYTHONPATH.
The native output-only repetition penalty must be set to 1 to avoid double use.
"""
import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class HFRepetitionPenalty(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list=None):
        if custom_param_list is None:
            raise RuntimeError('Missing HF repetition parameters')
        for row, params in enumerate(custom_param_list):
            penalty = float(params['hf_repetition_penalty'])
            request = params['__req__']
            if penalty == 1:
                continue
            # Cache the unchanged prompt indices on the request, not globally.
            # They are released with the request and never shared across questions.
            cached = getattr(request, '_hf_prompt_penalty_indices', None)
            if cached is None or cached.device != logits.device:
                cached = torch.tensor(sorted(set(request.origin_input_ids)),
                                      dtype=torch.long, device=logits.device)
                request._hf_prompt_penalty_indices = cached
            generated = torch.tensor(sorted(set(request.output_ids)),
                                     dtype=torch.long, device=logits.device)
            indices = torch.cat((cached, generated))
            values = logits[row, indices]
            adjusted = torch.where(values < 0, values * penalty, values / penalty)
            # Repeated IDs receive the same original score and penalty once.
            logits[row].scatter_(0, indices, adjusted)
        return logits


def apply_to_request(body, penalty):
    body['sampling_params'].update({
        'repetition_penalty': 1.0,
        'custom_params': {'hf_repetition_penalty': float(penalty)},
    })
    body['custom_logit_processor'] = HFRepetitionPenalty.to_str()

"""Frozen gold-independent answer reader, served by the same 7B GGUF model."""
import json
import re
from answer_parser import extract_answer
from grounded_answer import EXAMPLES, SCHEMA, SYSTEM, norm


async def extract(session, client, key, question, response):
    # A copied question asserts no answer, even if the reader knows its solution.
    if norm(response).strip(' .?!') == norm(question).strip(' .?!'):
        return {'answer': None, 'evidence': None, 'grounding_valid': False,
                'method': 'question_echo_abstention', 'seconds': 0, 'usage': {}}
    polar = bool(re.match(r'^(?:is|are|was|were|do|does|did|has|have|had)\b', question.strip(), re.I))
    deterministic = extract_answer(response)
    if deterministic and not re.search(r'\w', deterministic):
        deterministic = None
    if polar and deterministic not in ['yes', 'no']:
        deterministic = None
    if deterministic is not None:
        return {'answer': deterministic, 'evidence': response, 'grounding_valid': True,
                'method': 'deterministic', 'seconds': 0, 'usage': {}}
    system = SYSTEM + (' For this yes/no question, extract the yes/no stance actually asserted in the response, including an implied stance supported by the quoted response. Do not correct the stance using outside facts.' if polar else '')
    schema = SCHEMA if not polar else {**SCHEMA, 'properties': {**SCHEMA['properties'], 'answer': {'enum': ['yes', 'no', None]}}}
    messages = [{'role': 'system', 'content': system}]
    for q, r, output in EXAMPLES:
        messages.extend([{'role': 'user', 'content': 'Question: ' + q + '\nAssistant response:\n' + r},
                         {'role': 'assistant', 'content': json.dumps(output)}])
    messages.append({'role': 'user', 'content': 'Question: ' + question + '\nAssistant response:\n' + response})
    result = await client.generate(session, key, messages, 0, maximum=192, schema=schema)
    output = result['outputs'][0]
    try:
        parsed = json.loads(output['text'])
        answer, quote = parsed['answer'], parsed['evidence']
        valid = bool(answer and quote and norm(quote) in norm(response))
        if valid and norm(quote).strip(' .?!') == norm(question).strip(' .?!'):
            valid = False
        if valid and not (polar and answer in ['yes', 'no']):
            valid = norm(answer).strip(' ."\'') in norm(quote)
    except (ValueError, KeyError, TypeError):
        parsed = {'unparsed_text': output['text']}
        answer, quote, valid = None, None, False
    return {'answer': answer if valid else None, 'evidence': quote if valid else None,
            'grounding_valid': valid, 'raw_extraction': parsed, 'method': 'llm_grounded_gguf',
            'seconds': result['seconds'], 'usage': output['meta_info']}

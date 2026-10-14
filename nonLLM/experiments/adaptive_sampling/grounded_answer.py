"""Optional postprocessor. Receives question/response only, never gold answers."""
import json,re,time
from answer_parser import extract_answer
SCHEMA={'type':'object','properties':{'answer':{'type':['string','null']},'evidence':{'type':['string','null']}},'required':['answer','evidence'],'additionalProperties':False}
SYSTEM='You are a literal answer-span extractor, not a question solver. Read the assistant response and copy the shortest answer span it actually asserts. Never correct wrong facts or compute an answer. Keep number words exactly as written. The answer must be a contiguous span of the evidence, and evidence must be copied exactly from the response. If there is no asserted answer, return JSON null for answer and evidence. Copied questions, instructions, and reference lists alone do not assert an answer. Return JSON only.'
EXAMPLES=[('What is the capital of France?','The answer given is Rome.',{'answer':'Rome','evidence':'The answer given is Rome.'}),('What is 2 + 2?','The stated result is seven.',{'answer':'seven','evidence':'The stated result is seven.'}),('Who wrote this book?','Who wrote this book? Please answer the question.',{'answer':None,'evidence':None})]

def norm(s):return re.sub(r'\s+',' ',s).strip().lower()
async def extract(session,endpoint,tokenizer,question,response):
 polar=bool(re.match(r'^(?:is|are|was|were|do|does|did|has|have|had)\b',question.strip(),re.I))
 deterministic=extract_answer(response)
 if polar and deterministic not in ['yes','no']:deterministic=None
 if deterministic is not None:return {'answer':deterministic,'evidence':response,'grounding_valid':True,'method':'deterministic','seconds':0,'usage':{}}
 system=SYSTEM+(' For this yes/no question, extract the yes/no stance actually asserted in the response, including an implied stance supported by the quoted response. Do not correct the stance using outside facts.' if polar else '')
 schema=SCHEMA if not polar else {**SCHEMA,'properties':{**SCHEMA['properties'],'answer':{'enum':['yes','no',None]}}}
 messages=[{'role':'system','content':system}]
 for q,r,out in EXAMPLES:messages.extend([{'role':'user','content':'Question: '+q+'\nAssistant response:\n'+r},{'role':'assistant','content':json.dumps(out)}])
 messages.append({'role':'user','content':'Question: '+question+'\nAssistant response:\n'+response})
 text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 body={'input_ids':tokenizer.encode(text,add_special_tokens=False),'sampling_params':{'temperature':0,'max_new_tokens':192,'json_schema':json.dumps(schema),'top_k':50,'top_p':1,'repetition_penalty':1}}
 start=time.time()
 async with session.post(endpoint+'/generate',json=body) as res:
  raw=await res.json()
  if res.status!=200:raise RuntimeError(str(raw))
 parsed=json.loads(raw['text']);answer=parsed['answer'];quote=parsed['evidence'];valid=bool(answer and quote and norm(quote) in norm(response))
 if valid and not (polar and answer in ['yes','no']):
  valid=norm(answer).strip(' ."\'') in norm(quote)
 return {'method':'llm_grounded','answer':answer if valid else None,'evidence':quote if valid else None,'grounding_valid':valid,'raw_extraction':parsed,'seconds':time.time()-start,'usage':raw.get('meta_info',{})}

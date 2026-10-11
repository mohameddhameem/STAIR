"""Execute pinned upstream prompt methods without importing its GPU dependencies."""
import ast, re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

def load_protocol(root):
    root=Path(root)
    source=ast.parse((root/'code_others/archon_hotpotqa.py').read_text())
    env={'re':re,'Counter':Counter,'print_log':False}
    methods={}
    for cls,method in [('Retrieval','retrieve'),('CustomGenerator','generate'),('Merge','run')]:
        node=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name==cls)
        fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==method)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<upstream>','exec'),env)
        methods[cls]=env[method]
    tree=ast.parse((root/'code_others/archon/completions/components/prompts.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_fuser_prompt')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<upstream>','exec'),env)
    class Capture:
        def generate_from_messages(self,messages,**kwargs):
            self.messages=messages
            return ['captured']
    def prompt(kind,example,retrieval=None,answers=None):
        content='Query: '+example['question']+'References: '
        for i,ref in enumerate(example['ctxs_candidate']):content+=f'\n[{i+1}] '+ref
        conv=[{'role':'user','content':content}]
        if kind=='fusion':
            return [{'role':'system','content':'You are a helpful assistant who fuses multiple answers'}, {'role':'user','content':env['make_fuser_prompt'](conv,answers)}]
        capture=Capture(); obj=SimpleNamespace(model=capture,temperature=.9 if kind=='retrieval' else .7)
        if kind=='retrieval':methods['Retrieval'](obj,conv,[])
        else:
            # Preserve upstream Merge.run, including repeated IDs and fallback.
            state={'candidates':[]}; methods['Merge'](SimpleNamespace(),conv,{'candidates':[retrieval]},state)
            methods['CustomGenerator'](obj,conv,state['candidates'])
        return capture.messages
    return prompt

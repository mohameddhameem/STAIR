"""Conservative, gold-independent extraction; abstain on unmarked long text."""
import re

def extract_answer(text):
    clean=text.strip().replace('**','').replace('`','')
    clean=re.sub(r'^\s*(?:assistant\s*:\s*)','',clean,flags=re.I)
    direct=re.match(r'^(yes|no)(?:[,.;:!?]|\s*$)',clean,re.I)
    if direct:return direct.group(1).lower()
    found=re.findall(r'(?:\bfinal\s+answer|\banswer)\s*(?::|\bis\b)\s*([^\n]+)',clean,re.I)
    if found:
        candidate=found[-1].strip()
        direct=re.match(r'^(yes|no)(?:[,.;:!?]|\s*$)',candidate,re.I)
        if direct:return direct.group(1).lower()
        candidate=re.split(r'(?<=[.!?])\s+',candidate,maxsplit=1)[0].strip().strip(' ."\'')
        if 0<len(candidate.split())<=16 and not re.search(r'\b(?:cannot|unknown|not provided|not enough)\b',candidate,re.I):return candidate
        return None
    if '\n' not in clean and len(clean.split())<=6 and '?' not in clean and not re.search(r'\b(?:is|are|was|were|has|have|had|served)\b',clean,re.I) and not re.search(r'\b(?:references|query|provided|assistant|cannot|unknown)\b',clean,re.I):
        return clean.strip(' ."\'') or None
    return None

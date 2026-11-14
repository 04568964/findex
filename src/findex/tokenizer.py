import re
import unicodedata

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+([-'][a-zA-Z0-9]+)*")

def tokenizer(data):
    data = unicodedata.normalize('NFKC', data)
    data = data.lower()
    data = data.replace('’', "'")
    data = data.replace('—', '-')
    for match in TOKEN_PATTERN.finditer(data):
        yield match.group(0)
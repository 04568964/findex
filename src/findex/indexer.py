from collections import defaultdict, Counter
from findex.models import Posting, DocMeta
from findex.iter_documents import read_file
from findex.tokenizer import tokenizer

def build_index(data_path):
    inv_index = defaultdict(list)
    doc_meta={}

    for doc_id, (filename ,text) in enumerate(read_file(data_path)):
        tokens = list(tokenizer(text))
        doc_meta[doc_id] = DocMeta(doc_id=doc_id,filename=filename,length = len(tokens))
        term_freqs = Counter(tokens)
        for term, freq in term_freqs.items():
            posting = Posting(doc_id=doc_id, frequensy=freq)
            inv_index[term].append(posting)
    return dict(inv_index), doc_meta

from array import array

def build_index_array_test(data_path):
    inv_index = defaultdict(lambda: array('I')) 
    doc_meta = {}

    for doc_id, (filename, text) in enumerate(read_file(data_path)):
        tokens = list(tokenizer(text))
        doc_meta[doc_id] = DocMeta(doc_id=doc_id, filename=filename, length=len(tokens))
        term_freqs = Counter(tokens)
        
        for term, freq in term_freqs.items():
            inv_index[term].extend([doc_id, freq])
            
    return dict(inv_index), doc_meta




def search_set(inv_index, term1, term2):
    p1 = inv_index.get(term1,[])
    p2 = inv_index.get(term2,[])

    set1 = {p.doc_id for p in p1}
    set2 = {p.doc_id for p in p2}

    return list(set1&set2)

def search_merge(inv_index, term1, term2):
    p1 = inv_index.get(term1,[])
    p2 = inv_index.get(term2,[])

    result_docs=[]
    i=0
    j=0

    while i<len(p1) and j<len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            result_docs.append(p1[i].doc_id)
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            i += 1
        else:
            j += 1
            
    return result_docs

def search_merge_or(inv_index, term1, term2):
    p1 = inv_index.get(term1,[])
    p2 = inv_index.get(term2,[])
    result_docs = []
    i=0
    j=0
    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            result_docs.append(p1[i].doc_id)
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            result_docs.append(p1[i].doc_id)
            i += 1
        else:
            result_docs.append(p2[j].doc_id)
            j += 1
            
    while i < len(p1):
        result_docs.append(p1[i].doc_id)
        i += 1
    while j < len(p2):
        result_docs.append(p2[j].doc_id)
        j += 1
        
    return result_docs

def search_set_or(inv_index, term1, term2):
    set1 = {p.doc_id for p in inv_index.get(term1, [])}
    set2 = {p.doc_id for p in inv_index.get(term2, [])}
    return list(set1 | set2)

def search_merge_not(inv_index, term1, term2):
    p1 = inv_index.get(term1, [])
    p2 = inv_index.get(term2, [])
    
    result_docs = []
    i = 0
    j = 0
    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            result_docs.append(p1[i].doc_id)
            i += 1
        else:
            j += 1
            
    while i < len(p1):
        result_docs.append(p1[i].doc_id)
        i += 1
    return result_docs

def search_set_not(inv_index, term1, term2):
    set1 = {p.doc_id for p in inv_index.get(term1, [])}
    set2 = {p.doc_id for p in inv_index.get(term2, [])}
    return list(set1 - set2)
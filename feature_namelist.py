import os

def load_namelist(filenames, MANUAL_MUSIC_DIR, skip_first_row=False):
    namelist_feature_dict = {
        "entryMap": {} ,
        "words": set(),
        "lengths": set(),
        "pos": -1
    }
    for filename in filenames :
        if skip_first_row:
            label = open(os.path.join(MANUAL_MUSIC_DIR, filename)).readline().split("@@@")[1].strip()
            lines = open(os.path.join(MANUAL_MUSIC_DIR, filename), encoding="utf8").readlines()[1:]
        else:
            lines = open(os.path.join(MANUAL_MUSIC_DIR, filename), encoding="utf8").readlines()
        for line in lines:
            line = line.strip().split("\t")
            if line[0] == "": continue
            if not skip_first_row: label = line[1]
            tokens = line[0].lower().split(" ")
            namelist_feature_dict["lengths"].add(len(tokens))
            cleaned = "".join(tokens)
            if cleaned in namelist_feature_dict["entryMap"]:
                namelist_feature_dict["entryMap"][cleaned].add(label)
            else:
                namelist_feature_dict["entryMap"][cleaned] = set([label])
            for token in tokens:
                namelist_feature_dict["words"].add(token)
    return namelist_feature_dict

def get_matched_token_tags(nameListFeature_dict: list, tokens: list, start: int, end: int):
    for fromIndex in range(start, end):
        for toIndex in range(end, start, -1):
            length = toIndex - fromIndex
            if length not in nameListFeature_dict["lengths"]:
                continue
            sub = tokens[fromIndex:toIndex]
            words = "".join([tok.lower() for tok in sub])
            if words in nameListFeature_dict["entryMap"]:
                tags = nameListFeature_dict["entryMap"][words]
                for idx, token in enumerate(sub):
                    yield length, idx+fromIndex, token, list(tags), idx
                    ### each token in match return (length, idx+fromIndex, token, list(tags), idx)

def get_matches(nameListFeature_dict: dict, tokens: list):
    valid = range(len(tokens))
    index = 0
    size = len(valid)
    while index < size:
        value = valid[index]
        start = index
        end = start + 1
        while end < size:
            v = valid[end]
            if v == value+1:
                end += 1
                value += 1
            else:
                break
        startIndex = valid[start]
        endIndex = valid[end-1] + 1
        yield list(get_matched_token_tags(nameListFeature_dict, tokens, startIndex, endIndex))
        index = end

def get_namelist_match_idx(nameListFeature_dict: dict, instance_tokens_lowered: list):
    """Given a dictionary and a list of lowercase tokens, return a dictionary with token id and fields (length, label)"""
    namelist_match_idx = {}
    matches = list(get_matches(nameListFeature_dict, instance_tokens_lowered))
    for match in matches:
        if match:
            for length, tok_id, _, labels, idx in match:
                if tok_id not in namelist_match_idx: 
                    namelist_match_idx[tok_id] = {
                        "length": length,
                        "labels": labels,
                        "pos": idx
                    }
                else:
                    if length > namelist_match_idx[tok_id]["length"]:
                        namelist_match_idx[tok_id] = {
                            "length": length, 
                            "labels": labels,
                            "pos": idx
                        }
                    elif length == namelist_match_idx[tok_id]["length"]:
                        namelist_match_idx[tok_id] = {
                            "length": length,
                            "labels": namelist_match_idx[tok_id]["labels"] + labels,
                            "pos": idx
                        }
    return namelist_match_idx
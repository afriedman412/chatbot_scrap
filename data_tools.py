import json
import regex as re
import unicodedata
from config import MAX_LENGTH
from voc import Voc

def load_lines_and_convos(file_name):
    lines, convos = {}, {}
    with open(file_name, "r", encoding="iso-8859-1") as f:
        for line in f:
            # extract for line object
            j = json.loads(line)
            line_ = {}
            line_['line_id'] = j['id']
            line_['char_id'] = j['speaker']
            line_['text'] = j['text']
            lines[j['id']] = line_

            # extract for convo object
            if j['conversation_id'] not in convos:
                conv_ = {}
                conv_['conversation_id'] = j['conversation_id']
                conv_['movie_id'] = j['meta']['movie_id']
                conv_['lines'] = [line_]
            else:
                conv_ = convos[j['conversation_id']]
                conv_['lines'].insert(0, line_)
            convos[j['conversation_id']] = conv_
    return lines, convos

def extract_sentence_pairs(convos):
    # extract pairs of sentences
    qa_pairs = []
    for c in convos.values():
        for i in range(len(c['lines']) -1):
            in_, out_ = tuple(
                [c['lines'][i_]['text'].strip() for i_ in (i, i+1)])
            
            if in_ and out_:
                qa_pairs.append([in_, out_])
    return qa_pairs

def unicode_to_ascii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )

def normalize_str(s):
    s = unicode_to_ascii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def read_vocs(datafile, corpus_name):
    print('reading lines...')
    lines = open(datafile, encoding="utf-8").read().strip().split("\n")
    pairs = [[normalize_str(s) for s in l.split("\t")] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

def filter_pair(p):
    # you tweaked this code
    return all([len(p_.split(' ')) < MAX_LENGTH for p_ in p])

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def load_prepare_data(corpus_name, datafile):
    print("prepping training data...")
    voc, pairs = read_vocs(datafile, corpus_name)

    print(f"reading {len(pairs)} sentence pairs...")
    pairs = filter_pairs(pairs)

    print(f"trimmed to {len(pairs)} sentence pairs...")
    print("counting words...")
    for pair in pairs:
        voc.add_sentences(pair)

    print("counted words:", len(voc.index2word))
    return voc, pairs
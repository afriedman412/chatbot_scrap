import os
from convokit import Corpus, download
import codecs
import csv
from data_tools import (
    load_lines_and_convos, 
    extract_sentence_pairs, 
    unicode_to_ascii, 
    normalize_str,
    read_vocs, 
    filter_pair,
    filter_pairs,
    load_prepare_data
)

corpus_name = "movie-corpus"
corpus = os.path.join("data", corpus_name)
# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Load lines and conversations
lines, conversations = load_lines_and_convos(os.path.join(corpus, "utterances.jsonl"))

# Write new csv file
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extract_sentence_pairs(conversations):
        writer.writerow(pair)
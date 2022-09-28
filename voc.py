import torch
from typing import Union, Literal
import itertools
from config import PAD_token, SOS_token, EOS_token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.intialize_dicts()
    
    def intialize_dicts(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            PAD_token: "PAD",
            SOS_token: "SOS",
            EOS_token: "EOS"
        }

    def add_sentences(self, input_: Union[str, list]):
        if isinstance(input_, str):
            input_ = [input_]
        for sentence in input_:
            for word in sentence.split(" "):
                self.add_word(word)

    def add_word(self, word):
        # you adapted this to use i, not num_words
        if word not in self.word2index:
            i = len(self.index2word)
            self.word2index[word] = i
            self.word2count[word] = 1
            self.index2word[i] = word
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        else:
            keep_words = [
                k for k,v in self.word2count.items() if v >= min_count
            ]

            # reinitialize dictos
            self.intialize_dicts()
            for word in keep_words:
                self.add_word(word)

            self.trimmed = True

    def trim_pairs(self, pairs):
        keep_pairs = []
        if not self.trimmed:
            raise Exception("vocab not trimmed!")
        
        else:
            keep_pairs = []
            for pair in pairs:
                keep_input = True
                keep_output = True
                for word in pair[0].split():
                    if word not in self.word2index:
                        keep_input = False
                        break

                if keep_input and keep_output:
                    keep_pairs.append(pair)
        print(f"trimmed from {len(pairs)} to {len(keep_pairs)}")
        return keep_pairs

    def index_from_sentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(" ")] + [EOS_token]

    def zero_padding(self, l):
        return list(itertools.zip_longest(*l, fillvalue=PAD_token))
    
    def bin_matrix(self, l):
        m = [[
            0 if token == PAD_token else 1 for token in seq
            ] for seq in l]
        return m

    def var_prep(self, l, type=Literal["input", "output"]):
        # replaces input_var and output_var
        batch = [self.index_from_sentence(sentence) for sentence in l]
        lengths = [len(index) for index in batch]
        pad_list = self.zero_padding(batch)
        pad_var = torch.LongTensor(pad_list)
        
        if type=="output":
            max_target_len = max(lengths)
            mask = self.bin_matrix(pad_list)
            mask = torch.BoolTensor(mask)
            return pad_var, mask, max_target_len
        
        else:
            return pad_var, lengths

    def batch2train(self, pair_batch):
        pair_batch.sort(key=lambda x: len(x[0].split()), reverse=True)
        input_batch, output_batch = [], []
        for pair in pair_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        
        input_, lengths = self.var_prep(input_batch, "input")
        output_, mask, max_target_len = self.var_prep(output_batch, "output")
        return input_, lengths, output_, mask, max_target_len
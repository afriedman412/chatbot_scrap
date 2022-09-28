import os
import torch

model_name = 'cb_model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

corpus_name = "movie-corpus"
corpus = os.path.join("data", corpus_name)
MAX_LENGTH = 10
MIN_WORDS = 3

PAD_token = 0 # for padding
SOS_token = 1 # start of sentence
EOS_token = 2 # end of sentence

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iters = 4000
print_every = 1
save_every = 500

checkpoint = ""
checkpoint_iter = 0

def path_maker(
    save_dir,
    model_name,
    corpus_name,
    encoder_n_layers, 
    decoder_n_layers, 
    hidden_size,
    checkpoint):
    return os.path.join(
        save_dir, model_name, corpus_name, 
        f"{encoder_n_layers}-{decoder_n_layers}_{hidden_size}", f"{checkpoint}.tar")

def file_loader(load_file_name, diff_machine=False):
    # if loading on diff machine
    if diff_machine:
        checkpoint = torch.load(load_file_name, map_loaction=torch.device('cpu'))
    # if on training machine
    else:
        checkpoint = torch.load(load_file_name)
    return checkpoint

    
    
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']
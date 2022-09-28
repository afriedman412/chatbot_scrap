import torch
import torch.nn as nn
import random
import os
from voc import Voc
from data_tools import normalize_str
from config import device, SOS_token, checkpoint


def maskNLLLoss(input, target, mask):
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(input, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, n_total.item()

def train(
    input_var, lengths, target_var, mask, 
    max_target_len, encoder, decoder, embedding, 
    encoder_optimizer, decoder_optimizer,
    batch_size, clip, teacher_forcing_ratio, max_length, 
    ):

    # zero grads
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # set device options
    input_var = input_var.to(device)
    target_var = target_var.to(device)
    mask = mask.to(device)

    # lengths for rnn packing always on cpu!
    lengths = lengths.to('cpu')

    # initialize vars
    loss = 0
    print_losses = []
    n_totalengths = 0

    # forward pass thru encoder
    encoder_outputs, encoder_hidden =encoder(input_var, lengths)

    # create initial decoder input (starting w SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # determinte if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        if use_teacher_forcing:
            # teacher forcing: next input is current target
            decoder_input = target_var[t].view(1, -1)

        else:
            # no teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

        # calculate and accumulate loss
        mask_loss, n_total = maskNLLLoss(decoder_output, target_var[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * n_total)
        n_totals += n_total

    # backprop
    loss.backward()

    # clip gradients: modify gradients in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses)/n_totals

def train_iters(model_name, voc: Voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, hidden_size, save_dir, n_iters, batch_size, print_every, save_every, clip, corpus_name, load_file_name=None):

    # load batches for each iteration
    tr_batches = [
        voc.batch2train(
            [random.choice(pairs) for _ in range(batch_size)]
            ) for _ in range(n_iters)
            ]

    # initialize
    print("initializing...")
    iter = 1
    print_loss = 0
    if load_file_name:
        start_iteration = checkpoint['iteration'] + 1

    # training loop
    print("training...")
    for i in range(start_iteration, n_iters+1):
        batch = tr_batches[i-1]
        # extract fields
        input_var, lengths, target_var, mask, max_target_len = batch

        # run training loop w batch
        loss = train(
            input_var, lengths, target_var, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip
        )
        print_loss += loss

        # print progress
        if i % print_every == 0:
            print_loss_avg = print_loss / print_every
            print(f"iteration: {i} // % complete: {i / n_iters * 100} // avg loss: {print_loss_avg}")
            print_loss = 0

        # save checkpoint
        if i % save_every == 0:
            dir_ = os.path.join(save_dir, model_name, corpus_name, f"{encoder_n_layers}-{decoder_n_layers}_{hidden_size}")

            if not os.path.exists(dir_):
                os.makedirs(dir_)
            torch.save(
                dict(zip(
                    ['iteration', 'en', 'de', 'en_opt', 'de_opt', 'loss', 'voc_dict', 'embedding'],
                    [encoder.state_dict(), decoder.state_dict(), encoder_optimizer.state_dict(), decoder_optimizer.state_dict(), loss, voc.__dict__, embedding.state_dict()]
                )), os.path.join(dir_, f"{i}_checkpoint.tar")
            )

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_len, max_len):
        # forward inputs through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_len)
        # prep encoder final hidden layer to feed into decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # initialize decoder output with SOS token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # intialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        for _ in range(max_len):
            # forward pass thru decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # get most likely word token and softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores))
            # prep current token to be next decoder input (ie add a dim)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        
        return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, voc: Voc, sentence, max_length):
    # format batch
    # words -> indexes (check if this needs to be in another list)
    indexes_batch = [voc.index_from_sentence(sentence)]
    # create lengths tensor
    lengths = torch.tensor([len(i) for i in indexes_batch])
    # transpose dims of batch to match model expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0,1)
    # use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to('cpu')
    # decode sentence w searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2[token.item()] for token in tokens]
    return decoded_words

def evaluate_input(encoder, decoder, searcher, voc):
    input_sentence = ''
    while True:
        try:
            # get input sentence
            input_sentence = input("> ")
            # check for quit
            if input_sentence in ['q', 'quit']:
                break
            # normalize sentence
            input_sentence = normalize_str(input_sentence)
            # evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # format and print response sentence
            output_words[:] = [x for x in output_words if not (x=='EOS' or x=='PAD')]
            print("Bot:", ' '.join(output_words))
        except KeyError:
            print("Error: encountered unknown word")
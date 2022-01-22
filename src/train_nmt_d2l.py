from torch import optim
from read_data_nmt import load_data, read_val_data
from models import S2SAttentionDecoder, S2SEncoder, S2SDecoder, TransformerEncoderDecoder, TransformerEncoder, TransformerDecoder
from loss import MaskedCELoss
import torch
import torch.nn as nn
import collections
import math
from read_data_nmt import truncate_pad, load_val_data
from nltk.translate.bleu_score import corpus_bleu
import nltk
from tqdm import tqdm
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_sentence(model, sentence, src_vocab, tgt_vocab, max_len, device):

    model.eval()
    # get the src tokens in indexed form.
    src_tokens = src_vocab[nltk.tokenize.word_tokenize(sentence.lower())] + [src_vocab["<eos>"]]
    # Get the lengths of the input sentence
    x_len = torch.tensor([len(src_tokens)], device=device)
    # truncate and pad the input sentence
    src_tokens = truncate_pad(src_tokens, max_len, src_vocab["<pad>"])
    # Creating a batch from a single input sentence.
    x = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_output = model.encoder(x, x_len)    # (B, max_len, hidden_dim)
    state = model.decoder.init_state(enc_output, x_len)
    y = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq = []
    # Generate the output tokens one by one
    for i in range(max_len):
        output, state = model.decoder(y, state) # (B, 1, tgt_vocab_size)
        y = output.argmax(dim=2)
        # Get the predicted token label.
        y_pred = y.squeeze(dim=0).type(torch.int32).item()
        # quit if end of sentence is reached.
        if y_pred == tgt_vocab["<eos>"]:
            break
        output_seq.append(y_pred)

    # Converting the output generated sequence to words using target language vocabulary.
    return " ".join([tgt_vocab.idx2word[item] for item in output_seq])



def test_bleu(model, src_vocab, tgt_vocab, len_sequence, device, sentences_preprocessed, true_trans_preprocessed):

    predictions = []
    for sentence in tqdm(sentences_preprocessed):
        sentence_predicted = predict_sentence(model, sentence, src_vocab, tgt_vocab, len_sequence, device)
        predictions.append(sentence_predicted)

    references = [[nltk.tokenize.word_tokenize(sent.lower())] for sent in true_trans_preprocessed]
    candidates = [nltk.tokenize.word_tokenize(sent.lower()) for sent in predictions]
    score = corpus_bleu(references, candidates)
    print(score)
    return score


def validate(model, val_iter, src_vocab, tgt_vocab, loss_function):

    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_iter):
            x, x_len, y, y_len = [item.to(device) for item in batch]
            bos_token = torch.tensor([tgt_vocab["<bos>"]] * y.shape[0], device=device).reshape(-1, 1)
            # Append beginning of sentence (BOS) to the input to the decoder so that input to the decoder is shifted by one to the right.
            decoder_input = torch.cat([bos_token, y[:, :-1]], 1)
            output_model, state = model(x, decoder_input, x_len)
            # Passing the output of the model to the loss function.
            l = loss_function(output_model, y, y_len)
            running_loss += l.sum().item()
    return running_loss / len(val_iter.dataset)

def train_model(model, data_loader, learning_rate, n_epochs, tgt_vocab, src_vocab, device):
    val_loss_list = []
    train_loss_list = []
    bleu_score_list = []

    # Masked Cross Entropy loss function.
    loss_function = MaskedCELoss()

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    
    model.apply(xavier_init_weights)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Putting the model to train mode.
    model.train()

    sentences_preprocessed, true_trans_preprocessed = read_val_data(data_name="php")
    val_iter = load_val_data(128, len_sequence, src_vocab, tgt_vocab)
    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        for batch_idx, batch in enumerate(data_loader):
            optimizer.zero_grad()
            x, x_len, y, y_len = [item.to(device) for item in batch]
            bos_token = torch.tensor([tgt_vocab["<bos>"]] * y.shape[0], device=device).reshape(-1, 1)
            # Append beginning of sentence (BOS) to the input to the decoder so that input to the decoder is shifted by one to the right.
            decoder_input = torch.cat([bos_token, y[:, :-1]], 1)
            output_model, state = model(x, decoder_input, x_len)
            # Passing the output of the model to the loss function.
            l = loss_function(output_model, y, y_len)
            # Backpropagate the loss
            l.sum().backward() 
            # Do gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                running_loss += l.sum().item()

        val_loss = validate(model, val_iter, src_vocab, tgt_vocab, loss_function)
        train_loss = running_loss / len(data_loader.dataset)
        val_loss_list.append(val_loss)
        train_loss_list.append(train_loss)
        # bleu_score_list

        score = test_bleu(model, src_vocab, tgt_vocab, len_sequence, device, sentences_preprocessed, true_trans_preprocessed)
        bleu_score_list.append(score)
        print(predict_sentence(model, "A person crosses the street and avoids the spill of paint." , src_vocab, tgt_vocab, len_sequence, device))
        print(f"Epoch_Loss, {epoch}, {running_loss / len(data_loader.dataset)}")
        print("val loss", val_loss)

        # Save model every 19th epoch.
        if epoch % 20 == 19:
            PATH = "model_att.pt"
            torch.save(model.state_dict(), PATH)
            with open("val_loss", "wb") as fp:
                pickle.dump(val_loss_list, fp)
            with open("train_loss", "wb") as fp:
                pickle.dump(train_loss_list, fp)
            with open("blue_score", "wb") as fp:
                pickle.dump(bleu_score_list, fp)


# embedding_size = 100
# hidden_size = 200
# num_layers = 1
batch_size = 128
len_sequence = 20
lr = 0.0001
n_epochs = 80
print(n_epochs, lr, len_sequence)

data_iter, src_vocab, tgt_vocab = load_data(batch_size, len_sequence)
print(len(src_vocab))
print(len(tgt_vocab))
# encoder = S2SEncoder(len(src_vocab), embedding_size, hidden_size, num_layers)
# decoder = S2SAttentionDecoder(len(tgt_vocab), embedding_size, hidden_size, num_layers)
# model = S2SEncoderDecoder(encoder, decoder)
ss = 256
hs = ss
encoder = TransformerEncoder(
    query=ss, key=ss, value=ss, hidden_size=ss, num_head=4, dropout=0.1, lnorm_size=[ss], ffn_input=ss, ffn_hidden=hs, vocab_size=len(src_vocab), num_layers = 3
)
decoder = TransformerDecoder(
    query=ss, key=ss, value=ss, hidden_size=ss, num_head=4, dropout=0.1, lnorm_size=[ss], ffn_input=ss, ffn_hidden=hs, vocab_size=len(tgt_vocab), num_layers = 3
)
print("4 layers, 128 size")
model = TransformerEncoderDecoder(encoder, decoder)
# train_model(model, data_iter, lr, n_epochs, tgt_vocab, src_vocab, device)
PATH = "model_att.pt"
# torch.save(model.state_dict(), PATH)

model.load_state_dict(torch.load(PATH, map_location=device))


# sentences_preprocessed, true_trans_preprocessed = read_test_data(data_name="php")
# test_bleu(model, src_vocab, tgt_vocab, len_sequence, device, sentences_preprocessed, true_trans_preprocessed)


# sentences = ["PHP Manual", "Returns the name of the field corresponding to field_number.", "Home"]

sentences = ["The man is walking his dog"]
sentences_preprocessed = [sentence for sentence in sentences]
true_trans = ["PHP Handbuch", "Gibt den Namen des Feldes, das field_number entspricht, zurück.", "Zum Anfang"]
true_trans_preprocessed = [trans for trans in true_trans]
predictions = []
for sentence in sentences_preprocessed:
    sentence_predicted = predict_sentence(model, sentence, src_vocab, tgt_vocab, len_sequence, device)
    print(sentence_predicted)
    predictions.append(sentence_predicted)
print("abc")
# references = [[nltk.tokenize.word_tokenize(sent.lower())] for sent in true_trans_preprocessed]
# candidates = [nltk.tokenize.word_tokenize(sent.lower()) for sent in predictions]
# score = corpus_bleu(references, candidates)
# print(score)
# print(sentences_preprocessed, true_trans_preprocessed)
# print(predictions)

# sentences_preprocessed, true_trans_preprocessed = read_test_data(data_name="php")
# test_bleu(model, src_vocab, tgt_vocab, len_sequence, device, sentences_preprocessed, true_trans_preprocessed)

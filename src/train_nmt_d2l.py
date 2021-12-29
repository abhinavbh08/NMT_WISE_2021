from torch import optim
from read_data_nmt import load_data, read_test_data
from models import S2SAttentionDecoder, S2SEncoder, S2SDecoder, S2SEncoderDecoder, TransformerEncoder, TransformerDecoder
from loss import MaskedCELoss
import torch
import torch.nn as nn
import collections
import math
from read_data_nmt import truncate_pad
from nltk.translate.bleu_score import corpus_bleu
import nltk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_sentence(model, sentence, src_vocab, tgt_vocab, num_steps, device):

    model.eval()
    # src_tokens = src_vocab[sentence.lower().split(" ")] + [src_vocab["<eos>"]]
    src_tokens = src_vocab[nltk.tokenize.word_tokenize(sentence.lower())] + [src_vocab["<eos>"]]
    x_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab["<pad>"])
    x = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_output = model.encoder(x, x_len)
    state = model.decoder.init_state(enc_output, x_len)
    y = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for i in range(num_steps):
        output, state = model.decoder(y, state)
        y = output.argmax(dim=2)
        y_pred = y.squeeze(dim=0).type(torch.int32).item()
        if y_pred == tgt_vocab["<eos>"]:
            break
        output_seq.append(y_pred)

    return " ".join([tgt_vocab.idx2word[item] for item in output_seq])

def test_bleu(model, src_vocab, tgt_vocab, len_sequence, device, sentences_preprocessed, true_trans_preprocessed):

    predictions = []
    for sentence in sentences_preprocessed:
        sentence_predicted = predict_sentence(model, sentence, src_vocab, tgt_vocab, len_sequence, device)
        predictions.append(sentence_predicted)

    references = [[nltk.tokenize.word_tokenize(sent.lower())] for sent in true_trans_preprocessed]
    candidates = [nltk.tokenize.word_tokenize(sent.lower()) for sent in predictions]
    score = corpus_bleu(references, candidates)
    print(score)

def train_model(model, data_iter, lr, n_epochs, tgt_vocab, src_vocab,device):
    loss_function = MaskedCELoss()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    sentences_preprocessed, true_trans_preprocessed = read_test_data(data_name="php")
    for epoch in range(n_epochs):
        running_loss = 0.0
        model.train()
        for batch_idx, batch in enumerate(data_iter):
            optimizer.zero_grad()
            x, x_len, y, y_len = [item.to(device) for item in batch]
            bos = torch.tensor([tgt_vocab["<bos>"]] * y.shape[0], device=device).reshape(-1, 1)
            decoder_input = torch.cat([bos, y[:, :-1]], 1)
            output, state = model(x, decoder_input, x_len)
            l = loss_function(output, y, y_len)
            l.sum().backward()    
            optimizer.step()
            with torch.no_grad():
                running_loss += l.sum().item()
                batch_loss = l.sum().item() / x.size(0)

            # print(epoch, batch_idx, batch_loss)
        test_bleu(model, src_vocab, tgt_vocab, len_sequence, device, sentences_preprocessed, true_trans_preprocessed)
        print(f"Epoch_Loss, {epoch}, {running_loss / len(data_iter.dataset)}")


# embedding_size = 100
# hidden_size = 200
# num_layers = 1
batch_size = 64
len_sequence = 20
lr = 0.0008
n_epochs = 100
print(n_epochs)

data_iter, src_vocab, tgt_vocab = load_data(batch_size, len_sequence)
print(len(src_vocab))
print(len(tgt_vocab))
# encoder = S2SEncoder(len(src_vocab), embedding_size, hidden_size, num_layers)
# decoder = S2SAttentionDecoder(len(tgt_vocab), embedding_size, hidden_size, num_layers)
# model = S2SEncoderDecoder(encoder, decoder)
encoder = TransformerEncoder(
    query=64, key=64, value=64, hidden_size=64, num_head=4, dropout=0.1, norm_shape=[64], ffn_input=64, ffn_hidden=128, vocab_size=len(src_vocab), num_layers = 4
)
decoder = TransformerDecoder(
    query=64, key=64, value=64, hidden_size=64, num_head=4, dropout=0.1, norm_shape=[64], ffn_input=64, ffn_hidden=128, vocab_size=len(tgt_vocab), num_layers = 4
)
print("4 layers, 64 size")
model = S2SEncoderDecoder(encoder, decoder)
train_model(model, data_iter, lr, n_epochs, tgt_vocab, src_vocab, device)
PATH = "model_att.pt"
torch.save(model.state_dict(), PATH)

# model.load_state_dict(torch.load(PATH, map_location=device))


# sentences_preprocessed, true_trans_preprocessed = read_test_data(data_name="php")
# test_bleu(model, src_vocab, tgt_vocab, len_sequence, device, sentences_preprocessed, true_trans_preprocessed)


# sentences = ["PHP Manual", "Returns the name of the field corresponding to field_number.", "Home"]
# sentences_preprocessed = [sentence for sentence in sentences]
# true_trans = ["PHP Handbuch", "Gibt den Namen des Feldes, das field_number entspricht, zurück.", "Zum Anfang"]
# true_trans_preprocessed = [trans for trans in true_trans]
# predictions = []
# for sentence in sentences_preprocessed:
#     sentence_predicted = predict_sentence(model, sentence, src_vocab, tgt_vocab, len_sequence, device)
#     predictions.append(sentence_predicted)
# print("abc")
# references = [[nltk.tokenize.word_tokenize(sent.lower())] for sent in true_trans_preprocessed]
# candidates = [nltk.tokenize.word_tokenize(sent.lower()) for sent in predictions]
# score = corpus_bleu(references, candidates)
# print(score)
# print(sentences_preprocessed, true_trans_preprocessed)
# print(predictions)

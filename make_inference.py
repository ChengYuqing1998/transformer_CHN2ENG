import nltk
import numpy as np
from torch.utils.data import DataLoader
import torch
from transformer import *
from translator import *
from tqdm import tqdm
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('./models/c2e_transformer_[0526-test1].pt', map_location=device)


PAD_token = 0
BOS_token = 1
EOS_token = 2
UNK_token = 3


def indexes_from_sentence(lang, sentence):
    enc_list = []
    if lang.name == 'cn':
        for word in sentence:
            try:
                enc_list.append(lang.word2index[word])
            except:
                enc_list.append(UNK_token)
        return enc_list
    else:
        for word in sentence.split(' '):
            try:
                enc_list.append(lang.word2index[word])
            except:
                enc_list.append(UNK_token)
        return enc_list


def variable_from_sentence(lang, sentence, device):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = torch.LongTensor(indexes).view(-1, 1)
    var = var.to(device)
    return var


with open('./input_lang.pkl', 'rb') as f:
    input_lang = pickle.load(f)
with open('./output_lang.pkl', 'rb') as f:
    output_lang = pickle.load(f)


count = 1
limit = 10  # it allows you try 10 sentences till it'll be closed
translator = Translator(model, 5, model.max_len, BOS_token, EOS_token, device=torch.device('cuda'))
while True:
    if count <= limit:
        sentence = input("please enter Chinese sentence: ").strip()
        enc_inputs = variable_from_sentence(input_lang, sentence, device).view(1, -1)
        enc_input_list = list(*enc_inputs.cpu().numpy())
        enc_input_list.pop(-1)
        enc_input_list.append(EOS_token)
        enc_inputs = [BOS_token]
        enc_inputs.extend(enc_input_list)
        enc_inputs= torch.tensor(enc_inputs, dtype=torch.long, device=device).reshape(1, -1)
        pred_seqs = translator.translate(enc_inputs)
        output_sentence = [' '.join([output_lang.index2word[_] for _ in pred_seq ]) for pred_seq in pred_seqs]
        print(output_sentence[0])
        count += 1
    else:
        break








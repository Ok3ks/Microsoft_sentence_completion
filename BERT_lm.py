import torch 
import re
import torch.nn as nn
import operator

from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch 
import numpy as np

#OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
from nltk import word_tokenize as tokenize
from gensim.models import KeyedVectors
import numpy as np
import os

res_dir = "drive/My Drive/ML/Resources/"
embed = res_dir + "/GoogleNewsvectorsnegative300.bin"
vec = KeyedVectors.load_word2vec_format(embed, binary = True )

#Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
Bert_masked_model = BertForMaskedLM.from_pretrained("bert-base-uncased")

def comp_accuracy(alist1,alist2):
    """Takes in two lists"""
    incorrect = 0
    if isinstance(alist1, list) and isinstance(alist2, list):
        if len(alist1) == len(alist2):
            for goldword,word in zip(alist1,alist2):
                if goldword == word:
                    pass
                else: incorrect + 1
        else: print("Lists must be of equal length")

    else: print("function takes in two lists")
    accuracy = ((len(alist1) - incorrect)/len(alist1))*100
    return accuracy

def make_segment_ids(list_of_tokens):
    #this function assumes that up to and including the first '[SEP]' is the first segment, anything afterwards is the second segment
    current_id=0
    segment_ids=[]
    for token in list_of_tokens:
        segment_ids.append(current_id)
        if token == '[SEP]':
            current_id +=1
    return segment_ids


def similarity(vec_A,vecB):
    return (np.dot(vec_A,vecB)/np.sqrt((np.dot(vec_A,vec_A)+ np.dot(vecB,vecB))))


def predict_word(tok, model):
    """Takes in a list of tokenized corpus,tok, and model used for prediction"""
    masked_words = []; predicted_words = []
    for index,token in enumerate(tok):
        hidden_index = index
        tok[hidden_index] = '[MASK]' ; masked_words.append(token)
        indexed_tokens_n = tokenizer.convert_tokens_to_ids(tok) 
        segmented_ids = make_segment_ids(tok)

        token_tensor_n = torch.tensor([indexed_tokens_n])
        segments_tensors_n = torch.tensor([segmented_ids])

        #ONly forward gradient 
        with torch.no_grad():
            outputs = model(token_tensor_n, token_type_ids = segments_tensors_n)
            predictions = outputs[0]


        predicted_index = torch.argmax(predictions[0, hidden_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        predicted_words.append(predicted_token)
    print("The accuracy of the Model - {} is {}".format(str(model)[0:10],comp_accuracy(masked_words, predicted_words)))



def make_segment_ids(list_of_tokens):
    #this function assumes that up to and including the first '[SEP]' is the first segment, anything afterwards is the second segment
    current_id=0
    segment_ids=[]
    for token in list_of_tokens:
        segment_ids.append(current_id)
        if token == '[SEP]':
            current_id +=1
    return segment_ids

def preprocess_BERT(words):
    print(words)
    start_index = words.index("__START") 
    hidden_index = words.index("_____")
    end_index = words.index("__END") 


    words[hidden_index] = "[MASK]"
    words[start_index] = "[CLS]"
    words[end_index] = "[SEP]"

    return words


def BERT_MASK(sentence, tokenizer = tokenizer, Bert_masked_model = Bert_masked_model):

    words = preprocess_BERT(sentence)
    #words = tokenizer.tokenize(new_q)

    token_dict = {}

    tokenid = tokenizer.convert_tokens_to_ids(words)
    segment_ids = make_segment_ids(words)

    token_tensor = torch.tensor([tokenid])
    seg_tensor = torch.tensor([segment_ids])

    hidden_index = words.index("[MASK]")   #selects index of masked word
    
    with torch.no_grad():
        output = Bert_masked_model(input_ids = token_tensor, token_type_ids = seg_tensor)

    #obtaining predictions from words
    predictions = output[0]
    predicted_index = torch.argmax(predictions[0, hidden_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    return predicted_token

def BERT_rep(words, option, tokenizer = tokenizer, model = model):
    #Outputs BERT representation of a sentence
    words = preprocess_BERT(words)
    #words = tokenizer.tokenize(sentence)

    token_dict = {}

    hidden_index = words.index("[MASK]")   #selects index of masked word
    words[hidden_index] = option

    tokenid = tokenizer.convert_tokens_to_ids(words)
    segment_ids = make_segment_ids(words)

    token_tensor = torch.tensor([tokenid])
    seg_tensor = torch.tensor([segment_ids])

    output = model(token_tensor, token_type_ids = seg_tensor, output_hidden_states = True)
    return output[2][-1]



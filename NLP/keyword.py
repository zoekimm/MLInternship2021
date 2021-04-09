import pickle
import numpy as np
import pandas as pd
import csv
import nltk
import os
from phonetics import metaphone
#from metaphone import doublemetaphone
import sentencepiece as spm

def convert(data):
    sentence_list = []
    for x in data:          
        word_list = [metaphone(i) for i in x['sentence'].split(' ')] #metaphone each word
        x['sentence'] = ' '.join(word_list) #join them back to a sentence
        sentence_list.append(x['sentence']) 
    return sentence_list
  
def smdtemplate(file_name):
    templates= '--input={} \
                --pad_id={} \
                --bos_id={} \
                --eos_id={} \
                --unk_id={} \
                --model_prefix={} \
                --vocab_size={} \
                --character_coverage={} \
                --model_type={}'
    train_input_file = os.path.join(file_name)
    pad_id=0 
    vocab_size = 890 
    prefix = file_name[:-12] + 'spm'
    bos_id=1 
    eos_id=2 
    unk_id=3 
    character_coverage = 1.0
    model_type ='unigram'
    max_sentence_length = 10000
    cmd = templates.format(train_input_file,
                    pad_id,
                    bos_id,
                    eos_id,
                    unk_id,
                    prefix,
                    vocab_size,
                    character_coverage,
                    model_type,
                    max_sentence_length)
    return cmd
  
def load_spm(file, file_list, file_name):
    #save it as a text file for sentencepiece
    with open(file_name, mode='wt', encoding='utf-8') as text1:
        text1.write('\n'.join(file_list))
    spm.SentencePieceTrainer.Train(smdtemplate(file_name))
    sp = spm.SentencePieceProcessor()
    sp.load(file_name[:-12] + 'spm' +'.model')
    encode(file, sp)
    return 
  
def encode(data, sp):
    for i in data:
        i['sentence'] = sp.encode_as_ids(i['sentence'])
    return 
  
def main():
    with open('voice.pickle', 'rb') as pickle2:
        voice_f = pickle.load(pickle2)
    with open('comments.pickle', 'rb') as pickle1:
        comments_f = pickle.load(pickle1)
        
    file = [voice_f, comments_f]
    file_list = []
    
    for i in [voice_f, comments_f]:
        file_list.append(convert(i))
        
    load_spm(file[0], file_list[0], 'voice_sentence.txt')
    load_spm(file[1], file_list[1], 'comments_sentence.txt')

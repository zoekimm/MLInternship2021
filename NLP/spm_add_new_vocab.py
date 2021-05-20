import pickle
import os
import sys
import sentencepiece as spm
from phonetics import metaphone
from collections import defaultdict
from sentencepiece import sentencepiece_model_pb2 as model

class spm_new_vocab:
  
    def __init__(self):
        self.f = self.get_sentence()
        self.sentence_list = self.convert()
        self.encoded_list = []
        
    def get_sentence(self):
        with open('voice_lstm_result_2578cnt_lhr_210405.pickle', 'rb') as pickle2:
            voice_lstm = pickle.load(pickle2)
            
        with open('comments_lstm_result_3000cnt_lhr_210405.pickle', 'rb') as pickle1:
            comments_lstm = pickle.load(pickle1) 
            
        f = []
        for i in voice_lstm:
            f.append(i['sentence'])
            
        for j in comments_lstm:
            f.append(j['sentence'])
            
        with open('sentence_compiled.txt', mode='wt', encoding='utf-8') as text1:
            text1.write('\n'.join(f))
            
        return f
      
    def convert(self):
        sentence_list = []
        for x in self.f:          
            word_list = [metaphone(i) for i in x.split(' ')] #metaphone each word
            x = ' '.join(word_list) #join them back to a sentence
            sentence_list.append(x) 
        return sentence_list
      
    def load_spm(self, file_name):
        # save it as a text file for sentencepiece
        with open(file_name, mode='wt', encoding='utf-8') as text1:
            text1.write('\n'.join(self.sentence_list))
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
        prefix = file_name[:-4] + '_spm'
        bos_id=1 
        eos_id=2 
        unk_id=3 
        character_coverage = 1.0
        model_type ='bpe'
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
        spm.SentencePieceTrainer.Train(cmd)
        sp = spm.SentencePieceProcessor()
        sp.load(file_name[:-4] + '_spm' +'.model')
        
    def get_encoded_list(self):
        s = spm.SentencePieceProcessor(model_file = 'sentence_spm.model')
        file1 = open('sentence_compiled.txt', 'r')
        
        for line in file1:
            for n in range(5):
                self.encoded_list.append(s.encode(metaphone(line), out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1))
                
        s_list = [item for sublist in self.encoded_list for item in sublist]
        self.encoded_list = list(set(s_list))
        
        with open('./maimovie_voice_fulldata_spm3.vocab', encoding='utf-8') as f:
            x = [doc.strip().split("\t") for doc in f]
            
        word2idx = {w[0]: i for i, w in enumerate(x)}
        
        for i in word2idx:
            if i in self.encoded_list:
                self.encoded_list.remove(i)
                
    def add_new_vocab(self, filename):
        m = model.ModelProto()
        m.ParseFromString(open("maimovie_voice_fulldata_spm3.model", "rb").read())
        
        for token in self.encoded_list:
            new_token = model.ModelProto().SentencePiece()
            new_token.piece = token
            m.pieces.append(new_token)
            
        with open(filename, 'wb') as f:
            f.write(m.SerializeToString())
            
    def execute(self):
        self.load_spm('sentence.txt')
        self.get_encoded_list()
        self.add_new_vocab('maimovie_voice_fulldata_spm3_modified.model')

import re
import nltk
import string
import pickle
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt  
from nltk.corpus import stopwords
from normalise import normalise

class LanguagePreprocessor():
    
    def __init__(self, lang, stop):
        self.text = '' 
        self.lang = lang
        self.stop = stop

    def get_text_eng(self, text):
        regex = re.compile(f'[{string.punctuation+string.ascii_letters+string.whitespace}]+')
        return ' '.join(regex.findall(text))
    
    def get_text_kor(self, text):
        regex = re.compile(f'[^{string.punctuation+"ㄱ-힣"+string.whitespace}]+')
        self.text = regex.sub(f'', self.text)
        self.text = re.sub(r'(.)\1\1+',r'\1\1', self.text)
        return text
    
    def normalize_text_eng(self):
        self.text = ' '.join(normalise(word_tokenize(self.text), verbose=False))
   
    def stemming_eng(self):
        lemmatizer = WordNetLemmatizer()
        self.text = ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(self.text)])
    
    def stemming_kor(self):
        okt=Okt() 
        self.text = ' '.join(okt.nouns(self.text))
         
    def remove_stopwords_eng(self):
        self.text = ' '.join([word.lower() for word in word_tokenize(self.text) if word not in set(stopwords.words('english'))])
        
    def remove_stopwords_kor(self):
        if(self.stop != None):
            open_file = open(self.stop, "rb")
            stopwords = pickle.load(open_file)  
            self.text = ' '.join([word.lower() for word in word_tokenize(self.text) if word not in set(stopwords)])
        
        
    def __call__(self, text):     
        if (self.lang == 'KOR'):
            self.text = self.get_text_kor(text)
            self.remove_stopwords_kor()
            self.stemming_kor()
        
        if (self.lang == 'ENG'):
            self.text = self.get_text_eng(text)
            self.normalize_text_eng()
            self.remove_stopwords_eng()
            self.stemming_eng()

        print(self.text)
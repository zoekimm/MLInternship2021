from collections import defaultdict
from tensorflow import keras

def load_text(filename, data):
    keyword_dict = defaultdict(lambda : len(keyword_dict))
    #write into a text file
    with open(filename, 'w') as outFile:
        for i in data:
            id_arr = ",".join(map(str,i['sentence'])) 
            keyword_arr = []
            for j in i['intended_keywords']:
                keyword_arr.append(keyword_dict[j])
            keyword_arr = ",".join(map(str, keyword_arr))
            line = id_arr + ' : ' + keyword_arr + '\n'
            outFile.write(line)
    d = tf.data.TextLineDataset(filename)
    return d
  
def splitLine(line):
    string =tf.strings.split(line,sep=':')
    x = tf.strings.split(string[0],sep=',')
    y = tf.strings.split(string[1],sep=',')
    return x,y  
  
def sentence_size(length):
    def _sentence_size(x, y):
        x = tf.strings.to_number(x, tf.int32)
        x = tf.pad(x,[[0,length-tf.shape(x)[0]]]) #1d 
        y = tf.strings.to_number(y, tf.int32)
        y = tf.pad(y,[[0,length-tf.shape(y)[0]]]) #1d 
        return (y,x), y
    return _sentence_size

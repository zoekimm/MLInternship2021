from collections import defaultdict
from tensorflow import keras

def load():
  for i in data:
      key_id = []
      i['sentence'] = ",".join(map(str,i['sentence'])) 
      for j in i['keytalks']:
          key_id.append(d[j])
      i['keytalks'] = ",".join(map(str,key_id)) 
  return

d = defaultdict(lambda : len(dict))
df = pd.DataFrame(data)

X = np.array(df['sentence'].tolist())
y = np.array(df['keytalks'].tolist())
all_ds = tf.data.Dataset.from_tensor_slices(((y,X),y)) 

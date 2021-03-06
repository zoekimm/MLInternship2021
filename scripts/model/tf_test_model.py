import numpy as np
from scipy.sparse import load_npz
import tensorflow as tf
import keras
import tensorflow as tf
from tensorflow import keras

class movie_tf:
  
    def __init__(self, filename):
        self.mat = load_npz(filename)
        self.arr = np.array(self.mat.todense())
        self.prob = tf.cast(tf.random.categorical(tf.math.log([[0.1, 0.9]]), self.arr.shape[1]), dtype = tf.float32)
        self.input = tf.math.add(self.get_input1(), self.get_input2())
        self.model = keras.Sequential()
        
    def get_input1(self):
        m = tf.zeros([self.arr.shape[0], self.arr.shape[1]], dtype = tf.dtypes.float32)
        m = tf.math.multiply(self.prob, self.arr)
        return m
      
    def get_input2(self):
        prob_m = 1 - self.prob
        rand = tf.random.uniform(shape=[self.arr.shape[0], self.arr.shape[1]], minval = 0, maxval = 1 , dtype = tf.float32)
        return tf.math.multiply(prob_m, rand)
      
    def build_model(self):
        self.model.add(keras.Input(shape = [self.arr.shape[0], self.arr.shape[1]]))
        #model.add(keras.layers.Dense(mat.shape[1]))
        self.model.add(keras.layers.Dense(256, activation = "relu"))
        self.model.add(keras.layers.Dense(256, activation = "relu"))
        self.model.add(keras.layers.Dense(self.mat.shape[1]))
        
    def __call__(self):
        #self.build_model()
        #size = int(self.arr.shape[0] * 0.7)
        #self.model.compile(optimizer='sgd', loss='mse')
        #self.model.fit(self.arr[:size], self.input[:size], batch_size=64, epochs=10)
        #results = self.model.evaluate(self.arr[size:], self.input[size:], batch_size=128)
        #print('test loss:', results)

        fig, ax = plt.subplots()
        plt.xlabel('epoch size')
        plt.ylabel('accuracy') 
        size = int(self.arr.shape[0] * 0.7)
        
        self.build_model()
        
        initial_learning_rate = 0.1
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        #self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss='mse')
  
        r_list = []
        
        for i in tqdm(input_list):
            self.model.fit(self.arr[:size], self.input[:size], batch_size = 64, epochs = i)
            
            result = self.model.evaluate(self.arr[size:], self.input[size:], batch_size = 64)
            r_list.append(result[1])
            print(result[1])
            
        plt.plot(input_list, r_list, color="blue", marker="o", ms=5, label="label")  
        plt.show()

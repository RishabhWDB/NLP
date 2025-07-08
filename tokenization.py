import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

sentences = ['I love my dog', 
             'I love my cat',
             'Peter ate a very fat cat',
             'Don\'t you think my dog is amazing']

tokenizer = Tokenizer(num_words=100, oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

#test_data = ['I really love my dog', 'my dog loves my manatee']

#test_seq = tokenizer.texts_to_sequences(test_data)
#print(test_seq)
padded = pad_sequences(sequences, padding = 'post')
print(word_index)
print(sequences)
print(padded)

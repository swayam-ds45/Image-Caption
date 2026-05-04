import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, Sequence

# Paths
BASE_DIR = r"c:\Users\swaya\OneDrive\Desktop\image caption"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FEATURES_FILE = os.path.join(DATASET_DIR, "features.pkl")
CAPTIONS_DICT_FILE = os.path.join(DATASET_DIR, "captions_dict.pkl")
TOKENIZER_FILE = os.path.join(DATASET_DIR, "tokenizer.pkl")
MAX_LEN_FILE = os.path.join(DATASET_DIR, "max_length.txt")
MODEL_OUTPUT = os.path.join(DATASET_DIR, "model.h5")

# Load data
with open(FEATURES_FILE, 'rb') as f:
    features = pickle.load(f)

with open(CAPTIONS_DICT_FILE, 'rb') as f:
    mapping = pickle.load(f)

with open(TOKENIZER_FILE, 'rb') as f:
    tokenizer = pickle.load(f)

with open(MAX_LEN_FILE, 'r') as f:
    max_length = int(f.read().strip())

vocab_size = len(tokenizer.word_index) + 1

# Train Test Split (80/20)
all_images = list(mapping.keys())
split = int(len(all_images) * 0.8)
train_images = all_images[:split]
test_images = all_images[split:]

# Custom Data Generator subclassing Sequence (Keras 3 compliant)
class CustomDataGenerator(Sequence):
    def __init__(self, data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
        self.data_keys = data_keys
        self.mapping = mapping
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data_keys) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_keys = self.data_keys[idx * self.batch_size : (idx + 1) * self.batch_size]
        X1, X2, y = list(), list(), list()
        
        for key in batch_keys:
            captions = self.mapping[key]
            # retrieve image feature
            feature = self.features[key][0]
            
            for caption in captions:
                # encode the sequence
                seq = self.tokenizer.texts_to_sequences([caption])[0]
                # split sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
                    
        return (np.array(X1), np.array(X2)), np.array(y)

# Define Model Architecture
# 1. Feature Extractor Model (Images)
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

# 2. Sequence Processor Model (Text)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# 3. Decoder Model (Combine both)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# Tie it together
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print(model.summary())

# Train Model
epochs = 40
batch_size = 64

print("Starting training...")
generator = CustomDataGenerator(train_images, mapping, features, tokenizer, max_length, vocab_size, batch_size)

# Callbacks to save the best model and stop early if it stops improving
checkpoint = ModelCheckpoint(MODEL_OUTPUT, monitor='loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True)

model.fit(generator, epochs=epochs, verbose=1, callbacks=[checkpoint, early_stopping])

# The absolute best model is already saved by the checkpoint callback, but we can do a final save just in case
# (it might overwrite the best if we don't check, but ModelCheckpoint with restore_best_weights=True guarantees 
# the model object has the best weights at the end)
model.save(MODEL_OUTPUT)
print(f"Model saved to {MODEL_OUTPUT}")

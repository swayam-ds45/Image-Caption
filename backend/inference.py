import sys
import os
import json

# Optionally suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Determine paths
BASE_DIR = r"c:\Users\swaya\OneDrive\Desktop\image caption"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_FILE = os.path.join(DATASET_DIR, "model.h5")
TOKENIZER_FILE = os.path.join(DATASET_DIR, "tokenizer.pkl")
MAX_LEN_FILE = os.path.join(DATASET_DIR, "max_length.txt")

# Check if model exists
model_ready = os.path.exists(MODEL_FILE)

if model_ready:
    import pickle
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
    from tensorflow.keras.applications.xception import Xception, preprocess_input
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Load requirements
    try:
        with open(TOKENIZER_FILE, 'rb') as f:
            tokenizer = pickle.load(f)
            
        with open(MAX_LEN_FILE, 'r') as f:
            max_length = int(f.read().strip())
            
        # Create word mappings
        word_to_index = tokenizer.word_index
        index_to_word = {index: word for word, index in word_to_index.items()}
        vocab_size = len(word_to_index) + 1

        # Recreate the model architecture
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.4)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)

        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.4)(se1)
        se3 = LSTM(256)(se2)

        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        # Load weights into the model
        model.load_weights(MODEL_FILE)

        # Load Xception for feature extraction
        base_model = Xception(weights='imagenet')
        feature_extractor = tf.keras.models.Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
        
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        model_ready = False

def extract_features(filename):
    image = load_img(filename, target_size=(299, 299))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = feature_extractor.predict(image, verbose=0)
    return feature

def generate_caption(image_path, beam_index=3):
    if not model_ready:
        return "A wonderful scene captured in a photograph. (Note: Please train the model in the model_training folder to see real AI predictions!)"
        
    try:
        feature = extract_features(image_path)
        
        # start with the initial sequence
        start = [word_to_index.get('startseq', 0)]
        
        # list of [sequence, log_probability]
        start_word = [[start, 0.0]]
        
        while len(start_word[0][0]) < max_length:
            temp = []
            for s in start_word:
                # If this sequence already reached 'endseq', just carry it over
                if s[0][-1] == word_to_index.get('endseq', 0):
                    temp.append(s)
                    continue
                    
                sequence = pad_sequences([s[0]], maxlen=max_length)
                preds = model.predict([feature, sequence], verbose=0)
                
                # get top `beam_index` predictions
                word_preds = np.argsort(preds[0])[-beam_index:]
                
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += np.log(preds[0][w] + 1e-10)
                    temp.append([next_cap, prob])
                    
            # sort by probability descending
            start_word = sorted(temp, reverse=True, key=lambda l: l[1])
            # keep top beams
            start_word = start_word[:beam_index]
            
            # check if all top beams have ended
            if all(s[0][-1] == word_to_index.get('endseq', 0) for s in start_word):
                break
            
        start_word = start_word[0][0]
        intermediate_caption = [index_to_word.get(i, '') for i in start_word]
        
        final_caption = []
        for i in intermediate_caption:
            if i == 'startseq':
                continue
            if i == 'endseq':
                break
            if i:
                final_caption.append(i)
                
        return ' '.join(final_caption)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return f"Error generating caption: {str(e)}"

# Persistent loop to listen to stdin
print("READY")
sys.stdout.flush()

for line in sys.stdin:
    image_path = line.strip()
    if not image_path:
        continue
    if image_path.lower() == 'exit':
        break
        
    caption = generate_caption(image_path)
    
    # Output the result as JSON
    result = {
        "status": "success",
        "caption": caption
    }
    print(json.dumps(result))
    sys.stdout.flush()

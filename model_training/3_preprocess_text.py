import os
import string
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Paths
BASE_DIR = r"c:\Users\swaya\OneDrive\Desktop\image caption"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CAPTIONS_FILE = os.path.join(DATASET_DIR, "captions.txt")
OUTPUT_CAPTIONS_DICT = os.path.join(DATASET_DIR, "captions_dict.pkl")
OUTPUT_TOKENIZER = os.path.join(DATASET_DIR, "tokenizer.pkl")

def clean_text(text):
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove words with numbers in them
    text = " ".join(word for word in text.split() if word.isalpha())
    # Remove single character words (like 'a') except 'a' might be useful, let's keep all alpha words > 1 or 'a'
    text = " ".join(word for word in text.split() if len(word) > 1 or word == 'a')
    # Add start and end sequence tokens
    text = "startseq " + text + " endseq"
    return text

def process_captions(captions_path):
    df = pd.read_csv(captions_path)
    print(f"Loaded {len(df)} captions.")
    
    mapping = {}
    
    for i in range(len(df)):
        img_id = df.iloc[i]['image']
        caption = df.iloc[i]['caption']
        
        # Clean caption
        cleaned_caption = clean_text(str(caption))
        
        if img_id not in mapping:
            mapping[img_id] = []
        mapping[img_id].append(cleaned_caption)
        
    return mapping

if __name__ == "__main__":
    print("Processing captions...")
    mapping = process_captions(CAPTIONS_FILE)
    print(f"Total unique images with captions: {len(mapping)}")
    
    # Save the mapping
    with open(OUTPUT_CAPTIONS_DICT, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"Saved captions dictionary to {OUTPUT_CAPTIONS_DICT}")
    
    # Extract all captions to build tokenizer
    all_captions = []
    for key, captions in mapping.items():
        all_captions.extend(captions)
        
    # Build Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary Size: {vocab_size}")
    
    # Find max caption length
    max_length = max(len(caption.split()) for caption in all_captions)
    print(f"Max Caption Length: {max_length}")
    
    # Save the tokenizer
    with open(OUTPUT_TOKENIZER, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Saved tokenizer to {OUTPUT_TOKENIZER}")
    
    # Save max_length to a text file for reference
    with open(os.path.join(DATASET_DIR, "max_length.txt"), "w") as f:
        f.write(str(max_length))

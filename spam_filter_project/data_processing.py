import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import re 

nltk_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
nltk.data.path.append(nltk_data_path)

try:
    nltk.download('punkt', quiet=True, download_dir=nltk_data_path)
    nltk.download('stopwords', quiet=True, download_dir=nltk_data_path)
except Exception as e:
    print(f"[NLTK] Warning: Could not download resources. Error: {e}. Ensure 'nltk_data' directory exists or run 'nltk.download()' manually.")

STOPWORDS_SET = set(stopwords.words('english'))

def load_and_clean_data(data_path):
    # Load the dataset
    df = pd.read_csv(data_path, encoding='utf-8', names=['label', 'message'], sep='\t')

    # 1. Remove Duplicates
    initial_count = len(df)
    df.drop_duplicates(inplace=True)
    final_count = len(df)
    print(f"[DATA] Initial messages: {initial_count}, Duplicates removed: {initial_count - final_count}, Final messages: {final_count}")

    # 2. Engineer Features
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message_length'] = df['message'].apply(len)

    return df

def preprocess_text(text: str) -> str:
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Tokenization
    tokens = word_tokenize(text)

    # 3. Filtering and cleaning tokens
    cleaned_tokens = []
    spam_indicators = ['£', '$', '€', '!', '%', 'free', 'win'] 
    
    for word in tokens:
        cleaned_word = re.sub(r'^\W+|\W+$', '', word) 
        if not cleaned_word:
            continue
        if cleaned_word in STOPWORDS_SET and cleaned_word not in spam_indicators:
            continue
        if len(cleaned_word) <= 1 and not cleaned_word.isdigit():
             continue
        
        cleaned_tokens.append(cleaned_word)

    # 4. Join tokens back into a single string for TF-IDF
    return " ".join(cleaned_tokens) if cleaned_tokens else ""

def save_processed_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[DATA] Processed data saved to {output_path}")

if __name__ == '__main__':
    print("--- Testing data_processing.py ---")
    
    TEST_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'SMSSpamCollection.csv'))
    OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'preprocessed_data.csv'))

    try:
        df_cleaned = load_and_clean_data(TEST_DATA_PATH)
    except FileNotFoundError:
        print(f"TEST FAILED: Please ensure 'SMSSpamCollection.csv' is in the 'data/raw/' directory at: {TEST_DATA_PATH}")
        exit()

    if not df_cleaned.empty:
        test_message = "Free entry! Win £1000! Call 0800-123456 now."
        print(f"\n--- Preprocessing Test ---")
        print(f"Original: {test_message}")
        print(f"Processed: {preprocess_text(test_message)}")
        print(f"--------------------------")

        df_cleaned['processed_message'] = df_cleaned['message'].apply(preprocess_text)
        save_processed_data(df_cleaned, OUTPUT_PATH)
        print("\nCleaned DataFrame Sample:")
        print(df_cleaned[['message', 'processed_message', 'label_num', 'message_length']].head())
        print("\nFinal Label Counts:")
        print(df_cleaned['label'].value_counts())
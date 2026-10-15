

# Load the necessary libraries
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import nltk
import re
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load your experimental CSV data (synthetic sample provided in the repository)
csv_path = Path(__file__).resolve().parent.parent / "data" / "example_reviews.csv"
df = pd.read_csv(csv_path)

# Load the GoEmotions model from Hugging Face
model_name = "monologg/bert-base-cased-goemotions-original"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
emotion_analyzer = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Download NLTK stopwords data
nltk.download("stopwords")

# Load NLTK stopwords and stemmer
stop_words = set(stopwords.words("english"))
word_tokenizer = nltk.WordPunctTokenizer()
stemmer = PorterStemmer()

# Define a function for text cleaning, stemming, and emotion classification
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters, numbers, and extra spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize the text
    words = word_tokenizer.tokenize(text)

    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Join the words back into a sentence
    cleaned_text = ' '.join(words)

    # Ensure the text is within the model's maximum sequence length (tokens)
    max_seq_length = 512

    # Perform emotion classification
    predicted_emotion = emotion_analyzer(cleaned_text, truncation=True, max_length=max_seq_length)[0]['label']
    return predicted_emotion

# Apply the preprocessing and classification function to each row in your experimental CSV data
results = []
for text in df['review']:
    if pd.isna(text) or not str(text).strip():
        continue
    predicted_emotion = preprocess_text(str(text))
    results.append(predicted_emotion)

# Create a frequency table for the most common emotions in your experimental data
emotion_counts = {}
for emotion in results:
    if emotion in emotion_counts:
        emotion_counts[emotion] += 1
    else:
        emotion_counts[emotion] = 1

# Sort the emotion counts in descending order
sorted_emotion_counts = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)

# Display the frequency table
for emotion, count in sorted_emotion_counts:
    print(emotion + ": " + str(count))

from gensim.models import FastText
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import pandas as pd
from nltk.corpus import stopwords
import string

nltk.download('punkt')

# Load the dataset
train_df = pd.read_csv('../datasets/clef2019/train/traindf_labeled.csv')
test_df = pd.read_csv('../datasets/clef2019/test/testdf_labeled.csv')
val_df = pd.read_csv('../datasets/clef2019/valid/valdf_labeled.csv')
df = pd.concat([train_df, test_df, val_df])

# Create the lists of sentences
questions = df['question'].tolist()
answers = df['answer'].tolist()
example_corpus = questions + answers

# Tokenize the sentences
stop_words = set(stopwords.words('english'))
sentences = []

for sentence in example_corpus:
    tokens = word_tokenize(sentence.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    sentences.append(tokens)
    
# Train the model
model = FastText(sentences=sentences, vector_size=768, window=5, min_count=1, workers=4)

model.save("wordtovec.model")
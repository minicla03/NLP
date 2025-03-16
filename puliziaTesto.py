import nltk
nltk.download('punkt')  # Download punkt tokenizer
nltk.download('stopwords')  # Download stopwords for English
nltk.download('averaged_perceptron_tagger')  # Download POS tagger for English
nltk.download('wordnet')

from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
import emoji

'''
Remove emojis
Remove punctuation and special characters (except spaces and apostrophes)
Convert all text to lowercase
Remove stopwords in English
Tokenize the text into words
Apply lemmatization
Remove numbers
'''

txt = "Hello!!😊 Today is a beautiful day, right? But... wait: will it rain tomorrow?😕 " \
      "I read on www.meteo.it that there is a 70% chance of rain🌧️. " \
      "Anyway, today we are going to the park at 3:30 PM. See you there!!!"

# Remove emojis
txt = emoji.replace_emoji(txt, "")
print(txt)

# Split the text into sentences
tokenizer = PunktSentenceTokenizer()
phrases = tokenizer.tokenize(txt)
print(phrases)

# Remove punctuation (except spaces and apostrophes)
phrases = [re.sub(r'[^\w\s\'-]', '', phrase) for phrase in phrases]
print(phrases)

# Remove numbers
phrases = [re.sub(r"\d", "", phrase) for phrase in phrases]
print(phrases)

# Convert all text to lowercase
phrases = [phrase.lower() for phrase in phrases]
print(phrases)

# Tokenize each phrase in the 'phrases' list individually
stop_words = set(stopwords.words('english'))  # Use English stopwords
phrases_tokenized = [
    [word for word in word_tokenize(phrase, language="english") if word not in stop_words]
    for phrase in phrases
]
print(phrases_tokenized)

# Lemmatize (but first tag the words with POS tags)
tokens_tag = [pos_tag(phrase) for phrase in phrases_tokenized]  # POS tagging for English
print(tokens_tag)

# Apply lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_phrases = [
    [lemmatizer.lemmatize(word[0], pos='v') for word in phrase]  # 'v' for verb lemmatization
    for phrase in tokens_tag
]
print(lemmatized_phrases)

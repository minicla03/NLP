from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import re

text = "Dr. Marco went to the bank at 9:30. He passed by the riverbank and noticed some strange plants. " \
        "When he entered, he saw a sign that said: 'Come visit us, the Bank of Milan offers excellent services!'. " \
        "He sat down and ordered a coffee with an apple, but the barista gave him a coffee with a MELA, an electronic device he was trying to sell. " \
        "Marco smiled and said, 'I don't think you can sell a MELA in a bank, right?'. " \
        "Then, his phone, an iPhone, started vibrating, signaling a message: 'Don't forget to bring the documents for your visit tomorrow!'. " \
        "The message came from his assistant, who had warned him about an important meeting."

contractions = {
    "don't": "do not",
    "Don't": "Do not",
    "can't": "cannot",
    "isn't": "is not",
    "aren't": "are not",
    "it's": "it is",
    "i'm": "i am",
    "he's": "he is",
    "she's": "she is",
    "we're": "we are",
    "they're": "they are",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "didn't": "did not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not"
}

# Funzione per sostituire le contrazioni nel testo
def expand_contractions(text, contractions_dict):
    for contraction, expanded in contractions_dict.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", expanded, text)
    return text

def tagConversion(tag):
    match(tag[0]):
        case 'J':
            return wordnet.ADJ
        case 'V':
            return wordnet.VERB
        case 'N':
            return wordnet.NOUN
        case 'R':
            return wordnet.ADV
        case _:
            return wordnet.NOUN

text=expand_contractions(text, contractions)
punkt_param = PunktParameters()
abbreviation = ['dr', 'u.s.a', 'fig', 'etc', 'i.e', 'e.g']
postfix_abbr = "etc"

punkt_param.abbrev_types = set(abbreviation)
tokenizer = PunktSentenceTokenizer(punkt_param)
text = re.sub(r"\b" + r"(" + postfix_abbr + r")\.(\s+[\[\],(){};:-]*\s*[A-Z])", r"\1 .\2", text)

sentences = tokenizer.tokenize(text)

sentences = [re.sub(r"[^\w\s]", " ", sentence) for sentence in sentences]
sentences = [re.sub(r":\s", " ", sentence) for sentence in sentences]

stop_words = set(stopwords.words('english'))
words = [[word for word in word_tokenize(sentence) if word not in stop_words] for sentence in sentences]
acronymous = ['MELA']
words = [[word.lower() if word not in acronymous else word for word in phrase] for phrase in words]

tag=tokens_tag = [pos_tag(phrase) for phrase in words]

lemmatizer = WordNetLemmatizer()
lemmatized_phrases = [
    [lemmatizer.lemmatize(word[0], tagConversion(word[1])) for word in phrase]  
    for phrase in tokens_tag
]
print(lemmatized_phrases)

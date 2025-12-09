import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

def load_corpus(path="computers_extended.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = sent_tokenize(text)
    return sentences

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z0-9\s]", " ", sentence)
    tokens = word_tokenize(sentence)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def preprocess_sentences(sentences):
    return [preprocess_sentence(s) for s in sentences]

def get_best_answer(query, original_sents, cleaned_sents):
    cleaned_query = preprocess_sentence(query)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(cleaned_sents + [cleaned_query])

    scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

    best_index = scores.argmax()
    best_score = scores[best_index]

    if best_score < 0.1:
        return "Sorry, I don't have enough information about that."

    return original_sents[best_index]


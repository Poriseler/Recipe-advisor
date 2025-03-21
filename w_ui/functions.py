import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


def prepare_text(document: str) -> str:
    lower_text = document.lower()
    tokenized_text = nltk.word_tokenize(lower_text)

    wo_alphanumeric_text = [word for word in tokenized_text if word.isalnum()]
    clean_text = [
        word for word in wo_alphanumeric_text if word not in nltk.corpus.stopwords.words('english')]

    return clean_text


def search_25(query: str, model: BM25Okapi):
    tokenized_query = prepare_text(query)

    docs_scored = model.get_scores(tokenized_query)

    return docs_scored

import re
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, v_measure_score


class TextClusterer:
    """TF-IDF + KMeans clustering for the medical dataset."""

    def __init__(self, n_clusters=5, 
            ngram_range=(1, 2), 
            min_df=2, 
            random_state=42,
            stemming=False,
            lemmatization=False
        ):
        
        self.stemming = stemming
        self.lemmatization = lemmatization
        
        self.tokenizer = self._build_tokenizer()
        
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            tokenizer=self.tokenizer,
            lowercase=True,
            strip_accents="unicode"
        )
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def _build_tokenizer(self):
        if self.lemmatization:
            import spacy

            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            def lemmatize(text):
                doc = nlp(text.lower())
                return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
            return lemmatize

        elif self.stemming:
            from nltk.stem import PorterStemmer

            stemmer = PorterStemmer()

            def stem_tokenize(text):
                tokens = re.findall(r"[A-Za-z]+", text.lower())
                return [stemmer.stem(tok) for tok in tokens]

            return stem_tokenize

        else:
            return None

    def fit(self, texts, true_labels=None):
        """Fit clustering model to texts and return cluster labels + metrics."""
        X = self.vectorizer.fit_transform(texts)
        labels = self.model.fit_predict(X)

        metrics = {
            "silhouette": silhouette_score(X, labels, metric="euclidean")
        }

        if true_labels is not None:
            metrics["v_measure"] = v_measure_score(true_labels, labels)

        return labels, metrics

    def top_terms_per_cluster(self, n=10):
        """Print top TF-IDF terms per cluster."""
        terms = self.vectorizer.get_feature_names_out()
        order_centroids = self.model.cluster_centers_.argsort()[:, ::-1]
        for i in range(self.model.n_clusters):
            top_terms = [terms[ind] for ind in order_centroids[i, :n]]
            print(f"Cluster {i}: {', '.join(top_terms)}")
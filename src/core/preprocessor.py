import string
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud
import numpy as np


class Preprocessor:
    def __init__(self, extra_stopwords=None):
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            self.stop_words = set(stopwords.words("english"))
        self.puncts = set(string.punctuation)

        default_custom = [
            "ref",
            "cite",
            "http",
            "https",
            "www",
            "://",
            "com",
            "org",
            "net",
            "category",
            "isbn",
            "doi",
            "pmid",
            "arxiv",
        ]
        if extra_stopwords:
            default_custom.extend(extra_stopwords)
        self.stop_words.update(default_custom)

    def preprocess(self, text):
        """Tokenize, lowercase, remove stopwords/punctuations, stem words"""
        tokens = wordpunct_tokenize(text.lower())
        processed = []
        for token in tokens:
            if (
                token not in self.stop_words
                and token not in self.puncts
                and len(token) > 2
            ):
                processed.append(self.stemmer.stem(token))
        return processed

    def get_bag_of_words(self, text):
        """Return bag of words list and frequency dictionary"""
        tokens = self.preprocess(text)
        bow = []
        word_freq = defaultdict(int)
        for token in tokens:
            bow.append(token)
            word_freq[token] += 1
        return bow, dict(word_freq)

    def get_word_freq(self, dataset, max_docs=1000, use_preprocessing=True):
        """Compute word frequencies for dataset"""
        counter = Counter()
        for i, example in enumerate(dataset):
            if i >= max_docs:
                break
            text = example.get("text", "")
            tokens = (
                self.preprocess(text)
                if use_preprocessing
                else wordpunct_tokenize(text.lower())
            )
            counter.update(tokens)
        return counter

    def tokenize(self, text, keep_stopwords=False, keep_punct=False):
        """
        Tokenize text into lowercase tokens while preserving positions.
        Optionally removes stopwords and punctuations.
        """
        tokens = wordpunct_tokenize(text.lower())
        processed = []
        for token in tokens:
            # Skip unwanted tokens
            if not keep_stopwords and token in self.stop_words:
                continue
            if not keep_punct and token in self.puncts:
                continue
            if len(token) <= 2:
                continue
            # You can choose to stem or not — positional indexing doesn't require stemming
            processed.append(self.stemmer.stem(token))
        return processed

    def plot_comparison_all(
        self, raw_counter, clean_counter, top_n_bar=20, top_n_cum=50, top_n_zipf=100
    ):
        """
        Compare raw vs preprocessed frequencies in multiple visualizations:
        1. Bar plot of top words
        2. Word cloud
        3. Cumulative frequency
        4. Zipf plot
        """
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        import numpy as np

        # ------------------- Bar Plot -------------------
        raw_common = dict(raw_counter.most_common(top_n_bar))
        clean_common = dict(clean_counter.most_common(top_n_bar))

        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        axs[0].bar(raw_common.keys(), raw_common.values())
        axs[0].set_title("Bar Plot (Without Preprocessing)")
        axs[0].tick_params(axis="x", rotation=45)
        axs[1].bar(clean_common.keys(), clean_common.values())
        axs[1].set_title("Bar Plot (With Preprocessing)")
        axs[1].tick_params(axis="x", rotation=45)
        plt.show()

        # ------------------- Word Cloud -------------------
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        wc_raw = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(raw_counter)
        wc_clean = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(clean_counter)
        axs[0].imshow(wc_raw, interpolation="bilinear")
        axs[0].axis("off")
        axs[0].set_title("Word Cloud (Without Preprocessing)")
        axs[1].imshow(wc_clean, interpolation="bilinear")
        axs[1].axis("off")
        axs[1].set_title("Word Cloud (With Preprocessing)")
        plt.show()

        # ------------------- Cumulative Frequency -------------------
        raw_common = raw_counter.most_common(top_n_cum)
        clean_common = clean_counter.most_common(top_n_cum)
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        axs[0].plot(np.cumsum([freq for _, freq in raw_common]), marker="o")
        axs[0].set_xticks(range(top_n_cum))
        axs[0].set_xticklabels([w for w, _ in raw_common], rotation=90)
        axs[0].set_title("Cumulative Frequency (Without Preprocessing)")
        axs[1].plot(np.cumsum([freq for _, freq in clean_common]), marker="o")
        axs[1].set_xticks(range(top_n_cum))
        axs[1].set_xticklabels([w for w, _ in clean_common], rotation=90)
        axs[1].set_title("Cumulative Frequency (With Preprocessing)")
        plt.show()

        # ------------------- Zipf Plot -------------------
        raw_common = raw_counter.most_common(top_n_zipf)
        clean_common = clean_counter.most_common(top_n_zipf)
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        axs[0].loglog(
            range(1, top_n_zipf + 1), [freq for _, freq in raw_common], marker="o"
        )
        axs[0].set_title("Zipf Plot (Without Preprocessing)")
        axs[0].set_xlabel("Rank")
        axs[0].set_ylabel("Frequency")
        axs[1].loglog(
            range(1, top_n_zipf + 1), [freq for _, freq in clean_common], marker="o"
        )
        axs[1].set_title("Zipf Plot (With Preprocessing)")
        axs[1].set_xlabel("Rank")
        axs[1].set_ylabel("Frequency")
        plt.show()

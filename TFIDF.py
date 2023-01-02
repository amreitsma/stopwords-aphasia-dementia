import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import *
from nltk.corpus import stopwords
np.set_printoptions(threshold=np.inf)


def vectorize(docs, stopwds=False, max_ngram=3):
    """Turn a list of str documents into a matrix of TFIDF features.

    docs: list of strings

    stopwds: set to True to filter stop words from the data

    max_ngram: the max range of ngrams to use. Default is 3, meaning uni-gram, bigram and trigram features are included
    """

    if stopwds:
        print("EXCLUDING STOP WORDS")
        vct = TfidfVectorizer(token_pattern=r"(?u)(?!\bx\b)(\b\w+\b|\.{1,})", lowercase=True,
                              stop_words=stopwords.words('english'), ngram_range=(1, max_ngram))
    else:
        print("INCLUDING STOP WORDS")
        vct = TfidfVectorizer(token_pattern=r"(?u)(?!\bx\b)(\b\w+\b|\.{1,})", lowercase=True,
                              ngram_range=(1, max_ngram))
    vec_data = vct.fit_transform(docs)
    # print(vec_data.shape) # Uncomment to see how many features are extracted from the data

    return vec_data


def main():
    ad_data = preprocess_data("Dementia_chat.zip", "ProbableAD")
    ea_data = preprocess_data("Aphasia_chat.zip", "Broca")
    vectorize(ad_data + ea_data)


if __name__ == "__main__":
    main()

import numpy as np
import nltk
from nltk.corpus import stopwords

def tf_idf(D):

    def tf(t, d):
        return sum([1 for w in d.split() if t == w])/len(d.split())
    def idf(t):
        if sum([1 for d in D if t in d]) != 0:
            return np.log(len(D)/sum([1 for d in D if t in d]))
        else:
            return 0
    fd = list(nltk.FreqDist(nltk.word_tokenize(" ".join(D))))
    fd = [i for i in fd if i not in stopwords.words('english')]
    tf_idf_matrix = np.array([[tf(t, d)*idf(t) for d in D] for t in fd])
    return tf_idf_matrix.T

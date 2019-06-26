import gensim
import pickle
import jieba
import numpy as np
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
TaggededDocument = gensim.models.doc2vec.TaggedDocument

def train(x_train, size=100, epoch_num=1):
    model_dm = Doc2Vec(x_train, min_count=4, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('model_doc2vec_chinese')
    return model_dm

def get_data():
    f = open('authorlist_chinese.pickle', 'rb')
    tblogs = pickle.load(f)
    data = []
    i= 0
    for blogs in tblogs:
        for j in range(len(blogs)):
            s = list(jieba.cut(blogs[j]))
            document = TaggededDocument(s, tags=[i])
            data.append(document)
            i += 1
    return data

if __name__ == '__main__':
    data = get_data()
    model_dm = train(data)
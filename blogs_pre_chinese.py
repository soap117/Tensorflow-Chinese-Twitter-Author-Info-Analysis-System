import pickle
import jieba
import numpy as np
f = open('authorlist_chinese.pickle', 'rb')
tblogs = pickle.load(f)
f = open('word2idx_chinese.pickle', 'rb')
word2idx = pickle.load(f)

for blogs in tblogs:
    for j in range(len(blogs)):
        s = list(jieba.cut(blogs[j]))
        vec = []
        for it in s:
            if  it in word2idx.keys():
                vec.append(word2idx[it])
            else:
                vec.append(word2idx['_unknow_'])
        blogs[j] = vec
blogs = np.array(tblogs)
np.save('authorblogs_chinese.npy', blogs)

from gensim.models.doc2vec import Doc2Vec, LabeledSentence
model_dm = Doc2Vec.load("model_doc2vec_chinese")
tt = model_dm.docvecs
i=0
f = open('authorlist_chinese.pickle', 'rb')
tblogs = pickle.load(f)
for blogs in tblogs:
    for j in range(len(blogs)):
        blogs[j] =  tt[i]
        i += 1
    if i%1000==0:
        print(i)
blogs = np.array(tblogs)
np.save('authorblogs_words_chinese.npy', blogs)

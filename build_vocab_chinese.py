import pickle
import jieba
from collections import Counter
f = open('authorlist_chinese.pickle', 'rb')
tblogs = pickle.load(f)
word2idx = {}
counter = Counter()
i = 0
for blogs in tblogs:
    i += 1
    for blog in blogs:
        s = list(jieba.cut(blog))
        for w in s:
            counter[w]+=1
    if i%1000==0:
        print(i/len(tblogs))
vocab = [word for word in counter if counter[word] > 4]
word2idx['_unknow_'] = 1
word2idx['_END_'] = 0
idx = 2
for word in vocab:
    word2idx[word] = idx
    idx += 1
f = open('word2idx_chinese.pickle', 'wb')
pickle.dump(word2idx, f)
print(idx)
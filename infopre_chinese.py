import pickle
import numpy as np
f = open('authorinformation_chinese.pickle', 'rb')
infos = pickle.load(f)
infovec = np.array(infos)
np.save('authorinfos_chinese.npy', infovec)

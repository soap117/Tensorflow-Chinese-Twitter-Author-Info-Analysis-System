# coding=utf-8
import os
import pickle
import numpy
import re
import urllib.request
import time
path = './weibo/'
files = os.listdir(path)
name_lists = ['coder','doctor','lawyer','stu','teacher','writer']
authorlist = []
authorinformation = []
id = 0
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0','cookie':'_T_WM=1c5c09473305ba5dd573ba0ba7bb73d3; SCF=AsOr7eGuFCXNKGn8Zw_4YtVxfEuS8il03s3HCbe6QnnBxrQ4Qy0Yz3qM5NvyAbgJGPuijtYjZT5oEBrKGpayQnA.; SUB=_2A253DSg-DeRhGeRI7lcS9CrMyj-IHXVUDkh2rDV6PUJbktANLW6hkW1NHetkTy1dEauOufVeBlMe9WRdDXK4mgiv; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WWa6Y9DqfXuKSevH3S7OUva5JpX5K-hUgL.FozcSK-0ShB7eKe2dJLoI0zLxK-LBKeLB-zLxK-LB--L1KnLxK.L1K-LB.qLxKML1-2L1hBLxKnLB-qLBoB_Tgp3wBtt; SUHB=0P1NGT17SWgIX4; SSOLoginState=1510561902'}
for file_name in name_lists:
    files = path+file_name+'_result.pkl'
    dicts = path + file_name + '_dict.pkl'
    s = open(files, 'rb')
    files = pickle.load(s)
    s = open(dicts, 'rb')
    dicts = pickle.load(s)
    count = 0
    id = id + 1
f1 = open('authorlist_chinese.pickle', 'wb')
f2 = open('authorinformation_chinese.pickle', 'wb')
pickle.dump(authorlist, f1)
pickle.dump(authorinformation, f2)


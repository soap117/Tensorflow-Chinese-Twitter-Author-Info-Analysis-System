import tensorflow as tf
import pickle
import numpy as np
import jieba
from tkinter import *
import tkinter as tk
from gensim.models.doc2vec import Doc2Vec, LabeledSentence


class MainWindow:
    def __init__(self):
        self.frame = Tk()
        self.frame.geometry('300x200')
        self.frame.title('文本分析工具-罗君宇')
        self.sex_ans = StringVar()
        self.sex_ans.set("None")
        self.career_ans = StringVar()
        self.career_ans.set("None")
        self.label_inputs = Label(self.frame, text="短文")
        self.label_career = Label(self.frame, text="职业相关:")
        self.label_sex = Label(self.frame, text="性别:")
        self.label_condition = Label(self.frame, text="等待加载模型中...")

        self.text_inputs = Text(self.frame, height="4", width=30)
        self.text_career = Entry(self.frame, width=30, textvariable=self.career_ans)
        self.text_sex = Entry(self.frame,  width=30, textvariable=self.sex_ans)

        self.label_inputs.grid(row=0, column=0)
        self.label_career.grid(row=1, column=0)
        self.label_sex.grid(row=2, column=0)
        self.label_condition.grid(row=3, column=1)

        self.button_ok = Button(self.frame, text="ok", width=10)
        self.button_cancel = Button(self.frame, text="加载模型", width=10)

        self.text_inputs.grid(row=0, column=1)
        self.text_career.grid(row=1, column=1)
        self.text_sex.grid(row=2, column=1)

        self.button_ok.grid(row=4, column=1)
        self.button_cancel.grid(row=5, column=1)
        self.button_ok.bind("<ButtonRelease-1>", self.run)
        self.button_cancel.bind("<ButtonRelease-1>", self.load)
        self.frame.mainloop()
    def load(self, event):
        self.name_lists = ['程序员', '医生', '律师', '学生', '老师', '作家']
        self.gender_lists = ['男', '女']
        self.model_dm = Doc2Vec.load("model_doc2vec_chinese")
        f = open('word2idx_chinese.pickle', 'rb')
        self.word2idx = pickle.load(f)
        self.sexg = tf.Graph()
        self.careerg = tf.Graph()
        with self.sexg.as_default():
            self.sblogs, self.sblogs_vec, self.sans, self.ss, self.sdr = sex()
        with self.careerg.as_default():
            self.cblogs, self.cblogs_vec, self.cans, self.cs, self.cdr = career()
        self.label_condition['text'] = '加载完成'
    def get_text(self):
        text = self.text_inputs.get('1.0',END)
        return text
    def run(self,event):
        text = self.get_text()
        lists = list(jieba.cut(text))
        wvec = []
        for w in lists:
            if w in self.word2idx.keys():
                wvec.append(self.word2idx[w])
            else:
                wvec.append(self.word2idx['_unknow_'])
        temp = np.zeros(256)
        if len(wvec) > 255:
            temp[0:255] = wvec[0:255]
        else:
            temp[255 - len(wvec):255] = wvec
        vec = self.model_dm.infer_vector(lists, steps=20, alpha=0.025)
        with self.sexg.as_default():
            ans = self.ss.run(self.sans,
                              feed_dict={self.sblogs: temp[np.newaxis, :], self.sblogs_vec: vec[np.newaxis, :], self.sdr:1.0})
            self.sex_ans.set(self.gender_lists[ans[0]])
        with self.careerg.as_default():
            ans = self.cs.run(self.cans,
                              feed_dict={self.cblogs: temp[np.newaxis, :], self.cblogs_vec: vec[np.newaxis, :], self.cdr:1.0})
            self.career_ans.set(self.name_lists[ans[0]])

def career():
    from config import config
    from model_career_chinese import model
    f = open('word2idx_chinese.pickle', 'rb')
    word2idx = pickle.load(f)
    V = len(word2idx)
    config = config(V)
    config.batch_size = 1
    model = model(config)
    [dr, blogs, blogs_vec, ans, loss, accuracy, pans] = model.build(is_dropout=False)
    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=configs)
    saver = tf.train.Saver()
    saver.restore(sess, './saving/save_model_career_c.ckpt')
    writer = tf.summary.FileWriter("D://Graph//model_map", sess.graph)
    writer.close()
    return blogs, blogs_vec, tf.arg_max(pans,1), sess, dr
def sex():
    from config import config
    from model_sex_chinese import model
    f = open('word2idx_chinese.pickle', 'rb')
    word2idx = pickle.load(f)
    V = len(word2idx)
    config = config(V)
    config.batch_size = 1
    model = model(config)
    [dr, blogs, blogs_vec, ans, loss, accuracy, pans] = model.build(is_dropout=False)
    configs = tf.ConfigProto()
    configs.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=configs)
    saver = tf.train.Saver()
    saver.restore(sess, './saving/save_model_sex_c.ckpt')
    return blogs, blogs_vec, tf.arg_max(pans,1), sess, dr

def main():
    frame = MainWindow()
main()
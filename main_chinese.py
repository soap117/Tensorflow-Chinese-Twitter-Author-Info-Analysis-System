import tensorflow as tf
import pickle
import numpy as np
def create_set_shuffle(authorblogs, authorinfos, authorblogs_words):
    train_data = []
    test_data_words = []
    train_ans = []
    test_data = []
    train_data_words = []
    test_ans = []
    count = 0
    count2 = 0
    for i in range(len(authorblogs)):
        tlen = len(authorblogs[i])
        for j in range(tlen):
            rs = np.random.uniform(0,1)
            if rs<1.0:
                temp = np.zeros(256)
                if len(authorblogs[i][j]) > 255:
                    temp[0:255] = authorblogs[i][j][0:255]
                else:
                    temp[255-len(authorblogs[i][j]):255] = authorblogs[i][j]
                train_data.append(temp)
                train_data_words.append(authorblogs_words[i][j])
                train_ans.append(authorinfos[i])
                count += 1
            else:
                temp = np.zeros(256)
                if len(authorblogs[i][j]) > 255:
                    temp[0:255] = authorblogs[i][j][0:255]
                else:
                    temp[255 - len(authorblogs[i][j]):255] = authorblogs[i][j]
                test_data.append(temp)
                test_data_words.append(authorblogs_words[i][j])
                test_ans.append(authorinfos[i])
                count2 += 1
    return np.array(train_data), np.array(train_data_words), np.array(train_ans), np.array(test_data),np.array(test_data_words), np.array(test_ans)

def setdata():
    authorblogs = np.load('authorblogs_chinese.npy')
    authorinfos = np.load('authorinfos_chinese.npy')
    authorblogs_words = np.load('authorblogs_words_chinese.npy')
    train_data, train_data_words, train_ans, test_data, test_data_words, test_ans = create_set_shuffle(authorblogs,
                                                                                                       authorinfos,
                                                                                                       authorblogs_words)
    np.save('train_data_c.npy', train_data)
    np.save('train_data_words_c.npy', train_data_words)
    np.save('train_ans_c.npy', train_ans)
    np.save('test_data_c.npy', test_data)
    np.save('test_data_words_c.npy', test_data_words)
    np.save('test_ans_c.npy', test_ans)

def train_career():
    from config import config
    from model_career_chinese import model
    f = open('word2idx_chinese.pickle', 'rb')
    word2idx = pickle.load(f)
    V = len(word2idx)
    config = config(V)
    model = model(config)
    batch_size = config.batch_size
    train_data = np.load('train_data_c.npy')
    train_data_words = np.load('train_data_words_c.npy')
    train_ans = np.load('train_ans_c.npy')
    blogs, blogs_vec, ans, loss, accuracy, _ = model.build(is_dropout=True)
    var = tf.trainable_variables()
    optim = tf.train.AdamOptimizer(0.001).minimize(loss, var_list=var)
    saver = tf.train.Saver()
    configs = tf.ConfigProto()
    sess = tf.InteractiveSession(config=configs)
    init = tf.initialize_all_variables()
    sess.run(init)
    #saver.restore(sess, './save_model_career_c.ckpt')
    epoch = 5001
    for i in range(epoch):
        seed = np.random.randint(0, train_data.shape[0], size=(batch_size))
        data_list = np.array([train_data[w] for w in seed])
        ans_list = np.array([train_ans[w][0] for w in seed])
        data_list_words = np.array([train_data_words[w] for w in seed])
        sess.run(optim,
                 feed_dict={blogs: data_list, blogs_vec:data_list_words, ans: ans_list})
        if i % 10 == 0:
            ac, ls = sess.run([accuracy, loss],
                          feed_dict={blogs: data_list, blogs_vec:data_list_words, ans: ans_list})
            print("i: %d, acc: %f ,loss: %f" % (i, ac, ls))
        if i % 5000 == 0:
            saver.save(sess, './save_model_career_c.ckpt')

def test_career():
    from config import config
    from model_career_chinese import model
    f = open('word2idx_chinese.pickle', 'rb')
    word2idx = pickle.load(f)
    #f = open('career2idx.pickle', 'rb')
    #career2idx = pickle.load(f)
    V = len(word2idx)
    config = config(V)
    model = model(config)
    batch_size = config.batch_size
    test_data = np.load('test_data_c.npy')
    test_data_words = np.load('test_data_words_c.npy')
    test_ans = np.load('test_ans_c.npy')
    blogs, blogs_vec, ans, loss, accuracy, _ = model.build(is_dropout=False)
    configs = tf.ConfigProto()
    sess = tf.InteractiveSession(config=configs)
    saver = tf.train.Saver()
    saver.restore(sess, './save_model_career_c.ckpt')
    lens = test_data.shape[0]
    T = int(lens/batch_size)
    fin_ac = 0.0
    for i in range(T):
        data_list = test_data[i*batch_size:(i+1)*batch_size]
        data_list_words = test_data_words[i * batch_size:(i + 1) * batch_size]
        ans_list = test_ans[i*batch_size:(i+1)*batch_size, 0]
        ac = sess.run(accuracy,
                          feed_dict={blogs: data_list, blogs_vec:data_list_words, ans: ans_list})
        fin_ac += ac/T
    print('acc on career: %f' %fin_ac)
def train_sex():
    from config import config
    from model_sex_chinese import model
    f = open('word2idx_chinese.pickle', 'rb')
    word2idx = pickle.load(f)
    V = len(word2idx)
    config = config(V)
    model = model(config)
    batch_size = config.batch_size
    train_data = np.load('train_data_c.npy')
    train_data_words = np.load('train_data_words_c.npy')
    train_ans = np.load('train_ans_c.npy')
    blogs, blogs_vec, ans, loss, accuracy, _ = model.build(is_dropout=True)
    var = tf.trainable_variables()
    optim = tf.train.AdamOptimizer(0.001).minimize(loss, var_list=var)
    saver = tf.train.Saver()
    configs = tf.ConfigProto()
    sess = tf.InteractiveSession(config=configs)
    init = tf.initialize_all_variables()
    sess.run(init)
    saver.restore(sess, './save_model_sex_c.ckpt')
    epoch = 5001
    for i in range(epoch):
        seed = np.random.randint(0, train_data.shape[0], size=(batch_size))
        data_list = np.array([train_data[w] for w in seed])
        ans_list = np.array([train_ans[w, 1] for w in seed])
        data_list_words = np.array([train_data_words[w] for w in seed])
        sess.run(optim,
                 feed_dict={blogs: data_list, blogs_vec:data_list_words, ans: ans_list})
        if i % 10 == 0:
            ac, ls = sess.run([accuracy, loss],
                          feed_dict={blogs: data_list, blogs_vec:data_list_words, ans: ans_list})
            print("i: %d, acc: %f ,loss: %f" % (i, ac, ls))
        if i % 5000 == 0:
            saver.save(sess, './save_model_sex_c.ckpt')

def test_sex():
    from config import config
    from model_sex_chinese import model
    f = open('word2idx_chinese.pickle', 'rb')
    word2idx = pickle.load(f)
    #f = open('career2idx.pickle', 'rb')
    #career2idx = pickle.load(f)
    V = len(word2idx)
    config = config(V)
    model = model(config)
    batch_size = config.batch_size
    test_data = np.load('test_data_c.npy')
    test_data_words = np.load('test_data_words_c.npy')
    test_ans = np.load('test_ans_c.npy')
    blogs, blogs_vec, ans, loss, accuracy, _ = model.build(is_dropout=False)
    configs = tf.ConfigProto()
    sess = tf.InteractiveSession(config=configs)
    saver = tf.train.Saver()
    saver.restore(sess, './save_model_sex_c.ckpt')
    lens = test_data.shape[0]
    T = int(lens/batch_size)
    fin_ac = 0.0
    for i in range(T):
        data_list = test_data[i*batch_size:(i+1)*batch_size]
        data_list_words = test_data_words[i * batch_size:(i + 1) * batch_size]
        ans_list = test_ans[i*batch_size:(i+1)*batch_size, 1]
        ac = sess.run(accuracy,
                          feed_dict={blogs: data_list, blogs_vec:data_list_words, ans: ans_list})
        fin_ac += ac/T
    print('acc on sex: %f' % fin_ac)
if __name__ == "__main__":
    setdata()
    #train_career()
    tf.reset_default_graph()
    #test_career()
    tf.reset_default_graph()
    train_sex()
    tf.reset_default_graph()
    #test_sex()
    tf.reset_default_graph()

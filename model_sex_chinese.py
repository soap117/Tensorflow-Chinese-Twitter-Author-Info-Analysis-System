import tensorflow as tf
from ops import *
class model(object):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.V = config.V
        self.T = 256
        self.Vec = 100
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self.blogs = tf.placeholder(tf.int32, [self.batch_size, self.T])
        self.blogs_vec = tf.placeholder(tf.float32, [self.batch_size, self.Vec])
        self.ans = tf.placeholder(tf.int32, [self.batch_size])
        self.drop_out_rate = tf.placeholder(tf.float32)
    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.hidden_size], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x
    def build(self, is_dropout):
        with tf.variable_scope('blogs'):
            blogs_emb = tf.nn.tanh(fully_connected(self.blogs_vec, 64, 'emb'))
        with tf.variable_scope('cnn'):
            xt = self._word_embedding(inputs=self.blogs)
            conv_features = conv1d_noraml(xt, 1024)
            conv_features_max = tf.reduce_max(conv_features, 1)
            conv_emb = tf.nn.tanh(fully_connected(conv_features_max, 64, 'emb'))
        with tf.variable_scope('lstm'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size, reuse=False)
            state = lstm_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=int(self.hidden_size/2), reuse=False)
            state2 = lstm_cell2.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            for i in range(self.T):
                with tf.variable_scope('lstm1', reuse=(i!=0)):
                    xt = self._word_embedding(inputs=self.blogs[:, i])
                    out, state = lstm_cell(inputs=xt, state=state)
                    out_drop = tf.nn.dropout(out, self.drop_out_rate)
                with tf.variable_scope('lstm2', reuse=(i!=0)):
                    out2, state2 = lstm_cell2(inputs=out_drop, state=state2)
        total_emb = tf.concat([blogs_emb, conv_emb, out2], axis=1)
        if is_dropout:
            total_emb = tf.nn.dropout(total_emb, self.drop_out_rate)
        rs = fully_connected(total_emb, 2, 'output')
        ans_ = tf.nn.softmax(rs)
        correct_prediction = tf.equal(tf.cast(tf.argmax(ans_, 1), tf.int32), self.ans)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=rs, labels=tf.one_hot(self.ans, 2)))
        return self.drop_out_rate, self.blogs, self.blogs_vec, self.ans, loss, accuracy, ans_
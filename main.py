# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:00:05 2019

@author: 11955
"""

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn ###***

import numpy as np
import re
import itertools
from collections import Counter

import pandas as pd


#0 设置参数   不能重复定义参数名称
###***tf.flags.DEFINE_float("参数名称", 具体取值, "说明")
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "data/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "data/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 40, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("num_doplay", 20, "Num epochs to do test and some play")
tf.flags.DEFINE_string("save_path", "model/", "The path of the model to save")
tf.flags.DEFINE_integer("lstm_size", 256, "The size of every lstm")
tf.flags.DEFINE_integer("lstm_layers", 3, "The layers of RNN")
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()  ###***解析参数。新版本用FLAGS.flag_values_dict()  旧版本用FLAGS._parse_flags()


print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
    


#1 获取规整数据
#对每个句子去除标点符号、大小写等规范操作函数
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

#获取规整的数据集 X,y
def load_data_and_labels(positive_data_file, negative_data_file):
    positive = open(positive_data_file, "rb").read().decode('utf-8')  ###***读取原始数据  ["I am a good boy./nand you?/nfind thank you."]
    negative = open(negative_data_file, "rb").read().decode('utf-8') 
    
    positive_examples = positive.split('\n')[:-1] #["I am a good boy.", "and you?", "find thank you."]
    negative_examples = negative.split('\n')[:-1]
    
    positive_examples = [s.strip() for s in positive_examples]#["I am a good boy", "and you", "find thank you"]
    negative_examples = [s.strip() for s in negative_examples]
    
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text] #["i am a good boy", "and you", "find thank you"]
    
    #生成label
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0) #np.concatenate([], 0)函数[[[0,1],[0,1]], [[1,0],[1,0]]]-->[[0,1],[0,1], [1,0],[1,0]]
    
    return [x_text, y]  ###*** [x_text, y]两个取值

x_text, y = load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

#将x_text词索引化。 这里将每句话数字表示，维度为最长那句话单词个数。可以自己在训练模型的时候训练稠密矩阵W，也可以直接用别人已经有的稠密矩阵效果会更好
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

###***打乱数据集
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]


#2 建立RNN网络
class TextRNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, lstm_size, lstm_layers, batch_size):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x") 
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        #隐含层
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
#            self.W = tf.Variable(
#                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), 
#                                        #vocab_size表示词袋中所有词的个数。模型训练参数W更新的过程其实就是在训练稠密向量矩阵
#                                        #这里可以用别人已经训练好的稠密向量矩阵W，和对应的词索引
#                    name = "W"
#                    )
            self.W = tf.Variable(tf.truncated_normal((vocab_size, embedding_size), stddev=0.01))
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x) #embedded_chars-->[sequence_length, embedding_size]
            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        
        #搭建一层基本的lstm
#        with tf.name_scope("baselstm"):
#            self.lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        
        #dropout
#        with tf.name_scope("dropout"):
#            self.drop = tf.contrib.rnn.DropoutWrapper(self.lstm, self.dropout_keep_prob)
#            self.initial_state = self.drop.zero_state(batch_size, tf.float32)
        
        #搭LSTM网络
        with tf.name_scope("build_lstm_layers"):
            ###***多层  lstm_layers层    如果搭建单层，lstm_layers=1
            self.cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(lstm_layers)])
            #dropout
            self.drop = tf.contrib.rnn.DropoutWrapper(self.cell, self.dropout_keep_prob)
            #对于每条输入数据，都要有一个初始状态，所以一共有batch_size个初始状态
            self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        
        #输出
        with tf.name_scope("outputs_scores_predictions"):
            self.outputs, self.final_state = tf.nn.dynamic_rnn(self.drop, self.embedded_chars, initial_state=self.initial_state)
                        #outputs-->[batch_size, num_classes]
            #得到全连接层后的输出数据并添加激活函数
            self.scores = tf.contrib.layers.fully_connected(self.outputs[:, -1], num_outputs=2, activation_fn=tf.sigmoid) 
                                    ###***tf.contrib.layers.fully_connected()这个全连接函数自动初始化W,b；num_outputs表示输出的个数  scores-->[batch_size, num_classes]
            #对上一步得到的数据做softmax()
            self.predsm = tf.nn.softmax(self.scores) # predsm -->[batch_size, num_classes]
                                       ###***区分softmax()和argmax()
                                        #softmax()函数：
                                        #A = [[1.0,2.0,3.0,4.0,5.0,6.0],[1.0,2.0,3.0,4.0,5.0,6.0]]  
                                        #tf.nn.softmax(A)   #输出[[ 0.00426978  0.01160646  0.03154963  0.08576079  0.23312201  0.63369131], [ 0.00426978  0.01160646  0.03154963  0.08576079  0.23312201  0.63369131]]
                                        #函数tf.argmax(x=, axis=)
                                        #test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
                                        #np.argmax(test, 0)　　　＃输出：array([3, 3, 1]
                                        #np.argmax(test, 1)　　　＃输出：array([2, 2, 0, 0]123
                                        
                                      ###***tf.reshape(x, shape)
                                      
            self.predictions = tf.argmax(self.predsm, 1, name="predictions") #predictions -->[batch_size, 1] 例如：[[0],[0,[1],[0]......]]
                             
        ###***求损失loss，loss一般不用predictions一维的矩阵求，而是用概率分布num_classes的矩阵
        with tf.name_scope("loss"):
            
            self.loss = tf.losses.mean_squared_error(self.input_y, self.predsm) 
                        ###***例：tf.losses.mean_squared_error([[1,0],[1,0],[0,1]], [[0.25,0.75],[0.62,0.38],[0.84,0.16]])
            
        ###***求准确率一般用predictions和转化后的input_y(tf.argmax(self.input_y, 1))的一维矩阵求。
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1, name="input_y"))
                        #correct_predictions -->[batch_size, 1]
                        #例：tf.equal([[1],[0],[0],[1],[0]], [[0],[0],[0],[1],[1]])
                        ###***tf.equal()函数应用时数据类型格式的坑！！   数据类型转换函数tf.cast(x, tf.float32)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            
#3 训练
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
            allow_soft_placement = FLAGS.allow_soft_placement,
            log_device_placement = FLAGS.log_device_placement
            )
    sess = tf.Session(config=session_conf)
    
    with sess.as_default():
        rnn = TextRNN(
                sequence_length=x_train.shape[1], 
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_), 
                embedding_size=FLAGS.embedding_dim, 
                lstm_size=FLAGS.lstm_size, 
                lstm_layers=FLAGS.lstm_layers, 
                batch_size=FLAGS.batch_size
                )
            
#        global_step = tf.Variable(0, name="global_step", trainable=False)
#        optimizer = tf.train.AdamOptimizer(1e-3)
#        grads_and_vars = optimizer.compute_gradients(rnn.loss)
#        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        optimizer = tf.train.AdamOptimizer().minimize(rnn.loss)
        saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(FLAGS.num_epochs):
            total_batch = int(x_train.shape[0]/FLAGS.batch_size)  
                    ###***RNN好像只能输入batch_size大小的batch
                    ###？？？真正预测的时候如何解决这个问题？例如要预测一个测试集，预测代码如何写？补齐，再去掉无用？
            for i in range(total_batch):
                x_batch = x_train[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                y_batch = y_train[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                feed_dict = {
                        rnn.input_x:x_batch,
                        rnn.input_y:y_batch,
                        rnn.dropout_keep_prob:FLAGS.dropout_keep_prob
                        }
                _, loss = sess.run([optimizer, rnn.loss], feed_dict) ###***训练模型，要有优化器optimizer
            
            if epoch % FLAGS.num_doplay == 0:
                #求训练集的损失、准确度
                train_total_batch = int(x_train.shape[0]/FLAGS.batch_size)
                tra_loss = []
                tra_acc = []
                for i in range(train_total_batch):
                    x_train_batch = x_train[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                    y_train_batch = y_train[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                    feed_dict = {
                        rnn.input_x:x_train_batch,  ###***rnn.input_x
                        rnn.input_y:y_train_batch,
                        rnn.dropout_keep_prob:1.0 ###***keep_prob=1.0时，做预测。
                        }
                    train_loss, train_acc = sess.run([rnn.loss, rnn.accuracy], feed_dict)
                    tra_loss.append(train_loss)
                    tra_acc.append(train_acc)
                
                
                ###***测试集求损失、准确率、预测值
                val_loss = []
                val_acc = []
                predictions = []
                test_total_batch = int(x_dev.shape[0]/FLAGS.batch_size)
                for i in range(test_total_batch):
                    x_test_batch = x_dev[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                    y_test_batch = y_dev[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                    feed_dict = {
                        rnn.input_x:x_test_batch,
                        rnn.input_y:y_test_batch,
                        rnn.dropout_keep_prob:1.0 ###***keep_prob=1.0时，做预测。
                        }
                    test_loss, prediction, test_acc = sess.run([rnn.loss, rnn.predictions, rnn.accuracy], feed_dict)
                    val_loss.append(test_loss)
                    val_acc.append(test_acc)
                    for p in prediction:
                        predictions.append(p)
                        
                ###***得到测试集的所有预测值并保存
                pd.DataFrame(predictions).to_csv("result_data/pred"+"_"+str(epoch)+".csv", index=False)
               
                print("已完成{:g}/{:g}次迭代".format(epoch+1, FLAGS.num_epochs))
                print("train loss={:g}, train acc={:g}".format(np.mean(tra_loss), np.mean(tra_acc)))
                print("test loss={:g}, test acc={:g}".format(np.mean(val_loss), np.mean(val_acc))) ###***"...{:g/s}...".format()函数的用法
                
#                feed_dict = {
#                    rnn.input_x:x_dev,
#                    rnn.input_y:y_dev,
#                    rnn.dropout_keep_prob:1.0 ###***keep_prob=1.0时，做预测。
#                    }
#                test_loss, test_acc, predictions = sess.run([rnn.loss, rnn.accuracy, rnn.predictions], feed_dict)
#                ##***得到测试集的所有预测值并保存
#                pd.DataFrame(predictions).to_csv("result_data/pred"+"_"+str(epoch)+".csv", index=False)
#                print("已完成{:g}/{:g}次迭代".format(epoch+1, FLAGS.num_epochs))
#                print("train loss={:g}, train acc={:g}".format(np.mean(tra_loss), np.mean(tra_acc))) ###***"...{:g/s}...".format()函数的用法
#                print("test loss={:g}, test acc={:g}".format(test_loss, test_acc))
                    
            if (epoch % FLAGS.checkpoint_every == 0) or (epoch == (FLAGS.num_epochs-1)):
                saver.save(sess, FLAGS.save_path+"_"+str(epoch)+"_"+".ckpt")
            
        print("Done!")
        
        
        



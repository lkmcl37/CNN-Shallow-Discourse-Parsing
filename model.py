# -*- mode: Python; coding: utf-8 -*-

'''
@author: KaMan Leong
'''
import collections
import tensorflow as tf
import helper
import numpy as np
import json
import math

class DNN_Model():

    '''
    This class implements two models: two-CNN and two-LSTM
    They are two seperate types of models and executed according to user's
    command (whether it's -m cnn or -m lstm).

    They are NOT HYBRID.

    '''
    
    '''
    twoCNN Structure: one cnn for one sentence(get output1), 
    another cnn for another sentence(get output2). 
    then: softmax(output1 + output2)
    '''

    '''
    twoLSTM Structure: one bi-directional lstm for one sentence(get output1), 
    another one bi-directional lstm for another sentence(get output2). 
    then: softmax(output1 + output2)
    '''
        
    def __init__(self, num_classes, num_chars, seq_len, epoch=20, model_name="cnn"):
        
        self.num_epochs = epoch
        self.learning_rate = 0.001
        self.batch_size = 128
        
        self.num_chars = num_chars
        self.emb_dim = 100
        self.dropout = 0.75
        
        self.seq_len = seq_len
        self.max_acc = 0.0
        self.filter_sizes = [1, 2, 3]
        
        self.num_filters = 64
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.num_classes = num_classes
        self.l2_lambda = 0.0001 

        #hyper-parameter for lstm
        self.num_layers = 2
        self.hidden_dim = 100
        
        # argument 1 and 2
        self.input1 = tf.placeholder(tf.int32, [None, self.seq_len], name="input_x1")
        self.input2 = tf.placeholder(tf.int32, [None, self.seq_len], name="input_x2")
        
        self.targets = tf.placeholder(tf.int32, [None], name= "input_y")

        #embedding
        self.word_emb = tf.Variable(
        tf.random_uniform([self.num_chars, self.emb_dim], -1.0, 1.0),
        name="emb", trainable = True)
            
        self.inputs_emb1 = tf.nn.embedding_lookup(self.word_emb, self.input1)
        self.inputs_emb2 = tf.nn.embedding_lookup(self.word_emb, self.input2)
            
        #for bi-directional bilstm
        if model_name == "lstm":
            output1 = self.bilstem_cell(self.hidden_dim, self.dropout, self.inputs_emb1, 1)
            output2 = self.bilstem_cell(self.hidden_dim, self.dropout, self.inputs_emb2, 2)
            self.W = tf.get_variable("W", [self.hidden_dim * 4, self.num_classes])
        else:
            #for cnn
            self.sent_emb_expanded1 = tf.expand_dims(self.inputs_emb1, -1) 
            self.sent_emb_expanded2 = tf.expand_dims(self.inputs_emb2, -1) 
        
            output1 = self.conv_relu_pool_dropout(self.sent_emb_expanded1, name_scope_prefix="s1")
            output2 = self.conv_relu_pool_dropout(self.sent_emb_expanded2, name_scope_prefix="s2")
        
            self.W = tf.get_variable("W", [self.num_filters_total * 2, self.num_classes])

        #concate the features of two arguments
        output_rnn_pooled = tf.concat([output1, output2], axis=1)
        
        #softmax
        self.b = tf.get_variable("b", [self.num_classes])  
        self.logits = tf.matmul(output_rnn_pooled, self.W) + self.b
        
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits);
        self.loss = tf.reduce_mean(losses)
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
        self.loss = self.loss + l2_losses
        tf.summary.scalar('cross_entropy', self.loss)
        tf.summary.histogram("histogram loss", self.loss)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions") 
       
        #evaluate performance
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.targets) 
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy") 
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.histogram("histogram accuracy", self.accuracy)
        
        self.summary_op = tf.summary.merge_all()

    #CNN Wraper function
    def conv_relu_pool_dropout(self, sent_emb_expanded, name_scope_prefix=None):
        
        pooled_outputs = []
        
        for filter_size in self.filter_sizes:
            
            with tf.name_scope(name_scope_prefix + "convolution-pooling-%s" % filter_size):
               
                filter = tf.get_variable(name_scope_prefix + "filter-%s" % filter_size, [filter_size, self.emb_dim, 1, self.num_filters],
                                         initializer = tf.random_normal_initializer(stddev=0.1))
                
                # build convolutional layer
                conv = tf.nn.conv2d(sent_emb_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")
                
                # bias
                b = tf.get_variable(name_scope_prefix + "b-%s" % filter_size, [self.num_filters])
                
                # batch Normalization
                tmp, _, _ = self.batch_norm_layer(tf.nn.bias_add(conv, b))
                
                h = tf.nn.relu(tmp,
                               "relu") 
                pooled = tf.nn.max_pool(h, ksize=[1, self.seq_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID',
                                        name="pool")
                
                pooled_outputs.append(pooled)
       
        h_pool = tf.concat(pooled_outputs, 3) 
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total]) 
        
        with tf.name_scope('dropout'):
            h_drop = tf.nn.dropout(h_pool_flat, keep_prob = self.dropout) 
            
        return h_drop
    
    # Batch Normalization layer
    def batch_norm_layer(self, inputs, decay=0.9):

        epsilon = 1e-5
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))

        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=True)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])

        train_mean = tf.assign(pop_mean,
                pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                pop_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon), batch_mean, batch_var

    #LSTM Cell wrapper function
    def bilstem_cell(self, hidden_dim, dropout, inputs_emb, num):
        
        with tf.variable_scope("lstmcell_" + str(num)):
            lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(hidden_dim)  
            lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
                    
            lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw, output_keep_prob=(dropout))  
            lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob=(dropout))
            
            lstm_cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw] * 2)  
            lstm_cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw] * 2)
            
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_fw, lstm_cell_bw, inputs_emb, dtype=tf.float32)     
          
            output_rnn = tf.concat(outputs, axis=2)
            output_rnn_pooled = tf.reduce_mean(output_rnn, axis=1)
    
        return output_rnn_pooled
            
    def train(self, sess, save_file, X_train, y_train, X_val, y_val):  
        
        saver = tf.train.Saver()  
        
        summary_writer_train = tf.summary.FileWriter('loss_log/train_loss', sess.graph)    
        summary_writer_val = tf.summary.FileWriter('loss_log/val_loss', sess.graph)       
        
        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))  
  
        for epoch in range(self.num_epochs): 
             
            # shuffle train in each epoch  
            sh_index = np.arange(len(X_train))  
            np.random.shuffle(sh_index)  
            X_train = X_train[sh_index]  
            y_train = y_train[sh_index] 
             
            print("current epoch: %d" % (epoch))
             
            for iteration in range(num_iterations):  
                # train  
                X_train_batch1, X_train_batch2, y_train_batch = helper.nextBatch(X_train, y_train, iteration * self.batch_size, self.batch_size)  
                
                _, train_loss, train_acc, train_summary = sess.run([
                                    self.optimizer,
                                    self.loss,
                                    # self.predictions,
                                    self.accuracy,
                                    self.summary_op
                                    ],
                                    feed_dict={
                                        self.input1:X_train_batch1,
                                        self.input2:X_train_batch2,
                                        self.targets:y_train_batch})
                
                if iteration % 20 == 0:  
                    # train_acc = helper.extractSense(y_train_batch, train_y_hat)
                    summary_writer_train.add_summary(train_summary, iteration)
                    print("iteration: %5d, train loss: %5d, train precision: %.5f" % (iteration, train_loss, train_acc))  
                      
                # validation  
                if iteration % 20 == 0:  
                    
                    X_val_batch1, X_val_batch2, y_val_batch = helper.nextRandomBatch(X_val, y_val, self.batch_size)  
                    dev_loss, dev_acc, val_summary = sess.run([
                                    self.loss,
                                    # self.predictions,
                                    self.accuracy,
                                    self.summary_op
                                    ],
                                    feed_dict={
                                        self.input1:X_val_batch1,
                                         self.input2:X_val_batch2,
                                        self.targets:y_val_batch})
                    
                    # test_acc = helper.extractSense(y_val_batch, dev_y_hat)
                    summary_writer_val.add_summary(val_summary, iteration)
                    print("iteration: %5d, dev loss: %5d, dev precision: %.5f" % (iteration, dev_loss, dev_acc))
                    
                    if dev_acc > self.max_acc:  
                        self.max_acc = dev_acc  
                        saver.save(sess, save_file)  
                        print("saved the best model with accuracy: %.5f" % (self.max_acc))  
           
    def test(self, sess, X_test, y_test, output_path, id2label, doc_meta):

        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        
        print("number of iteration: " + str(num_iterations)) 
        y_hat = []
        
        for iteration in range(num_iterations): 
                
            X_test_batch1, X_test_batch2, y_test_batch = helper.nextBatch(X_test, y_test, iteration * self.batch_size, self.batch_size)  
                
            pred = sess.run([
                            self.predictions],
                            feed_dict={
                            self.input1: X_test_batch1,
                            self.input2: X_test_batch2,
                            self.targets: y_test_batch,
                            })
            y_hat += list(pred[0])
            
        for i in range(len(doc_meta)):
                    
            meta_data = doc_meta[i]
                
            dic = collections.OrderedDict()
            sense = id2label[y_hat[i]]
                
            dic["Arg1"] = {"TokenList" : meta_data.arg_token_list[0]}
            dic["Arg2"] = {"TokenList" : meta_data.arg_token_list[1]}
            dic["Connective"] = {"TokenList" : meta_data.connective["TokenList"]}
            dic["DocID"] = meta_data.docid
                
            dic["Sense"] = [sense]
            dic["Type"] = meta_data.type
            
            with open(output_path, 'a', encoding="utf8") as outfile:
                outfile.write(json.dumps(dic))
                outfile.write('\n')

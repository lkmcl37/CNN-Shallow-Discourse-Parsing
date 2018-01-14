'''
@author: KaMan Leong
'''
import os
import sys
import time
import helper
import argparse
import tensorflow as tf
from model import DNN_Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#command example:
# python train.py train_feat dev_feat ./twoLSTM_10_model.ckpt -m cnn -n 10

parser = argparse.ArgumentParser()
parser.add_argument("train_path", help="the path of the train file")
parser.add_argument("dev_path", help="the path of the dev file")
parser.add_argument("save_path", help="the path of the saved model")
parser.add_argument("-n", "--seq_len", type=int, help="the maximum length of sequence")
parser.add_argument("-m", "--model", type=str, help="network used, for lstm, use -m lstm, for cnn, use -cnn", default="cnn")

args = parser.parse_args()

model_name = args.model

if model_name != "cnn" and model_name != "lstm":
    print("model name invalid! Please use -m cnn or -m lstm in command")
    sys.exit()

start_time = time.time()
seq_len = args.seq_len

print("Preparing train and validation data...")

train = helper.loadFile(args.train_path)
dev = helper.loadFile(args.dev_path)
map_dir = "token_label_id_mapping"

X_train, y_train, X_val, y_val, feat2id, id2label = helper.getTrain(train, dev, map_dir, seq_len)
num_chars = len(feat2id)
num_classes = len(id2label)

save_path = args.save_path
#emb_path = "word_emb_matrix_100d"

print("Building model...")
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
gpu_config = "/gpu:0"

with tf.Session(config=config) as sess:
    with tf.device(gpu_config): 
        initializer = tf.random_normal_initializer(stddev=0.1)
        
        with tf.variable_scope("model", reuse=None, initializer=initializer):

            model = DNN_Model(num_classes, num_chars, seq_len, 15, model_name)
            
            print("Training model...")
            tf.global_variables_initializer().run()
            model.train(sess, save_path, X_train, y_train, X_val, y_val)
    
            print("Final best accuracy is: %f" % (model.max_acc))
            end_time = time.time()
            print("Time used %f(hour)" % ((end_time - start_time) / 3600))

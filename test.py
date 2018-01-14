'''
@author: KaMan Leong
'''
import time
import helper
import argparse
import tensorflow as tf
from model import DNN_Model

#command example:
#python test.py test_feat ./twoLSTM_10_model.ckpt output.json -m lstm -n 10

parser = argparse.ArgumentParser()
parser.add_argument("test_path", help="the path of the dev file")
parser.add_argument("save_path", help="the path of the saved model")
parser.add_argument("output_path", help="the path of the output json file")
parser.add_argument("-n", "--seq_len", type=int, help="the maximum length of sequence")
parser.add_argument("-m", "--model", type=str, help="network used, for lstm, use -m lstm, for cnn, use -cnn", default="cnn")

args = parser.parse_args()

model_name = args.model

if model_name != "cnn" and model_name != "lstm":
    print("model name invalid! Please use -m cnn or -m lstm in command")
    sys.exit()

start_time = time.time()
seq_len = args.seq_len
model_path = args.save_path
output_path = args.output_path

print("Preparing test data...")
test = helper.loadFile(args.test_path)
map_dir = "token_label_id_mapping"

X_test, y_test, feat2id, label2id, id2label = helper.getTest(test, map_dir, seq_len)
num_chars = len(feat2id)
num_classes = len(id2label)


with tf.Session() as sess:
  
    initializer = tf.random_normal_initializer(stddev=0.1)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        
        model = DNN_Model(num_classes, num_chars, seq_len, 15, model_name)

        print("loading model parameter...")
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        print("Testing...")
        model.test(sess, X_test, y_test, output_path, id2label, test)

        end_time = time.time()
        print("Time used %f(hour)" % ((end_time - start_time) / 3600))
        

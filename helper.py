# -*- mode: Python; coding: utf-8 -*-
'''
@author: KaMan Leong
'''
import os
import json
import codecs
import pickle
import numpy as np

class Document():
        
    def __init__(self, sense, docid, arg_raw, offset, connective, type):
        
        self.connective = connective
        self.sense = sense
        self.docid = docid
       
        self.arg_raw = arg_raw
        self.arg_token_list = offset
        self.type = type
        
#a function for extracting fields of json files
def load_corpus(data_dir):
    
    pdtb_file = codecs.open(data_dir, encoding='utf8')
    relations = [json.loads(x) for x in pdtb_file]
    
    return relations

def extractSense(y, y_hat):
    
    correct = 0.0
    for pred, true in enumerate(y):
        if y_hat[pred] == true:
            correct += 1
   
    return correct / len(y)

def load_features():
    
    train_feat = []
    feat2id = {}
    id2feat = {}
    
    label2id = {}
    id2label = {}
    
    relations = load_corpus("train/relations.json")
    connect2id = {}
    connect2id[''] = 0
    
    feat2id["UNK"] = 0
    id2feat[0] = "UNK"
    
    for data in relations:
        
        doc_feat = []
        sent = []
        '''
        if data["Type"] == "Implicit":
            continue
        '''
        for i in range(1, 3):
            
            key = 'Arg' + str(i)
            tokenList = data[key]["TokenList"]
            raw = data[key]["RawText"]
            offset = data[key]["CharacterSpanList"][0][0]
            
            token_offset = []
            #if i == 1:
             #   tokenList = tokenList[-3:]
            #else:
             #   tokenList = tokenList[:3]
            for token in tokenList:
                word = raw[token[0] - offset: token[1] - offset]
                sent.append(word)

                word = word.lower()
                if word not in feat2id:
                    feat2id[word] = len(feat2id)
                    id2feat[len(feat2id) - 1] = word
                    
                token_offset.append(token[2])
             
            #boundary of two arguments
            sent.append("EOS")  
            doc_feat.append(token_offset) 
        
        doc_feat.append(sent[:-1]) 
        
        '''
        print(data["Sense"][0])
        print(data["DocID"])
        print(doc_feat[0])
        print(doc_feat[1])
        print(doc_feat[2])
        print(data["Type"])
        '''
        
        #sense = data["Type"] + "." + data["Sense"][0]
        sense = data["Sense"][0]
        con = data["Connective"]
        
        if sense not in label2id:
            label2id[sense] = len(label2id)
            id2label[len(label2id) - 1] = sense
            
        train_feat.append(Document(sense, data["DocID"], doc_feat[2], [doc_feat[0], doc_feat[1]], con, data["Type"] ))
    
    pickle.dump(train_feat, open("train_feat", 'wb'))
    #pickle.dump([feat2id, id2feat, label2id, id2label] , open("3_token_label_id_mapping", 'wb'))
    
    dev_feat = []
    relations = load_corpus("dev/relations.json")
             
    for data in relations:

        if data["Type"] == "Implicit":
            continue
        
        sent = []
        doc_feat = []
        for i in range(1, 3):
            
            key = 'Arg' + str(i)
            tokenList = data[key]["TokenList"]
            raw = data[key]["RawText"]
            offset = data[key]["CharacterSpanList"][0][0]
            
            token_offset = []
            
            for token in tokenList:
                word = raw[token[0] - offset: token[1] - offset]
                sent.append(word)
              
                token_offset.append(token[2])
                
            sent.append("EOS") 
            doc_feat.append(token_offset) 
             
        doc_feat.append(sent[:-1])
        sense = data["Sense"][0]
        #sense = data["Type"] + "." + data["Sense"][0]
        dev_feat.append(Document(sense, data["DocID"], doc_feat[2], [doc_feat[0], doc_feat[1]], data["Connective"], data["Type"]))
        
    pickle.dump(dev_feat, open("dev_feat", 'wb'))
    
    test_feat = []
    relations = load_corpus("test/relations.json")
    
    for data in relations:
        sent = []
        doc_feat = []

        if data["Type"] == "Implicit":
            continue
        
        for i in range(1, 3):
            
            key = 'Arg' + str(i)
            tokenList = data[key]["TokenList"]
            raw = data[key]["RawText"]
            offset = data[key]["CharacterSpanList"][0][0]
            
            token_offset = []
            
            for token in tokenList:
                word = raw[token[0] - offset: token[1] - offset]
                sent.append(word)
              
                token_offset.append(token[2])
                
            sent.append("EOS")
            doc_feat.append(token_offset) 
            
        doc_feat.append(sent[:-1])
        sense = data["Sense"][0]
        #sense = data["Type"] + "." + data["Sense"][0]
        test_feat.append(Document(sense, data["DocID"], doc_feat[2], [doc_feat[0], doc_feat[1]], data["Connective"], data["Type"]))
        
    pickle.dump(test_feat, open("test_feat", 'wb'))
    
#generate batch for training
def nextBatch(X, y, start_index, batch_size):
    
    last_index = start_index + batch_size
       
    X_batch = list(X[start_index:min(last_index, len(X))])
        
    y_batch = list(y[start_index:min(last_index, len(X))])
    
    if last_index > len(X):
        left_size = last_index - len(X)
        
        for _ in range(left_size):
            index = np.random.randint(len(X))
            
            X_batch.append(X[index])
            y_batch.append(y[index])

    X_batch1 = np.array([x[0] for x in X_batch])
    X_batch2 = np.array([x[1] for x in X_batch])
    
    y_batch = np.array(y_batch)
    
    return X_batch1, X_batch2, y_batch

def nextRandomBatch(X, y, batch_size=128):
    
    X_batch = []
    y_batch = []
    
    for _ in range(batch_size):
        index = np.random.randint(len(X))
          
        X_batch.append(X[index])
        y_batch.append(y[index])
        
    X_batch1 = np.array([x[0] for x in X_batch])
    X_batch2 = np.array([x[1] for x in X_batch])
    
    y_batch = np.array(y_batch)
    
    return X_batch1, X_batch2, y_batch

def getEmbedding(infile_path, vocab_path="token_label_id_mapping"):

    feat2id, _, _, _ = loadMap(vocab_path)

    with open(infile_path, "rb") as infile:
        
        emb_matrix = np.zeros((len(feat2id), 200))
        
        for row in infile:
            row = row.strip()            
            items = row.split()
            word = items[0]
            
            emb_vec = [float(val) for val in items[1:]]
            
            if word in feat2id:
                emb_matrix[feat2id[word]] = emb_vec
              
    pickle.dump(emb_matrix, open("word_emb_matrix_200d", 'wb'))  
    
    return emb_matrix

def padding(vector, pad, head = 0):
    
    if len(vector) >= pad:
        if head == 0:
            return vector[-pad:]
        else:
            return vector[:pad]

    if head == 1:
        vector = vector + ((pad - len(vector))*[0])
    else:
        vector = ((pad - len(vector))*[0]) + vector
    return vector

#vectorize the training data
#turn an argument containing words into a vector of word ids
def prepare(features, feat2id, label2id, pad):
    
    X = []
    y = []
    
    for data in features:
        
        X_i = []
        X_sent = []
        for word in data.arg_raw:
            if word == "EOS":
                X_i = padding(X_i, pad, 0)
                X_sent.append(X_i)
                X_i = []
            else:     
                word = word.lower()
                if word in feat2id:
                    X_i.append(feat2id[word])
                else:
                    X_i.append(feat2id["UNK"])
            
        X_i = padding(X_i, pad, 1)
        X_sent.append(X_i)
        X.append(X_sent)
        
    for data in features:
        y.append(label2id[data.sense])
    
    X = np.array([np.array(x) for x in X])
    y = np.array(y)

    return X, y

def getTest(test, dir, pad):
    
    feat2id, id2feat, label2id, id2label = loadMap(dir)
    X, y = prepare(test, feat2id, label2id, pad)

    print("test size:", len(X))
    
    return X, y, feat2id, label2id, id2label

def getTrain(train, dev, dir, seq_len):

    # convert the data in maxtrix
    feat2id, _, label2id, _ = loadMap(dir)
    X, y = prepare(train, feat2id, label2id, seq_len)

    # shuffle the samples
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    #for i in range(len(X)):
    X = X[indices]
    y = y[indices]

    dev_X, dev_y = prepare(dev, feat2id, label2id, seq_len)
    
    num_samples = len(dev_X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    #for i in range(len(dev_X)):
    dev_X = dev_X[indices]
    dev_y = dev_y[indices]
    
    print("train size: %d, dev size: %d" % (len(X), len(dev_X)))
    return X, y, dev_X, dev_y, feat2id, label2id

def loadMap(token2id_filepath):
    
    if not os.path.isfile(token2id_filepath):
        print("file not exist")
        return
       
    feat2id, id2feat, label2id, id2label = pickle.load(open(token2id_filepath, "rb"))

    return feat2id, id2feat, label2id, id2label
    
def loadFile(data_path):
    
    with open(data_path, "rb") as file:
        data_file = pickle.load(file)
        
    return data_file

#load_features()
#getEmbedding("glove.6B.200d.txt", "token_label_id_mapping")

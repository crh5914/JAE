import math
import numpy as np
from functools import reduce

import keras
from keras import Model,backend as K,regularizers
from keras.layers import Dense,Embedding,Input,Reshape,Subtract,Lambda,Flatten,Multiply,Concatenate,Dropout
from keras.optimizers import Adadelta,Adam
import pandas as pd
import time
from scipy import sparse
import argparse
import logging

#from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description='keras jae')
    parser.add_argument('--dataset',type=str,default='ml100k',help='dataset')
    parser.add_argument('--sep',type=str,default='\t',help='data seperator')
    parser.add_argument('--du',type=str,default='[500,100]',help='user encoder dimension')
    parser.add_argument('--di',type=str,default='[500,100]',help='item encoder dimension')
    parser.add_argument('--fc',type=str,default='[200,100,1]',help='full connected layer dimension')
    parser.add_argument('--batch_size',type=int,default=256,help='batch size')
    parser.add_argument('--epoch',type=int,default=50,help='epoch')
    parser.add_argument('--lr',type=float,default=1e-4,help='learning rate')
    parser.add_argument('--reg',type=float,default=0.01,help='regurazier parameter')
    parser.add_argument('--alpha',type=float,default=0.1,help='alpha')
    parser.add_argument('--split',type=float,default=0.9,help='train ratio')
    parser.add_argument('--log',type=str,default='log',help='log file name')
    return parser.parse_args()
def get_logger(name):
    formatter = "%(asctime)s %(levelname)-8s: %(message)s"
    app_name = 'JAE'
    url ="./logs/{}.log".format(name)
    logger = logging.getLogger(app_name)
    formatter = logging.Formatter(formatter)
    file_handler = logging.FileHandler(url)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger
def recostruction_loss(true_y,pred_y):
    diff = K.square(true_y - pred_y)
    weight =K.cast(true_y>0,dtype='float32')
    weighted_diff = diff * weight
    return K.mean(K.sum(weighted_diff,axis=1))
def edge_wise_loss(true_y,pred_y):
    """ 1st order proximity
    """
    return K.mean(K.square(pred_y-true_y))

def build_model(N,M,adj_mat,du,di,fc,lr,alpha,reg):
    input_a = Input(shape=(1,),name='input-a',dtype='int32')
    input_b = Input(shape=(1,),name='input-b',dtype='int32')
    user_embedding_layer = Embedding(input_dim=N,output_dim=M,trainable=False,input_length=1,name='user-table')
    user_embedding_layer.build((None,))
    user_embedding_layer.set_weights([adj_mat])
    user_embedding = user_embedding_layer(input_a)
    user_embedding = Flatten()(user_embedding)
    item_embedding_layer = Embedding(input_dim=M,output_dim=N,trainable=False,input_length=1,name='item-table')
    item_embedding_layer.build((None,))
    item_embedding_layer.set_weights([adj_mat.T])
    item_embedding = item_embedding_layer(input_b)
    item_embedding = Flatten()(item_embedding)
    user_hidden = Dense(du[0],activation='relu',kernel_regularizer=regularizers.l2(reg),name='user-encoding-layer-{}'.format(0))(user_embedding)
    item_hidden = Dense(di[0],activation='relu',kernel_regularizer=regularizers.l2(reg),name='item-encoding-layer-{}'.format(0))(item_embedding)
    encoded_a = Dense(du[1],activation='sigmoid',kernel_regularizer=regularizers.l1(reg),name='user-encoding-layer-{}'.format(1))(user_hidden)
    encoded_b = Dense(di[1],activation='sigmoid',kernel_regularizer=regularizers.l1(reg),name='item-encoding-layer-{}'.format(1))(item_hidden)
    user_out_hidden = Dense(du[0],activation='relu',kernel_regularizer=regularizers.l2(reg),name='user-decoding-layer-{}'.format(0))(encoded_a)
    item_out_hidden = Dense(di[0],activation='relu',kernel_regularizer=regularizers.l2(reg),name='item-decoding-layer-{}'.format(0))(encoded_b)
    decoded_a = Dense(M,activation='relu',kernel_regularizer=regularizers.l2(reg),name='user-decoding-layer-{}'.format(1))(user_out_hidden)
    decoded_b = Dense(N,activation='relu',kernel_regularizer=regularizers.l2(reg),name='item-decoding-layer-{}'.format(1))(item_out_hidden)
    embedding_diff = Concatenate()([encoded_a,encoded_b])
    fc_map = [embedding_diff]
    for idx,s in enumerate(fc):
         fc_map.append(Dense(s,activation='relu',kernel_regularizer=regularizers.l2(reg),name='predict-layer-{}'.format(idx))(fc_map[-1]))  
    result =  fc_map[-1]
    model = Model(inputs=[input_a,input_b],outputs=[decoded_a,decoded_b,result])
    opt = Adam(lr=lr)
    model.compile(optimizer=opt,loss=[recostruction_loss,recostruction_loss,edge_wise_loss],loss_weights=[alpha,alpha,1])
    predictor = Model(inputs=[input_a,input_b],outputs=[result])
    predictor.compile(optimizer=opt,loss=[edge_wise_loss],metrics=[edge_wise_loss])
    model.summary()
    encoder = Model(input_a,encoded_a)
    decoder = Model(input_a,decoded_a)
    decoder.compile(optimizer='adadelta',loss=recostruction_loss)
    return model,encoder,decoder,predictor
def main(it):
    #file = './data/ml100k_ratings.csv'    
    args = parse_args()
    logger = get_logger(args.log)
    print(args)
    logger.info(str(args))
    ds,split = args.dataset,args.split
    file = './data/{}.ratings'.format(ds)
    uids = {}
    iids = {}
    def get_user_idx(uid):
        if uid not in uids:
            uids[uid] = len(uids)
        return uids[uid]
    def get_item_idx(iid):
        if iid not in iids:
            iids[iid] = len(iids)
        return iids[iid]
    
    headers = ['uid','iid','rate','timestamp']
    data = pd.read_csv(file,sep=args.sep,header=None)
    data.columns = headers
    data['uid'] = data.uid.apply(get_user_idx)
    data['iid'] = data.iid.apply(get_item_idx)
    N,M,count = data.uid.nunique(),data.iid.nunique(),len(data)
    print('num user:{},num item:{},num record:{}'.format(N,M,count))
    logger.info('num user:{},num item:{},num record:{}'.format(N,M,count))   
    idxs = list(range(count))
    np.random.shuffle(idxs)
    train_size = int(count*split)
    train_idxs = idxs[:train_size]
    test_idxs = idxs[train_size:]
    s = np.sum(data.rate.values)
    rates = data.rate.values.copy()
    rates[test_idxs] = 0
    adj_mat = sparse.coo_matrix((rates,(data.uid.values,data.iid.values)),shape=(N,M))
    adj_mat = np.array(adj_mat.todense())
    du,di,fc = eval(args.du),eval(args.di),eval(args.fc)
    model,encode,decoder,predictor = build_model(N,M,adj_mat,du,di,fc,args.lr,args.alpha,args.reg)
    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []
    epochs,batch_size = args.epoch,args.batch_size 
    for epoch in range(args.epoch):
        se,ae = 0,0
        for i in range(0,len(train_idxs),batch_size):
            if i+batch_size >= len(train_idxs):
                batch_idxs = train_idxs[i:]
            else:
                batch_idxs = train_idxs[i:i+batch_size]
            batch_train_uids = data.uid.values[batch_idxs]
            batch_train_iids = data.iid.values[batch_idxs]
            batch_train_uvecs = adj_mat[batch_train_uids]
            batch_train_ivecs = adj_mat.T[batch_train_iids]
            batch_train_rates = data.rate.values[batch_idxs][:,None]
            model.train_on_batch([batch_train_uids,batch_train_iids],[batch_train_uvecs,batch_train_ivecs,batch_train_rates])
            pred_rates = predictor.predict_on_batch([batch_train_uids,batch_train_iids])
            se += np.sum(np.square(pred_rates-batch_train_rates))
            ae += np.sum(np.abs(pred_rates-batch_train_rates))
        rmse = np.sqrt(se/len(train_idxs))
        mae = ae/len(train_idxs)
        train_rmse.append(rmse)
        train_mae.append(mae)
        print('Train--ecpoh:{},rmse:{},mae:{}'.format(epoch+1,rmse,mae))
        logger.info('Train ecpoh:{},rmse:{},mae:{}'.format(epoch+1,rmse,mae))
        se,ae = 0,0
        for i in range(0,len(test_idxs),batch_size):
            if i+batch_size >= len(test_idxs):
                batch_idxs = test_idxs[i:]
            else:
                batch_idxs = test_idxs[i:i+batch_size]
            batch_test_uids = data.uid.values[batch_idxs]
            batch_test_iids = data.iid.values[batch_idxs]
            batch_test_uvecs = adj_mat[batch_test_uids]
            batch_test_ivecs = adj_mat.T[batch_test_iids]
            batch_test_rates = data.rate.values[batch_idxs][:,None]
            pred_rates = predictor.predict_on_batch([batch_test_uids,batch_test_iids])
            se += np.sum(np.square(pred_rates-batch_test_rates))
            ae += np.sum(np.abs(pred_rates-batch_test_rates))
        rmse = np.sqrt(se/len(test_idxs))
        mae = ae/len(test_idxs)
        test_rmse.append(rmse)
        test_mae.append(mae)
        print('Test--ecpoh:{},rmse:{},mae:{}'.format(epoch+1,rmse,mae))
        logger.info('Train ecpoh:{},rmse:{},mae:{}'.format(epoch+1,rmse,mae))
    result = pd.DataFrame({'train_rmse':train_rmse,'train_mae':train_mae,'test_rmse':test_rmse,'test_mae':test_mae})
    rfile = "./logs/{}_du_{}_di_{}_{}_{}_{}.csv".format(args.dataset,du[1],di[1],args.alpha,args.reg,it)
    result.to_csv(rfile,index=False)
if __name__ == '__main__':
    for i in range(5):
        main(i)

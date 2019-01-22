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
from tqdm import tqdm
import os
import argparse
import logging
def parse_args():
    parser = argparse.ArgumentParser(description='keras jae')
    parser.add_argument('--dataset',type=str,default='./data/ml100k.ratings',help='data file path')
    parser.add_argument('--sep',type=str,default='\t',help='data seperator')
    parser.add_argument('--user_encoder_layer',type=str,default='[1000,500]',help='user encoder dimension')
    parser.add_argument('--item_encoder_layer',type=str,default='[1000,500]',help='item encoder dimension')
    parser.add_argument('--fc_layer',type=str,default='[500,100,1]',help='full connected layer dimension')
    parser.add_argument('--batch_size',type=int,default=512,help='batch size')
    parser.add_argument('--epoch',type=int,default=50,help='epoch')
    parser.add_argument('--lr',type=float,default=1e-4,help='learning rate')
    parser.add_argument('--reg',type=float,default=0.01,help='regurazier parameter')
    parser.add_argument('--train_size',type=float,default=0.9,help='train ratio')
    parser.add_argument('--log',type=str,default='./logs/log.log',help='log file path')
    return parser.parse_args()

def get_logger(url):
    formatter = "%(asctime)s %(levelname)-8s: %(message)s"
    app_name = 'JAE'
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
    num = K.sum(weight,axis=1)
    return K.mean(K.sum(weighted_diff,axis=1)/num)

def edge_wise_loss(true_y,pred_y):
    """ 1st order proximity
    """
    return K.mean(K.square(pred_y-true_y))

def get_model(num_user,num_item,rating_matrix,user_encoder_layer,item_encoder_layer,fc_layer,lr,l2_param):
    #user = Input(shape=(1,),name='user',dtype='int32')
    #item = Input(shape=(1,),name='item',dtype='int32')
    user_emb = Input(shape=(num_item,),name='user_emb',dtype='float32')
    item_emb = Input(shape=(num_user,),name='item_emb',dtype='float32')
    user_encoding = [user_emb]
    # user encoder
    for idx,s in enumerate(user_encoder_layer):
        layer = Dense(s,activation='relu',kernel_regularizer=regularizers.l2(l2_param),name='user_encoder_layer%d'%(idx+1))
        user_encoding.append(layer(user_encoding[-1]))
    user_vec = user_encoding[-1]
    user_decoder_layer = user_encoder_layer[::-1][1:] + [num_item]
    user_decoding = [user_vec]
    # user decoder
    for idx,s in enumerate(user_decoder_layer):
        layer = Dense(s,activation='relu',kernel_regularizer=regularizers.l2(l2_param),name='user_decoder_layer%d'%(idx+1))
        user_decoding.append(layer(user_decoding[-1]))
    reconstructed_user_emb = user_decoding[-1]
    item_encoding = [item_emb]
    # item encoder
    for idx,s in enumerate(item_encoder_layer):
        layer = Dense(s,activation='relu',kernel_regularizer=regularizers.l2(l2_param),name='item_encoder_layer%d'%(idx+1))
        item_encoding.append(layer(item_encoding[-1]))
    item_vec = item_encoding[-1]
    item_decoder_layer = item_encoder_layer[::-1][1:] + [num_user]
    item_decoding = [item_vec]
    # item decoder
    for idx,s in enumerate(item_decoder_layer):
        layer = Dense(s,activation='relu',kernel_regularizer=regularizers.l2(l2_param),name='item_decoder_layer%d'%(idx+1))
        item_decoding.append(layer(item_decoding[-1]))
    reconstructed_item_emb = item_decoding[-1] 
    uv_feat = Concatenate()([user_vec,item_vec])
    # predictor
    fc_hidden = [uv_feat]
    for idx,s in enumerate(fc_layer):
        layer = Dense(s,activation='relu',kernel_regularizer=regularizers.l2(l2_param),name='full_connected_layer%d'%(idx+1))
        fc_hidden.append(layer(fc_hidden[-1]))
    result = fc_hidden[-1]  
    model = Model([user_emb,item_emb],[reconstructed_user_emb,reconstructed_item_emb,result])
    predictor = Model([user_emb,item_emb],result)
    opt = Adam(lr=lr)
    model.compile(optimizer=opt,loss=[recostruction_loss,recostruction_loss,"mse"])
    model.summary()
    return model,predictor

def generate_vector_batch(uids,iids,rating_matrix):
    batch_uvecs = [] 
    batch_vvecs = []
    for uid,iid in zip(uids,iids):
        uvec = rating_matrix[uid].copy()
        uvec[iid] = 0
        vvec = rating_matrix.T[iid].copy()
        vvec[uid] = 0
        batch_uvecs.append(uvec)
        batch_vvecs.append(vvec)
    return batch_uvecs,batch_vvecs
def main(it):
    args = parse_args()
    logger = get_logger(args.log)
    print(args)
    logger.info(str(args))
    ds,train_size = args.dataset,args.train_size
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
    
    headers = ['uid','iid','rating','timestamp']
    data = pd.read_csv(ds,sep=args.sep,header=None)
    data.columns = headers
    num_user,num_item = data.uid.nunique(),data.iid.nunique()
    print('num_user:{},num_item:{}'.format(num_user,num_item))
    logger.info('num_user:{},num_item:{}'.format(num_user,num_item))
    data['uid'] = data.uid.apply(get_user_idx)
    data['iid'] = data.iid.apply(get_item_idx)
    count = len(data)
    idxs = list(range(count))
    np.random.shuffle(idxs)
    train_size = int(count*train_size)
    train_idxs = idxs[:train_size]
    test_idxs = idxs[train_size:]
    s = np.sum(data.rating.values)
    rates = data.rating.values.copy()
    rates[test_idxs] = 0
    adj_mat = sparse.coo_matrix((rates,(data.uid.values,data.iid.values)),shape=(num_user,num_item))
    rating_matrix = np.array(adj_mat.todense())
    # model parameter
    epoches = args.epoch
    batch_size = args.batch_size
    user_encoder_layer,item_encoder_layer,fc_layer = eval(args.user_encoder_layer),eval(args.item_encoder_layer),eval(args.fc_layer)
    lr,l2_param = args.lr,args.reg
    model,predictor = get_model(num_user,num_item,rating_matrix,user_encoder_layer,item_encoder_layer,fc_layer,lr,l2_param)
    best_rmse,best_mae = 10,10
    for ep in range(epoches):
        se,ae = 0,0
        for i in tqdm(range(0,len(train_idxs),batch_size)):
            if i+batch_size >= len(train_idxs):
                batch_idxs = train_idxs[i:]
            else:
                batch_idxs = train_idxs[i:i+batch_size]
            batch_train_uids = data.uid.values[batch_idxs]
            batch_train_iids = data.iid.values[batch_idxs]
            #batch_train_uvecs = rating_matrix[batch_train_uids]
            #batch_train_ivecs = rating_matrix.T[batch_train_iids]
            batch_train_uvecs,batch_train_ivecs = generate_vector_batch(batch_train_uids,batch_train_iids,rating_matrix)
            batch_train_rates = data.rating.values[batch_idxs][:,None]
            model.train_on_batch([batch_train_uvecs,batch_train_ivecs],[batch_train_uvecs,batch_train_ivecs,batch_train_rates])
            pred_rates = predictor.predict_on_batch([batch_train_uvecs,batch_train_ivecs])
            se += np.sum(np.square(pred_rates-batch_train_rates))
            ae += np.sum(np.abs(pred_rates-batch_train_rates))
        rmse = np.sqrt(se/len(train_idxs))
        mae = ae/len(train_idxs)
        print('Train,ecpoh:{},rmse:{},mae:{}'.format(ep+1,rmse,mae))
        logger.info('Training,ecpoh:{},rmse:{},mae:{}'.format(ep+1,rmse,mae))
        se,ae = 0,0
        for i in tqdm(range(0,len(test_idxs),batch_size)):
            if i+batch_size >= len(test_idxs):
                batch_idxs = test_idxs[i:]
            else:
                batch_idxs = test_idxs[i:i+batch_size]
            batch_test_uids = data.uid.values[batch_idxs]
            batch_test_iids = data.iid.values[batch_idxs]
            #batch_test_uvecs = rating_matrix[batch_test_uids]
            #batch_test_ivecs = rating_matrix.T[batch_test_iids]
            batch_test_uvecs,batch_test_ivecs = generate_vector_batch(batch_test_uids,batch_test_iids,rating_matrix)
            batch_test_rates = data.rating.values[batch_idxs][:,None]
            pred_rates = predictor.predict_on_batch([batch_test_uvecs,batch_test_ivecs])
            #print(pred_rates)
            se += np.sum(np.square(pred_rates-batch_test_rates))
            ae += np.sum(np.abs(pred_rates-batch_test_rates))
        rmse = np.sqrt(se/len(test_idxs))
        mae = ae/len(test_idxs)
        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
        print('Testing,ecpoh:{},rmse:{},mae:{}'.format(ep+1,rmse,mae))
        logger.info('Testing,ecpoh:{},rmse:{},mae:{}'.format(ep+1,rmse,mae))
    logger.info('Best rmse:{},best mae:{}'.format(best_rmse,best_mae))  
if __name__ == '__main__':
    for i in range(5):
        main(i)

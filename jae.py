import tensorflow as tf
import argparse
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='run jae')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--epoch',type=int,default=100,help='trainning epochs')
    parser.add_argument('--dataset',type=str,default='ml100k',help='dataset')
    parser.add_argument('--sep',type=str,default='\t',help='delimiter')
    parser.add_argument('--batch_size',type=int,default=1024,help='delimiter')
    parser.add_argument('--test_rate',type=float,default=0.2,help='test set ratio')
    parser.add_argument('--enlayer',type=str,default='[100,100]',help='encoder layer size')
    parser.add_argument('--fclayer',type=str,default='[100,50,1]',help='full connected layer size')
    return parser.parse_args()
class JAE:
    def __init__(self,sess,num_users,num_items,rating_matrix,layer_sizes,full_layer_sizes,lr):
        self.num_users = num_users
        self.num_items = num_items
        self.sess = sess
        self.lr = lr
        self.rating_table = rating_matrix
        self.layer_sizes = layer_sizes
        self.full_layer_sizes = full_layer_sizes
        self.build_up()
    def build_up(self):
        self.user = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.item = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.rating = tf.placeholder(shape=(None,),dtype=tf.float32)
        self.rating_matrix = tf.constant(self.rating_table)
        uvec = tf.nn.embedding_lookup(self.rating_matrix,self.user)
        ivec = tf.nn.embedding_lookup(tf.transpose(self.rating_matrix),self.item)
        umask = tf.cast(uvec>0,dtype=tf.float32)
        vmask = tf.cast(ivec>0,dtype=tf.float32)
        nuvec = uvec + tf.random_normal(tf.shape(uvec))
        nivec = ivec + tf.random_normal(tf.shape(ivec))
        uencode,iencode = nuvec,nivec
        for i,size in enumerate(self.layer_sizes):
            uencode = tf.layers.dense(uencode,size,activation=tf.nn.relu,name='uencoder_layer_%d'%i)
            iencode = tf.layers.dense(iencode,size,activation=tf.nn.relu,name='iencoder_layer_%d'%i)
        udecode,idecode = uencode,iencode
        dec_layer = self.layer_sizes[::-1][1:]
        for i,size in enumerate(dec_layer):
            udecode = tf.layers.dense(udecode,size,activation=tf.nn.relu,name='udecoder_layer_%d'%i)
            idecode = tf.layers.dense(idecode,size,activation=tf.nn.relu,name='idecoder_layer_%d'%i)
        ruvec = tf.layers.dense(udecode,self.num_items,activation=tf.nn.relu,name='user_reconstruction_layer')
        rivec = tf.layers.dense(udecode,self.num_users,activation=tf.nn.relu,name='item_reconstruction_layer')
        ureconstructionloss = tf.reduce_sum(umask*tf.square(ruvec-uvec),axis=1)
        ireconstructionloss = tf.reduce_sum(vmask*tf.square(rivec-ivec),axis=1)
        f = tf.concat([uencode,iencode],axis=1)
        for i,size in enumerate(self.full_layer_sizes):
            f = tf.layers.dense(f,size,activation=tf.nn.relu,name='full_layer_%d'%i)
        rating_loss = tf.square(tf.reduce_sum(f,axis=1)-self.rating)
        self.loss = tf.reduce_mean(tf.add_n([rating_loss,ureconstructionloss,ireconstructionloss]))
        self.mse = tf.reduce_mean(rating_loss)
        self.mae = tf.reduce_mean(tf.abs(tf.reduce_sum(f,axis=1)-self.rating))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    def train(self,users,items,ratings):
        feed_dict = {self.user:users,self.item:items,self.rating:ratings}
        _,loss,mse,mae = self.sess.run([self.opt,self.loss,self.mse,self.mae],feed_dict=feed_dict)
        return loss,mse,mae
    def test(self,users,items,ratings):
        feed_dict = {self.user:users,self.item:items,self.rating:ratings}
        mse,mae = self.sess.run([self.mse,self.mae],feed_dict=feed_dict)
        return mse,mae
def load(file,test_rate,sep='\t'):
    users,items = {},{}
    count = 0
    with open(file,'r') as fp:
        for line in fp:
            vals = line.strip().split(sep)
            u,i = vals[:2]
            users[u] = users.get(u,0) + 1
            items[i] = items.get(i,0) + 1
            count += 1
    num_users,num_items = len(users),len(items)
    user2idx = {u:idx for idx,u in enumerate(users)}
    item2idx = {it: idx for idx,it in enumerate(items)}
    idxs = list(range(count))
    np.random.shuffle(idxs)
    test_idxs =  idxs[-int(count*test_rate):]
    idx = 0
    rating_matrix = np.zeros((num_users,num_items),dtype=np.float32)
    train,test = [],[] 
    with open(file,'r') as fp:
        for line in fp:
            vals = line.strip().split(sep)
            u,i,r = vals[:3]
            u,i,r = user2idx[u],item2idx[i],float(r)
            if idx in test_idxs:
                test.append([u,i,r])
            else:
                train.append([u,i,r])
                rating_matrix[u][i] = r
            idx += 1
    return train,test,num_users,num_items,rating_matrix
def get_batch(data,batch_size=256):
    batch_users,batch_items,batch_ratings = [],[],[]
    for u,i,r in data:
        batch_users.append(u)
        batch_items.append(i)
        batch_ratings.append(r)
        if len(batch_users) == batch_size:
            yield batch_users,batch_items,batch_ratings
            batch_users,batch_items,batch_ratings = [],[],[]
    if len(batch_users) > 0:
        yield batch_users,batch_items,batch_ratings
def main():
    args = parse_args()
    enlayer_size = eval(args.enlayer)
    fclayer_size = eval(args.fclayer)
    # load dataset
    train,test,num_users,num_items,rating_matrix = load(args.dataset,args.test_rate,args.sep)
    sess = tf.Session()
    model = JAE(sess,num_users,num_items,rating_matrix,enlayer_size,fclayer_size,args.lr)
    #sess.run(tf.assign(model.rating_matrix,rating_matrix))
    init = tf.global_variables_initializer()
    sess.run(init)
    best_rmse,best_mae,batch_epoch = 0,0,0
    for epoch in range(args.epoch):
        loss,mse,mae = 0,0,0
        for users,items,ratings in get_batch(train,args.batch_size):
            batch_loss,batch_mse,batch_mae = model.train(users,items,ratings)
            size = len(users)
            loss += batch_loss*size
            mse += batch_mse*size
            mae += batch_mae*size
        rmse = np.sqrt(mse/len(train))
        mae = mae / len(train)
        loss = loss/len(train)
        print('train epoch:{},loss:{},rmse:{},mae:{}'.format(epoch+1,loss,rmse,mae))
        rmse,mae = 0,0
        for users,items,ratings in get_batch(test,args.batch_size):
            batch_mse,batch_mae = model.test(users,items,ratings)
            mse += batch_mse * len(users)
            mae += batch_mae * len(users)
        rmse = np.sqrt(mse/len(test))
        mae = mae / len(test)
        print('test epoch:{},rmse:{},mae:{}'.format(epoch+1,rmse,mae))
        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
            best_epoch = epoch +1 
    print('complete, best rmse:{},best mae:{},taken at epoch:{}'.format(best_rmse,batch_mae,best_epoch))
if __name__ == '__main__':
    main()

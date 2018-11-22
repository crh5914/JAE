import tensorflow as tf
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='run jae')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--dataset',type=str,default='ml100k',help='dataset')
    parser.add_argument('--test_rate',type=float,default=0.2,help='test set ratio')
    parser.add_argument('--enlayer',type=str,default='[500,100]',help='encoder layer size')
    parser.add_argument('--fclayer',type=str,default='[100,50,1]',help='full connected layer size')
    return parser.parse_args()
class JAE:
    def __init__(self,sess,num_users,num_items,layer_sizes,full_layer_sizes):
        self.num_users = num_users
        self.num_items = num_items
        self.sess = sess
        self.layer_sizes = layer_sizes
        self.full_layer_sizes = full_layer_sizes
    def build_up(self):
        self.user = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.item = tf.placeholder(shape=(None,),dtype=tf.int32)
        self.rating = tf.placeholder(shape=(None,),dtype=tf.float32)
        self.rating_matrix = tf.Variable(tf.random_normal(shape=(self.num_users,self.num_items),stddev=0.01),trainable=False)
        uvec = tf.nn.embedding_lookup(self.rating_matrix,self.user)
        ivec = tf.nn.embedding_lookup(tf.transpose(self.rating_matrix),self.item)
        umask = tf.cast(uvec>0,dtype=tf.float32)
        vmask = tf.cast(ivec>0,dtype=tf.float32)
        nuvec = uvec + tf.random_normal(tf.shape(uvec))
        nivec = ivec + tf.random_normal(tf.shape(ivec))
        uencode,iencode = nuvec,nivec
        for i,size in enumerate(self.layer_sizes):
            uencode = tf.dense(uencode,size,name='uencoder_layer_%d'%i)
            iencode = tf.dense(iencode,size,name='iencoder_layer_%d'%i)
        udecode,idecode = uencode,iencode
        dec_layer = self.layer_sizes[::-1][1:]
        for i,size in enumerate(dec_layer):
            udecode = tf.dense(udecode,size,name='udecoder_layer_%d'%i)
            idecode = tf.dense(idecode,size,name='idecoder_layer_%d'%i)
        ruvec = tf.dense(udecode,self.num_items,name='user_reconstruction_layer')
        rivec = tf.dense(udecode,self.num_users,name='item_reconstruction_layer')
        ureconstructionloss = tf.reduce_sum(umask*tf.square(ruvec-uvec),axis=1)
        ireconstructionloss = tf.reduce_sum(imask*tf.square(rivec-ivec),axis=1)
        f = tf.concat([uencode,vencode],axis=1)
        for i,size in enumerate(self.full_layer_sizes):
            f = tf.layers.dense(f,size,name='full_layer_%d'%i)
        rating_loss = tf.square(tf.reduce_sum(f,axis=1)-self.rating)
        self.loss = tf.reduce_mean(tf.add_n([rating_loss,ureconstructionloss,ireconstructionloss])
        self.mse = tf.reduce_mean(rating_loss)
        self.mae = tf.reduce_mean(tf.abs(tf.reduce_sum(f,axis=1)-self.rating))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    def train(self):
        pass
    def test(self):
        pass
def main():
    args = parse_args()
    enlayer_size = eval(args.enlayer)
    fclayer_size = eval(args.fclayer)
    # load dataset
    train,test = load(file,args.test_rate)

if __name__ == '__main__':
    main()

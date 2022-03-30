import math
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras.initializers
from keras import Input, Model, optimizers
from keras.constraints import non_neg
from keras.layers import Embedding, Dot, Flatten, Dropout
from keras.models import load_model
from keras.regularizers import l2
import keras.backend as K
from keras.callbacks import LearningRateScheduler
tf.config.run_functions_eagerly(True)

class KerasSequenceData(tf.keras.utils.Sequence):
    """
    Keras Sequence Data Class.
    """
    def __init__(self, data_instances, user_ids, item_ids, batch_size):
        self.size = len(data_instances)
        if self.size <= 0:
            raise ValueError("empty data")

        user_ids_map = {uid: i for i, uid in enumerate(user_ids)}
        item_ids_map = {iid: i for i, iid in enumerate(item_ids)}
        self.x = np.zeros((self.size, 2))
        self.y = np.zeros((self.size, 1))
        self._keys = []
        for index, rows in data_instances.iterrows():
            self._keys.append(rows[0])
            uid = rows[1]
            iid = rows[2]
            rate = float(rows[3])
            self.x[index] = [user_ids_map[uid], item_ids_map[iid]]
            self.y[index] = rate
        self.batch_size = batch_size if batch_size > 0 else self.size

    def __getitem__(self, index):
        """
        Gets batch at position `index`.
        :param index: position of the batch in the Sequence.
        :return: A batch
        """
        start = self.batch_size * index
        end = self.batch_size * (index + 1)
        return [self.x[start: end, 0],
                self.x[start: end, 1]], self.y[start: end]

    def __len__(self):
        """Number of batch in the Sequence.
        "return: The number of batches in the Sequence.
        """
        return int(np.ceil(self.size / float(self.batch_size)))

    def get_keys(self):
        """
        Return keys of data.
        :return: keys of data.
        """
        return self._keys


class KerasSeqDataConverter:
    """
    Keras Sequence Data Converter.
    """
    @staticmethod
    def convert(data, user_ids, item_ids, batch_size):
        return KerasSequenceData(data, user_ids, item_ids, batch_size)

def Recmand_model(num_user, num_item, k):
    user_input_layer = Input(shape=(1,), dtype='int32', name='user_input')
    user_embedding_layer = Embedding(
        input_dim=num_user,
        output_dim=k,
        input_length=1,
        name='user_embedding',
        embeddings_regularizer=l2(0.00015),
        embeddings_initializer='uniform')(user_input_layer)
    user_embedding_layer = Flatten(name='user_flatten')(user_embedding_layer)

    # item embedding
    item_input_layer = Input(shape=(1,), dtype='int32', name='item_input')
    item_embedding_layer = Embedding(
        input_dim=num_item,
        output_dim=k,
        input_length=1,
        name='item_embedding',
        embeddings_regularizer=l2(0.00015),
        embeddings_initializer='uniform')(item_input_layer)
    item_embedding_layer = Flatten(name='item_flatten')(item_embedding_layer)

    rmsprop = optimizers.rmsprop_v2.RMSprop(learning_rate=0.05,momentum=0.9,decay=0.0)
    dot_layer = Dot(axes=-1,
                    name='dot_layer')([user_embedding_layer,
                                       item_embedding_layer])
    model = Model(
        inputs=[user_input_layer, item_input_layer], outputs=[dot_layer])
    losses = getattr(tf.keras.losses, 'mse')
    model.compile(optimizer=rmsprop,loss=losses, metrics=['accuracy',['mae', 'mse']])
    model.summary()
    return model

def train(train_data,num_user,num_item):
    # def scheduler(epoch):
    #     # 每隔100个epoch，学习率减小为原来的1/10
    #     if epoch % 100 == 0 and epoch != 0:
    #         lr = K.get_value(model.optimizer.lr)
    #         K.set_value(model.optimizer.lr, lr * 0.5)
    #         print("lr changed to {}".format(lr * 0.5))
    #     return K.get_value(model.optimizer.lr)

    data = KerasSeqDataConverter().convert(train_data,all_user,all_item,batch_size=0)
    model = Recmand_model(num_user,num_item,10)
    # reduce_lr = LearningRateScheduler(scheduler) callbacks=[reduce_lr]
    model.fit(data,batch_size = 1024,epochs =200)
    model.save("model.h5")

if __name__ == '__main__':
    train_data = pd.read_csv('E:\资料\研究生\大创\dataset\LastFM\\train.csv',
                             header=None, names=['id','user','item','score'],error_bad_lines=False)
    test_data = pd.read_csv('E:\资料\研究生\大创\dataset\LastFM\\test.csv',
                            header=None,names=['id','user','item','score'],error_bad_lines=False)
    all_user = np.unique(train_data['user'])
    num_user = len(all_user)
    all_item = np.unique(train_data['item'])
    num_item = len(np.unique(train_data['item']))
    train(train_data,num_user,num_item)
    # test(train_data,test_data,all_user,all_item)
    model = load_model('model.h5')
    data = KerasSeqDataConverter().convert(test_data, all_user , all_item ,batch_size=0)
    list_ = model.evaluate(data,verbose=1)
    pred_equal = model.predict(data)
    cm = confusion_matrix(data.y, np.round(pred_equal), labels=[True, False])
    sum_ = cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
    print("tn{0}, fp{1}, fn{2}, tp{3}".format(cm[0][0]/sum_ , cm[0][1]/sum_,cm[1][0]/sum_,cm[1][1]/sum_))

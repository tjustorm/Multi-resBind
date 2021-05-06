import sys
import os
import numpy as np
import h5py
import scipy.io

np.random.seed(7)  # for reproducibility

import tensorflow as tf
# tf.python.control_flow_ops = tf

from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import concatenate
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import keras.backend as K
import matplotlib as mpl
import math
mpl.use('Agg')

################################################################################
# Accessry functions
################################################################################
def create_class_weight(labels_dict, total, mu=0.15):
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight

################################################################################
# loading data
#
# Input: path to file (consist of train, valid and test data)
#
################################################################################

def load_data_struc(path_to_data):
    data = h5py.File(path_to_data, 'r')
    X_train_struc = np.transpose(np.array(data['train_in_annotation']),axes=(0,2,1))
    y_train = np.array(data['train_out'])

    X_valid_struc = np.transpose(np.array(data['valid_in_annotation']),axes=(0,2,1))
    y_valid = np.array(data['valid_out'])

    X_test_struc = np.transpose(np.array(data['test_in_annotation']),axes=(0,2,1))
    y_test = np.array(data['test_out'])

    data.close()
    return X_train_struc, y_train, X_valid_struc, y_valid, X_test_struc, y_test


################################################################################
# Creating model
#
# Input: path to file (consist of train, valid and test data)
#
################################################################################

### Model only using structure data
def create_model1(num_task, input_len):
    K.clear_session()
    tf.set_random_seed(5005)
    dim = 6
    input = input_len

    nb_f= [90, 100]
    f_len= [7, 7]
    p_len= [4, 10]
    s= [2, 5]

    input = Input(shape=(input, dim), name="input")

    conv1 = Conv1D(filters=nb_f[0], kernel_size=f_len[0],
                        padding='valid', activation="relu", name="conv1")(input)
    pool1 = MaxPooling1D(pool_size=p_len[0], strides=s[0],
                              name="pool1")(conv1)
    drop1 = Dropout(0.25, name="drop1")(pool1)

    conv2 = Conv1D(filters=100, kernel_size=5, padding='valid',
                         activation="relu", name="conv2")(drop1)
    conv2_pool = MaxPooling1D(pool_size=10, strides=5)(conv2)
    conv2_drop = Dropout(0.25)(conv2_pool)
    conv2_flat = Flatten()(conv2_drop)

    hidden1 = Dense(250, activation='relu', name="hidden1")(conv2_flat)
    output = Dense(num_task, activation='sigmoid', name="output")(hidden1)
    model = Model(inputs=input, outputs=output)
    print(model.summary())
    return model

################################################################################
# Training the model
#
# Input: path to file (consist of train, valid and test data)
#
################################################################################

def train_diff_model(data_path, model_funname, res_path, model_name, num_task,
                     input_len, num_epoch, batchsize,
                     model_path="./weights.hdf5", plot=False):
    print ('creating model')
    if isinstance(model_funname, str):
        dispatcher = {'create_model1': create_model1}
        try:
            model_funname = dispatcher[model_funname]
        except KeyError:
            raise ValueError('invalid input')
    model = model_funname(num_task, input_len)
    print ('compiling model')
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=['accuracy', precision, recall])
    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1,
                                   save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    tb = TensorBoard(log_dir='./', histogram_freq=0, write_graph=True,
                     write_images=False)

    print ('loading data')
    X_train_struc, y_train, X_valid_struc, y_valid, X_test_struc, y_test = load_data_struc(data_path)

    print("input shape:", X_train_struc.shape)

    total = y_train.shape[0]
    labels_dict = dict(
        zip(range(num_task), [sum(y_train[:, i]) for i in range(num_task)]))
    class_weight = create_class_weight(labels_dict, total, mu=0.5)

    print ('fitting the model')
    history = model.fit(X_train_struc, y_train,
                        epochs=num_epoch, batch_size=batchsize,
                        validation_data=(X_valid_struc, y_valid),
                        class_weight=class_weight, verbose=2,
                        callbacks=[checkpointer, earlystopper, tb])

    print ('saving the model')
    model.save(os.path.join(res_path, model_name + ".h5"))

    print ('testing the model')
    score = model.evaluate(X_test_struc, y_test)

    print(model.metrics_names)

    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(score[i]))

    print("{}: {:.2f}".format(model.metrics_names[0], score[0]))
    print("{}: {:.2f}".format(model.metrics_names[1], score[1]))
    print("{}: {:.2f}".format(model.metrics_names[2], score[2]))


################################################################################
# Testing the model
#
# Input: path to file (consist of train, valid and test data)
#
################################################################################

def test_model(output_path, data_path, model_path, model_funname, res_path,
               model_name, input_len, num_task):
    print('test the model and plot the curve')
    model = load_model(os.path.join(res_path, model_name + ".h5"),
                       custom_objects={'precision': precision,
                                       'recall': recall})
    data = h5py.File(data_path, 'r')
    X_test_struc = np.transpose(np.array(data['test_in_annotation']), axes=(0, 2, 1))
    y_test = np.array(data['test_out'])
    data.close()

    print ('predicting on test data')
    y_pred = model.predict(X_test_struc, verbose=1)
    model.evaluate(X_test_struc, y_test)

    print ("saving the prediction to " + output_path)
    f = h5py.File(output_path, "w")
    f.create_dataset("y_pred", data=y_pred)
    f.close()


################################################################################
# custume metric####
################################################################################

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


################################################################################
# main function
################################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-nt', dest='num_task', default=None, type=int,
                        help='number of tasks')
    parser.add_argument('-l', dest='input_len', default=None, type=int,
                        help='input length')
    parser.add_argument('-ne', dest='num_epoch', default=None, type=int,
                        help='number of epochs')
    parser.add_argument('-bs', dest='batchsize', default=None, type=int,
                        help='Batch size')
    parser.add_argument('-dp', dest='data_path', default=None, type=str,
                        help='path to the data')
    parser.add_argument('-op', dest='output_path', default=None, type=str,
                        help='path to the output')
    parser.add_argument('-mp', dest='model_path', default=None, type=str,
                        help='path to the model')
    parser.add_argument('-pp', dest='prediction_path', default=None, type=str,
                        help='path to the prediction')
    parser.add_argument('-fun', dest='model_funname', default=None, type=str,
                        help='name of the model')
    parser.add_argument('-name', dest='model_name', default=None, type=str,
                        help='name of the model')
    parser.add_argument('-t', dest='test', default=False, type=bool,
                        help='test the model')
    args = parser.parse_args()

    dispatcher = {'create_model1': create_model1}
    try:
        funname = dispatcher[args.model_funname]
    except KeyError:
        raise ValueError('invalid input')

    train_diff_model(data_path=args.data_path, model_funname=funname,
                     res_path=args.output_path, model_name=args.model_name,
                     num_task=args.num_task, input_len=args.input_len,
                     num_epoch=args.num_epoch, batchsize=args.batchsize,
                     model_path=args.model_path)
    test_flag = args.test
    print("test flag:", test_flag)
    if test_flag:
        print("testing the model and plot the curves")
        test_model(output_path=args.prediction_path, data_path=args.data_path,
                   model_path=args.model_path, model_funname=funname,
                   res_path=args.output_path, model_name=args.model_name,
                   input_len=args.input_len, num_task=args.num_task)
if __name__ == '__main__':
    main()

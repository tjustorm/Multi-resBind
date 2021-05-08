from __future__ import print_function
import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from models import residualbind
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from keras.callbacks import Callback
import numpy as np
import tensorflow as tf
import os
import h5py
import math
np.random.seed(7)  # for reproducibility

################################################################################
# generate AsymmetricLoss(
################################################################################
def AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.5):
    gamma_neg = K.constant(gamma_neg, tf.float32)
    gamma_pos = K.constant(gamma_pos, tf.float32)

    def focal_loss_function(y_true, y_pred):
        """
        Focal loss for multi-label classification.
        https://arxiv.org/abs/1708.02002
        Arguments:
            y_true {tensor} : Ground truth labels, with shape (batch_size, number_of_classes).
            y_pred {tensor} : Model's predictions, with shape (batch_size, number_of_classes).
        Keyword Arguments:
            class_weights {list[float]} : Non-zero, positive class-weights. This is used instead
                                          of Alpha parameter.
            gamma {float} : The Gamma parameter in Focal Loss. Default value (2.0).
            class_sparsity_coefficient {float} : The weight of True labels over False labels. Useful
                                                 if True labels are sparse. Default value (1.0).
        Returns:
            loss {tensor} : A tensor of focal loss.
        """

        predictions_0 = (1.0 - y_true) * y_pred
        predictions_1 = y_true * y_pred

        cross_entropy_0 = (1.0 - y_true) * (-K.log(K.clip(1.0 - predictions_0, K.epsilon(), 1.0 - K.epsilon())))
        cross_entropy_1 = y_true *(-K.log(K.clip(predictions_1, K.epsilon(), 1.0 - K.epsilon())))

        cross_entropy = cross_entropy_1 + cross_entropy_0

        weight_1 = K.pow(K.clip(1.0 - predictions_1, K.epsilon(), 1.0 - K.epsilon()), gamma_pos)
        weight_0 = K.pow(K.clip(predictions_0, K.epsilon(), 1.0 - K.epsilon()), gamma_neg)

        weight = weight_0 + weight_1

        focal_loss_tensor = weight * cross_entropy

        return K.mean(focal_loss_tensor, axis=1)

    return focal_loss_function


################################################################################
# Accessry functions
################################################################################
def create_class_weight(labels_dict,total,mu=0.15):
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight

################################################################################
# loading data
#
# Input: path to file (consist of train, valid and test data)
#
################################################################################
def load_data(path_to_data):
    data = h5py.File(path_to_data, 'r')

    X_train_seq = np.transpose(np.array(data['train_in_seq']), axes=(0, 2, 1))
    X_train_region = np.transpose(np.array(data['train_in_region']),axes=(0, 2, 1))
    X_train = np.concatenate((X_train_seq, X_train_region[:,50:200,:]), axis=2)
    y_train = np.array(data['train_out'])

    X_valid_seq = np.transpose(np.array(data['valid_in_seq']), axes=(0, 2, 1))
    X_valid_region = np.transpose(np.array(data['valid_in_region']),
                                  axes=(0, 2, 1))
    X_valid = np.concatenate((X_valid_seq, X_valid_region[:,50:200,:]), axis=2)
    y_valid = np.array(data['valid_out'])

    X_test_seq = np.transpose(np.array(data['test_in_seq']), axes=(0, 2, 1))
    X_test_region = np.transpose(np.array(data['test_in_region']),
                                  axes=(0, 2, 1))
    X_test = np.concatenate((X_test_seq, X_test_region[:, 50:200, :]), axis=2)
    y_test = np.array(data['test_out'])

    data.close()
    return X_train, y_train, X_valid, y_valid, X_test, y_test

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

def mAP(y_true,y_pred):
	num_classes = 27
	average_precisions = []
	relevant = K.sum(K.round(K.clip(y_true, 0, 1)))
	tp_whole = K.round(K.clip(y_true * y_pred, 0, 1))
	for index in range(num_classes):
		temp = K.sum(tp_whole[:,:index+1],axis=1)
		average_precisions.append(temp * (1/(index + 1)))
	AP = keras.layers.Add()(average_precisions) / relevant
	mAP = K.mean(AP,axis=0)
	return mAP

################################################################################
# training Convolutional Block Attention Module
################################################################################
# Training parameters
batch_size = 128
epochs = 40
num_task = 11
base_model = 'residualbind'
# Choose what attention_module to use: cbam_block / se_block / None
attention_module = None
model_type = base_model if attention_module == None else base_model+'_'+attention_module

# Load the data.
print ('loading data')
path_data = "./Data/data_RBPshigh.h5"
x_train, y_train, x_valid, y_valid, x_test, y_test=load_data(path_data)

# Input image dimensions.
input_shape = x_train.shape[1:]

print('x_train shape:', x_train.shape)
print('x_valid shape:', x_valid.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_valid shape:', y_valid.shape)
print('y_test shape:', y_test.shape)

print ('creating model')
model = residualbind.ResidualBind(input_shape=input_shape,num_class=num_task)
print ('compiling model')
adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
model.compile(loss = 'binary_crossentropy',
              optimizer= adam,
              metrics=['accuracy',precision,recall])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'residualbind_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# generate the class-aware weights
total=y_train.shape[0]
labels_dict=dict(zip(range(num_task),[sum(y_train[:,i]) for i in range(num_task)]))
class_weight=create_class_weight(labels_dict,total,mu=0.5)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only= True)

callbacks = [checkpoint]
history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_valid, y_valid),
              shuffle=True,
              class_weight=class_weight,
              verbose=2,
              callbacks=callbacks)

# Score trained model.
print ('testing the model')
score = model.evaluate(x_test, y_test, verbose=2)
print(model.metrics_names)
for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i]) + ": " + str(score[i]))

print ('print model losses graph')
pdf_name = '%s_model_losses' % model_type
f = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
f.savefig(pdf_name, bbox_inches='tight')

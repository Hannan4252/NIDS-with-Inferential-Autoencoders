from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
import keras
from tensorflow.keras import regularizers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,precision_score,recall_score
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import SVC

nn = 3

labels_kdd = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
'num_access_files', 'num_outbound_cmds', 'is_host_login',
'is_guest_login', 'count', 'srv_count', 'serror_rate',
'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level']

train = pd.read_csv('https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.csv')
test = pd.read_csv('https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv')

train.columns , test.columns = labels_kdd , labels_kdd

train = train.drop(columns =['protocol_type', 'service', 'flag'])
test = test.drop(columns =['protocol_type', 'service', 'flag'])

def under_sample(df, n=5000, random_state=0xFFFF):
    tmp2 = pd.DataFrame(columns=df.columns)
    attacks, counts = np.unique(df['attack_type'], return_counts=True)
    for each_attack in attacks:
        tmp = df[df['attack_type'] == each_attack]
        if each_attack == 'normal':
            n1= n*(len(attacks)-1)
            samples = tmp.sample(n =n1, random_state= random_state ,replace=True)
        else:
            samples = tmp.sample(n =n, random_state= random_state ,replace=True)
        
        tmp2 = tmp2.append(samples)
    tmp2 = tmp2.sample(frac=1, random_state=random_state)
    return tmp2
    
    
def remove_attacks(df, attack_types):
    tmp =[]
    for each in attack_types:
        tmp.append(df['attack_type'] != each)
    tmp = tuple(tmp)
    return np.logical_and.reduce(tmp)

test = under_sample(test, 300)

svm_params =[0.1 , 1, 10]
cvae_params = [5, 10, 20 ]
scoress = []

for cc in range(len(svm_params)):
    scores_dt = []
    scores_svm = []
    scores_ae = []
    scores_vae = []
    scores_cvae =[]
    for n in range(nn):
        random_i = int(np.random.randint(0,10000, size=1, dtype=int))
        test = under_sample(test, 300, random_state=random_i)

        train_x = np.asarray( train.drop(columns = 'attack_type') )
        test_x = np.asarray( test.drop(columns = 'attack_type') )

        train_y = np.asarray(train['attack_type']) != 'normal'
        test_y = np.asarray(test['attack_type']) != 'normal'

        scaler = StandardScaler()
        scaler.fit(train_x)

        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

        train_x_normal = train_x[~train_y]
        train_y_normal = train_y[~train_y]

        train_x_anomaly = train_x[train_y]

        full_train_x = train_x

        train_x, val_x, labels_condition, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=random_i)
        labels_condition =np.asarray(labels_condition).astype('int')
        labels_condition[labels_condition==1] = cvae_params[cc]




        #####SVM

        clf = SVC(C=svm_params[cc])
        clf.fit(full_train_x,train_y)
        predicted = clf.predict(test_x)
        scores_svm.append(f1_score(test_y,predicted)*100)

        ####model params

        original_dim = train_x.shape[1]
        input_shape = (original_dim, )
        intermediate_dim = int(original_dim/2)
        batch_size = 500
        latent_dim = int(original_dim/4)
        epochs = 500


        # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
        def sampling(args):
            """Reparameterization trick by sampling from an isotropic unit Gaussian.

            # Arguments
                args (tensor): mean and log of variance of Q(z|X)

            # Returns
                z (tensor): sampled latent vector
            """

            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean = 0 and std = 1.0
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        def test_loss(inputss, outputss , mean, log_var):
            reconstruction_losses = np.mean((inputss-outputss)**2, axis = -1)
            reconstruction_losses *= original_dim
            kl_losses = 1 + log_var - np.square(mean) - np.exp(log_var)
            kl_losses = np.sum(kl_losses, axis = -1)
            kl_losses *= -0.5
            vae_losses = reconstruction_losses + kl_losses
            return vae_losses



        def train_loss(inputs,outputs):
        #    labels1 = inputs1[1]
        #     inputs =inputs1[0]
        #     outputs = outputs1
            #labels1 =tf.reshape(labels1, (11,-1))
        #     normal_labels = tf.where(tf.equal(labels,False))
        #     anomaly_labels = tf.where(tf.equal(labels,True))


        #     inputs =tf.keras.backend.reshape (tf.keras.backend.gather(inputs1[0], normal_labels).re(-1,original_dim)
        #     outputs = tf.keras.backend.gather(outputs1, normal_labels).reshape(-1,original_dim)


            reconstruction_loss = K.mean(K.pow(inputs-outputs, 2), axis = -1)
            reconstruction_loss *= original_dim
            #tf.print(reconstruction_loss.shape)
            kl_loss = 1 + z_log_var - K.square(z_mean-labels) - K.exp(z_log_var)
            kl_loss = K.mean(kl_loss, axis=-1)
            kl_loss *= -0.5
            normal_vae_loss = K.mean(reconstruction_loss + kl_loss)

        #     inputs = tf.keras.backend.gather(inputs1[0], anomaly_labels)
        #     outputs = tf.keras.backend.gather(outputs1, anomaly_labels)
        #     reconstruction_loss = K.mean(K.pow(inputs-outputs, 2), axis = -1)
        #     reconstruction_loss *= original_dim
        #     #tf.print(reconstruction_loss.shape)
        #     kl_loss = 1 + z_log_var - K.square(z_mean-10) - K.exp(z_log_var)
        #     kl_loss = K.sum(kl_loss, axis=-1)
        #     kl_loss *= -0.5
        #     anomaly_vae_loss = 0.5 *K.mean(reconstruction_loss + kl_loss)

            return normal_vae_loss

        # VAE model = encoder + decoder
    # build encoder model

        inputs = Input(shape=input_shape, name='encoder_input')

        labels = Input(shape = (1,), name='label_input')
        #labels = K.tile(labels, (batc,))

        x = Dense(intermediate_dim, activation='relu' ,     kernel_regularizer=regularizers.l2(1e-4),
                  bias_regularizer=regularizers.l2(1e-4) )(inputs)
        z_mean = Dense(latent_dim, name='z_mean',     kernel_regularizer=regularizers.l2(1e-4),
                       bias_regularizer=regularizers.l2(1e-4) )(x)
        z_log_var = Dense(latent_dim, name='z_log_var',     kernel_regularizer=regularizers.l2(1e-4), 
                          bias_regularizer=regularizers.l2(1e-4) , kernel_initializer='zeros',
                          bias_initializer= 'zeros' )(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')


        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu',     kernel_regularizer=regularizers.l2(1e-4),
                  bias_regularizer=regularizers.l2(1e-4) )(latent_inputs)
        outputs = Dense(original_dim ,    kernel_regularizer=regularizers.l2(1e-4),
                        bias_regularizer=regularizers.l2(1e-4))(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        cvae = Model([inputs, labels], outputs, name='vae_mlp')

        optimizer = keras.optimizers.Adam(learning_rate=0.0001)

        #vae.add_loss(vae_loss)
        cvae.compile(optimizer=optimizer , loss=train_loss)

        #train the model
        callbacks = [
            tf.keras.callbacks.EarlyStopping( monitor='val_loss',min_delta= 0.001, patience=5, verbose=0, mode='auto',
                                             baseline=None, restore_best_weights=True),  ]
        history = cvae.fit([train_x,labels_condition] , train_x, epochs=500, batch_size= 50 , callbacks= callbacks,
                          verbose =0, validation_data=([val_x,val_y], val_x), shuffle= True)

        estimated_vae_output = cvae.predict([val_x, np.zeros(len(val_x))] , batch_size= batch_size)
        mean_latent_test, std_latent_test, _ = encoder.predict(val_x, batch_size= batch_size)

        anomaly_detected_loss = test_loss(val_x,estimated_vae_output, mean_latent_test, std_latent_test)

        xx = range(1,8000)
        f1_scores=[]
        for i in xx:
            anomaly_detected = anomaly_detected_loss > i
            f1_scores.append(f1_score(val_y.astype('bool'), anomaly_detected))
        threshold = np.argmax(f1_scores)+xx[0]

        estimated_vae_output = cvae.predict([test_x, np.zeros(len(test_x))] , batch_size= batch_size)
        mean_latent_test, std_latent_test, test_z = encoder.predict(test_x, batch_size= batch_size)

        anomaly_detected_loss = test_loss(test_x,estimated_vae_output, mean_latent_test, std_latent_test)
        anomaly_detected = anomaly_detected_loss > threshold
        scores_cvae.append(f1_score(test_y, anomaly_detected)*100)





    print("f1 scores for SVM\n", scores_svm)
    print(" mean f1 scores for SVM is ", np.asarray(scores_svm).mean())
    print(" std of f1 scores for SVM is ", np.asarray(scores_svm).std())
    print('\n')


    print("f1 scores for conditional variational autoencoders \n", scores_cvae)
    print(" mean f1 scores for conditional variational autencoders is ", np.asarray(scores_cvae).mean())
    print(" std of f1 scores for conditional variational autoencoders is ", np.asarray(scores_cvae).std())
    print('\n')
    scoress.append([float(np.asarray(scores_svm).mean()), float(np.asarray(scores_cvae).mean())])
    
    
    

import keras
import os 
import random
import numpy as np
import kapre

from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dense, Reshape, Permute
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.pooling import GlobalAveragePooling1D
from keras import metrics
from utils import util
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.core import Dropout 
from keras import backend as K
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from keras.layers import Add, Average, Concatenate, Multiply, Lambda, BatchNormalization, Activation, TimeDistributed
from keras import regularizers
from kapre.time_frequency import Melspectrogram, Spectrogram


# def Conv2D_TD_BAC_mel(filter_size):
# 	def f(inputs):
# 		x = BatchNormalization()(inputs)
# 		x = Activation('relu')(x)
# 		x = Conv2D(filter_size, (1, 1), padding='same', activation='linear')(x)
# 		x = BatchNormalization()(x)
# 		x = Activation('relu')(x)
# 		x = Conv2D(filter_size, (3, 3), padding='same', activation='linear')(x)
		
# 		se = GlobalAveragePooling2D()(x)
# 		se = Reshape((1,1,filter_size))(se)
# 		se = Dense(filter_size//2, activation='relu')(se)
# 		se = Dense(filter_size, activation='sigmoid')(se)
# 		x = Multiply()([x, se])
		
# 		x = Concatenate()([x,inputs])
# 		return x
# 	return f

# def build_model_mel(input_shape, n_class):   
# 	# build model...
# 	print 'build model...'

# 	sr = 32000
# 	n_fft=1024
# 	n_hop=n_fft/8

# 	num_class = n_class
# 	n_feature = 64

# 	inputs = Input(shape=(None,))
# 	inputs2 = Reshape((-1,1))(inputs)
# 	inputs2 = BatchNormalization()(inputs2)
# 	inputs2 = Reshape((1,-1))(inputs2)
	
# 	melspec=Melspectrogram(n_dft=n_fft, n_hop=n_hop, input_shape=(1,None),
# 							 padding='same', sr=sr, n_mels=n_feature,
# 							 fmin=0.0, fmax=sr/2, power_melgram=1.0,
# 							 return_decibel_melgram=True, trainable_fb=False,
# 							 trainable_kernel=False,
# 							 name='trainable_stft')(inputs2)

# 	x = melspec #(64,none,1)
# 	x = Permute((3,2,1))(x) #(1,none,64)
# 	x = BatchNormalization()(x)
# 	x = Permute((3,2,1))(x)

# 	x1 = Conv2D(15, (3, 3), padding='same', activation='linear')(x)
	
# 	x = Concatenate()([x,x1])
	
# 	for i in range(8):    
# 		x = Conv2D_TD_BAC_mel(np.minimum(512,16*(2**i)))(x)
# 		x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
								
# 	x = BatchNormalization()(x)
# 	x = Activation('relu')(x) 
	
# 	outputsx = [0,0,0,0,0,0,0,0]
# 	for i in range(8):
# 		outputs = x
# 		outputs = Dense(n_class, activation='linear')(outputs)
# 		outputs = GlobalAveragePooling2D()(outputs)
# 		outputsx[i] = Activation('softmax')(outputs) 
# 	outputsx = Average()([o for o in outputsx])

# 	outputs  = outputsx

# 	model = Model(inputs=[inputs], outputs=[outputs])    
# 	return model                  

def ccategorical_crossentropy(y_true, y_pred):
	
	y_true_v = K.greater(K.sum(y_true, axis=-1),1.1)
	y_true = y_true%1
	
	y_pred = K.clip(y_pred, K.epsilon(), 1)
	loss = -K.sum(y_true*K.log(y_pred),axis=-1)

	m = K.max(loss)*0.8
	
	loss = loss 
	
	el = 1-(K.cast(K.greater(loss,m),'float32')*K.cast(y_true_v,'float32'))
	loss = loss*el
	return loss                      

def load_model(filename, custom_objects = None):
	return keras.models.load_model(filename, custom_objects={'Melspectrogram':kapre.time_frequency.Melspectrogram, 'Spectrogram':kapre.time_frequency.Spectrogram, 'ccategorical_crossentropy':ccategorical_crossentropy})

def predict(m, X, n_class):
	n_file = len(X)
	Z = np.zeros((n_file,n_class))
	
	num_ensemble = 1
	
	for i in range(n_file):
		x = np.reshape(X[i],(1,len(X[i])))
		xorig = x
	
		if np.shape(x)[1]>0:
			maxx = np.max(np.abs(x))
			for k in range(num_ensemble):
				if np.shape(x)[1] < 64000:
					x = np.zeros((1,64000))
					offset=random.randrange(0,64000-np.shape(xorig)[1]+1)
					x[0,offset:offset+np.shape(xorig)[1]] = xorig[0,:]
					
				Z[i,:] = Z[i,:] + np.log(1e-20+m.predict( x*(1./(k+1) / np.maximum(maxx,1e-20))))
			Z[i,:] = Z[i,:]/num_ensemble
			# print n_file-i
		else:
			Z[i,:] = Z[i-1,:]*0+(1./41)
	return np.exp(Z)
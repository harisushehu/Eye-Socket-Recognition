# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:44:09 2020

@author: harisushehu
"""

import cv2
import glob
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

#KTF.set_session(sess)

n_folds= 10
model_history = []
BATCH_SIZE = 32
EPOCHS = 250
NUM_CLASSES = 123
num_runs = 30
maximum_acc = 0

#Save results

import csv
from csv import writer

#Append data in csv function

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
 
csv_filename = '../EyeSocketRecognition/VGG19_CK.csv'
       
#read in CSV file
if os.path.exists(csv_filename):
    print()
else:
    with open(csv_filename, 'w', newline = '') as f:
        
        header = ['Iteration', 'Accuracy'] 
        filewriter = csv.DictWriter(f, fieldnames = header)
        filewriter.writeheader()




ckta_train = []
ckta_train_label = []

#read train images
print("Reading train images...")
rootdir = '../EyeSocketRecognition/Dataset/CK/train_gabor'

for file in os.listdir(rootdir):
        
    image_path = rootdir +"/"+ file
    for filename in glob.glob(image_path + '/*.png'):
        
        identity = filename.partition(file)
        
        im = cv2.imread(filename)
        im = cv2.resize(im, (224, 224))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        ckta_train.append(im) 
        ckta_train_label.append(identity[1]) 


        
ckta_test= []
ckta_test_label = []
       
#read test images
print("Reading test images...")
rootdir = '../EyeSocketRecognition/Dataset/CK/test_gabor'

for file in os.listdir(rootdir):
    
        
    image_path = rootdir +"/"+ file
    for filename in glob.glob(image_path + '/*.png'):
        
        identity = filename.partition(file)
        
        im = cv2.imread(filename)
        im = cv2.resize(im, (224, 224))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)
        ckta_test.append(im) 
        ckta_test_label.append(identity[1])


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


ckta_train = np.array(ckta_train)
ckta_test = np.array(ckta_test)

le = LabelEncoder()
ckta_train_label = le.fit_transform(ckta_train_label)   
ckta_test_label = le.fit_transform(ckta_test_label)  


#(x_train, x_test, y_train, y_test) = train_test_split(ckta_train, ckta_train_label, test_size=0.10)
 
from tensorflow.keras.applications.efficientnet import preprocess_input
#from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.utils import to_categorical

# We normalize the input according to the methods used in the paper
X_train = preprocess_input(ckta_train)
Y_train = to_categorical(ckta_train_label)

# We one-hot-encode the labels for training
X_test = preprocess_input(ckta_test)
Y_test = to_categorical(ckta_test_label)

import time
    
start_time = time.time()

for i in range(num_runs):    
    
    #from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
    #from keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.applications.vgg19 import VGG19
    #from keras.applications.vgg16 import VGG16
    #from tensorflow.keras.applications.resnet50 import ResNet50
    #from tensorflow.keras.applications import EfficientNetB0
    #keras.applications.resnet.ResNet152
    #keras.applications.resnet_v2.ResNet50V2
    #from tensorflow.keras.applications import MobileNet
    
    model = VGG19(
        weights=None, 
        include_top=True, 
        classes= NUM_CLASSES,
        input_shape=(224,224,3)
    )
    
    # Expand this cell for the model summary
    model.summary()
    
    
    
    
    def lr_schedule(epoch):
        """Learning Rate Schedule
    
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
    
        # Arguments
            epoch (int): The number of epochs
    
        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr
    
    
    from tensorflow.keras import optimizers
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    
    '''
    from tensorflow.keras.optimizers import Adam
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['accuracy'])
    '''
    
    from tensorflow.keras.callbacks import ModelCheckpoint
    
    checkpoint = ModelCheckpoint(
        'model.h5', 
        monitor='val_acc', 
        verbose=0, 
        save_best_only=True, 
        save_weights_only=False,
        mode='auto'
    )
    
    
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False, 
            # epsilon for ZCA whitening
            zca_epsilon=1e-06,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # set range for random shear
            shear_range=0.,
            # set range for random zoom
            zoom_range=0.,
            # set range for random channel shifts
            channel_shift_range=0.,
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
            cval=0.,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=True, #new
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.1)
    
        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)
    
    
    # Prepare callbacks for model saving and for learning rate adjustment.
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    
    callbacks = [lr_reducer, lr_scheduler]
  
    from sklearn.metrics import accuracy_score
    
    print("Training on Fold: ", i + 1)
    
    x_train = X_train
    y_train = Y_train
    x_test = X_test
    y_test = Y_test
    
    
    history = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(x_test, y_test),
                    epochs=EPOCHS, steps_per_epoch = len(x_train) / BATCH_SIZE , verbose=1, workers=1,
                    callbacks=callbacks, use_multiprocessing=False)
                 
    # Conf matrix
    test_true = np.argmax(y_test, axis=1)
    test_pred = np.argmax(model.predict(x_test), axis=1)
    
    acc = accuracy_score(test_true, test_pred)
    row_contents = [str(i),str(acc)]
    # Append a list as new line to an old csv file
    append_list_as_row(csv_filename, row_contents)   

    if acc >= maximum_acc:
        maximum_acc = acc
        #model.save('model.h5')
    
    print("===" * 3, end="\n\n\n")


from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_true, test_pred))

end_time = time.time()
elapsed_time = end_time - start_time

hours = elapsed_time//3600
temp = elapsed_time - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print("Elapsed time: {}".format(hours,minutes,seconds))
print("\n---------Another way---------->\n")
print('%d:%d:%d' %(hours,minutes,seconds))


#print(maximum_acc)
print("***VGG19 CK***")





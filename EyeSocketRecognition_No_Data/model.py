# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 22:22:34 2021

@author: harisushehu
"""


if __name__ == '__main__':
    import tensorflow.compat.v1 as tf
    tf.enable_eager_execution()
    import tensorflow_datasets as tfds
    
    import numpy as np
    import cv2
    import os 
    import glob

    
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True   
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    
        
    from tensorflow.python.keras.backend import set_session
    
    set_session(sess)
    
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


    class MyDS(object):
        class SubDS(object):
            import numpy as np
            def __init__(self, ds, *, one_hot):
                np = self.__class__.np
                self.ds = [e for e in ds.as_numpy_iterator()]
                self.sds = {(k + 's') : np.stack([
                    (e[k] if len(e[k].shape) > 0 else e[k][None]).reshape(-1) for e in self.ds
                ], 0) for k in self.ds[0].keys()}
                self.one_hot = one_hot
                if one_hot is not None:
                    self.max_one_hot = np.max(self.sds[one_hot + 's'])
            def _to_one_hot(self, a, maxv):
                np = self.__class__.np
                na = np.zeros((a.shape[0], maxv + 1), dtype = a.dtype)
                for i, e in enumerate(a[:, 0]):
                    na[i, e] = True
                return na
            def _apply_one_hot(self, key, maxv):
                assert maxv >= self.max_one_hot, (maxv, self.max_one_hot)
                self.max_one_hot = maxv
                self.sds[key + 's'] = self._to_one_hot(self.sds[key + 's'], self.max_one_hot)
            def next_batch(self, num = 16):
                np = self.__class__.np
                idx = np.random.choice(len(self.ds), num)
                res = {k : np.stack([
                    (self.ds[i][k] if len(self.ds[i][k].shape) > 0 else self.ds[i][k][None]).reshape(-1) for i in idx
                ], 0) for k in self.ds[0].keys()}
                if self.one_hot is not None:
                    res[self.one_hot] = self._to_one_hot(res[self.one_hot], self.max_one_hot)
                for i, (k, v) in enumerate(list(res.items())):
                    res[i] = v
                return res
            def __getattr__(self, name):
                if name not in self.__dict__['sds']:
                    return self.__dict__[name]
                return self.__dict__['sds'][name]
        def __init__(self, name, *, one_hot = None):
            self.ds = tfds.load(name)
            self.sds = {}
            for k, v in self.ds.items():
                self.sds[k] = self.__class__.SubDS(self.ds[k], one_hot = one_hot)
            if one_hot is not None:
                maxh = max(e.max_one_hot for e in self.sds.values())
                for e in self.sds.values():
                    e._apply_one_hot(one_hot, maxh)
        def __getattr__(self, name):
            if name not in self.__dict__['sds']:
                return self.__dict__[name]
            return self.__dict__['sds'][name]
        
        
            
    # Get the MNIST data
    mnist = MyDS('mnist', one_hot = 'label') # tensorflow_datasets.load('mnist')

    import argparse
    print ('argparse version: ', argparse.__version__)
   
    tf.disable_eager_execution()
    tf.disable_v2_behavior()


    def get_weights(shape):
        data = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(data)

    def get_biases(shape):
        data = tf.constant(0.1, shape=shape)
        return tf.Variable(data)

    def create_layer(shape):
        # Get the weights and biases
        W = get_weights(shape)
        b = get_biases([shape[-1]])

        return W, b

    def convolution_2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                padding='SAME')

    def max_pooling(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME')

    #args = build_arg_parser().parse_args()

    # The images are 28x28, so create the input layer
    # with 784 neurons (28x28=784)
    x = tf.placeholder(tf.float32, [None, 784])

    # Reshape 'x' into a 4D tensor
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Define the first convolutional layer
    W_conv1, b_conv1 = create_layer([5, 5, 1, 32])

    # Convolve the image with weight tensor, add the
    # bias, and then apply the ReLU function
    h_conv1 = tf.nn.relu(convolution_2d(x_image, W_conv1) + b_conv1)

    # Apply the max pooling operator
    h_pool1 = max_pooling(h_conv1)

    # Define the second convolutional layer
    W_conv2, b_conv2 = create_layer([5, 5, 32, 64])

    # Convolve the output of previous layer with the
    # weight tensor, add the bias, and then apply
    # the ReLU function
    h_conv2 = tf.nn.relu(convolution_2d(h_pool1, W_conv2) + b_conv2)

    # Apply the max pooling operator
    h_pool2 = max_pooling(h_conv2)

    # Define the fully connected layer
    W_fc1, b_fc1 = create_layer([7 * 7 * 64, 1024])

    # Reshape the output of the previous layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    # Multiply the output of previous layer by the
    # weight tensor, add the bias, and then apply
    # the ReLU function
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Define the dropout layer using a probability placeholder
    # for all the neurons
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Define the readout layer (output layer)
    W_fc2, b_fc2 = create_layer([1024, 123])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Define the entropy loss and the optimizer
    y_loss = tf.placeholder(tf.float32, [None, 123])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_loss, logits=y_conv))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # Define the accuracy computation
    predicted = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_loss, 1))
    accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

    
	
	#read train and test data 
	
    ckta_train = []
    ckta_train_label = []

    #read train images
    print("Reading train images...")
    rootdir = 'train dataset path'

    for file in os.listdir(rootdir):
        
        image_path = rootdir +"/"+ file
        for filename in glob.glob(image_path + '/*.png'):
			
            identity = filename.partition(file)
			
            im = cv2.imread(filename)
            im = cv2.resize(im, (28, 28))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            ckta_train.append(im) 
            ckta_train_label.append(identity[1])
           


    ckta_test= []
    ckta_test_label = []
		   
    #read test images
    print("Reading test images...")
    rootdir = 'test dataset path'

    for file in os.listdir(rootdir):
		
			
        image_path = rootdir +"/"+ file
        for filename in glob.glob(image_path + '/*.png'):
			
            identity = filename.partition(file)
			
            im = cv2.imread(filename)
            im = cv2.resize(im, (28, 28))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            ckta_test.append(im) 
            ckta_test_label.append(identity[1])
    
   
    from sklearn.preprocessing import LabelEncoder
    
    ckta_train = np.array(ckta_train)
    ckta_test = np.array(ckta_test)
    
    le = LabelEncoder()
    ckta_train_label = le.fit_transform(ckta_train_label)   
    ckta_test_label = le.fit_transform(ckta_test_label)  
   
    
    ckta_train = np.reshape(ckta_train, (-1, 784))
    ckta_test = np.reshape(ckta_test, (-1, 784))
    
    ckta_train = ckta_train.astype('int32')
    ckta_test = ckta_test.astype('int32')
    
    from tensorflow.keras.utils import to_categorical

    # We normalize the input according to the methods used in the paper
    ckta_train_label = to_categorical(ckta_train_label)
    ckta_test_label = to_categorical(ckta_test_label)
    
    # Create and run a session
    sess = tf.InteractiveSession()
    init = tf.initialize_all_variables()
    sess.run(init)
    
    #Save results
    import csv
    from csv import writer
    
	csv_filename = 'your csv file path + name'
	
    #Append data in csv function
    
    def append_list_as_row(file_name, list_of_elem):
        # Open file in append mode
        with open(file_name, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)
            
    #read in CSV file
    if os.path.exists(csv_filename):
        print()
    else:
        with open(csv_filename, 'a+', newline = '') as f:
            
            header = ['Iteration', 'Accuray'] 
            filewriter = csv.DictWriter(f, fieldnames = header)
            filewriter.writeheader()

    epochs_completed = 0
    index_in_epoch = 0
    num_examples = ckta_train.shape[0]
    
    def next_batch(num, data, labels):
        '''
        Return a total of `num` random samples and labels. 
        '''        
        maxi = len(labels)
        
        idx = np.arange(0 , maxi)
        
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [data[ i] for i in idx]
        labels_shuffle = [labels[ i] for i in idx]
    
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    
    
    for k in range(0, 30):
        
        print("Iteration :", k)
        
        # Start training
        num_iterations = 21000
        batch_size = 128 #32
        test_batch_size = 8 #350
       
        print('\nTraining the model....')
        for i in range(num_iterations):
            # Get the next batch of images
            batch = next_batch(batch_size, ckta_train, ckta_train_label)
            
            # Print progress
            if i % 50 == 0:
                cur_accuracy = accuracy.eval(feed_dict = {
                        x: batch[0], y_loss: batch[1], keep_prob: 1.0})
                print('Iteration', i, ', Accuracy =', cur_accuracy)
    
            # Train on the current batch
            #optimizer.run(feed_dict = {x: ckta_train, y_loss: tf.one_hot(ckta_train_label, 7), keep_prob: 0.5})
            optimizer.run(feed_dict = {x: batch[0], y_loss: batch[1], keep_prob: 0.5})
        
        test_batch = next_batch(test_batch_size, ckta_test, ckta_test_label)
        
        # Compute accuracy using test data
        print('Test accuracy =', accuracy.eval(feed_dict = {
                x: test_batch[0], y_loss: test_batch[1],
                keep_prob: 1.0}))
    
        row_contents = [str(k),str(accuracy.eval(feed_dict = {
                x: test_batch[0], y_loss: test_batch[1],
                keep_prob: 1.0}))]
        # Append a list as new line to an old csv file
        append_list_as_row(csv_filename, row_contents)  
    

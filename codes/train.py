from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from utils import *
from models import GAutoencoder

def train(FLAGS,train_arr, test_arr):
        # Settings
    
    # Load data
    adj, features, size_u, size_v, logits_train, logits_test, train_mask, test_mask, labels = load_data(train_arr, test_arr)

    # Some preprocessing
    # features = preprocess_features(features)
    if FLAGS.model == 'GAutoencoder':
        model_func = GAutoencoder
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    
    
    # Define placeholders
    placeholders = {
        'adjacency_matrix': tf.placeholder(tf.int32, shape=adj.shape),
        'Feature_matrix': tf.placeholder(tf.float32, shape=features.shape),
        'labels': tf.placeholder(tf.float32, shape=(None, logits_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'negative_mask': tf.placeholder(tf.int32)
    }
    
    # Create model
    model = model_func(placeholders, size_u, size_v, FLAGS.latent_factor_num)
    
    # Initialize session
    sess = tf.Session()
    
    
    # Define model evaluation function
    def evaluate(adj, features, labels, mask, negative_mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(adj, features,labels, mask, negative_mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)
    
    
    # Init variables
    sess.run(tf.global_variables_initializer())
    
    # Train model
    for epoch in range(FLAGS.epochs):
     
        t = time.time()
        # Construct feed dictionary
        negative_mask = generate_mask(labels, len(train_arr))
        
        feed_dict = construct_feed_dict(adj, features, logits_train, train_mask, negative_mask, placeholders)
             
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
#         print(sess.run(model.outputs, feed_dict=feed_dict))
     
        # Print results
        print("Epoch:", '%04d' % (epoch + 1), 
              "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), 
              "time=", "{:.5f}".format(time.time() - t))
     
#         if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
#             print("Early stopping...")
#             break
#      
    print("Optimization Finished!")
     
    # Testing
    test_cost, test_acc, test_duration = evaluate(adj, features, logits_test, test_mask, negative_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
 
    # Computing ROC curves from positions
    feed_dict_val = construct_feed_dict(adj, features, logits_test, test_mask, negative_mask ,placeholders)
    outs = sess.run(model.outputs, feed_dict=feed_dict_val)
    hid = sess.run(model.hid, feed_dict=feed_dict_val)
    print(type(hid))
    print(len(hid))
    print(hid)
    outs = np.array(outs)[:,0]
    outs = outs.reshape((106,754))
    outs_temp = outs.copy()
    # print(labels.shape[0])
    positive_position ={}
    
    for i in range(labels.shape[0]):        
        outs_temp[int(labels[i][0])-1,int(labels[i][1])-1] = float("-inf")

    outs_temp = outs_temp.reshape((1,-1))    
    outs_temp.sort()
        
    for j in range(len(test_arr)):
        i = test_arr[j]
        p = outs[int(labels[i][0]-1),int(labels[i][1])-1]
        p1 = outs_temp.shape[1]-np.where(p>=outs_temp)[0].shape[0]
        p2 = outs_temp.shape[1]-np.where(p>outs_temp)[0].shape[0]
        positive_position[i] = int((p1+p2)/2)
    
    return positive_position
    
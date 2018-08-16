from utils import *
from models import GAutoencoder
import time
import tensorflow as tf

import numpy as np
from train import *


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'GAutoencoder', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('latent_factor_num',25,'The size of latent factor vector.')

labels = np.loadtxt("adj.txt")
reorder = np.arange(labels.shape[0])
np.random.shuffle(reorder)

T = 10
cv_num = 5
position = np.zeros([T, labels.shape[0]])
for t in range(T):
    order = div_list(reorder.tolist(),cv_num)    
    for i in range(cv_num):
        print("cross_validation:", '%01d' % (i))
        test_arr = order[i]
        arr = list(set(reorder).difference(set(test_arr)))
        np.random.shuffle(arr)
        train_arr = arr
        A = train(FLAGS, train_arr, test_arr)
        for j in A:
            position[t,j] = A[j]
np.savetxt('expression_25_position(5-fold-cv)time5.txt',position)
import pandas as pd
import numpy as np
import os

# plot imports
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

# keras imports
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.optimizers import Adam, SGD, Adadelta, Adagrad
from sklearn.model_selection import train_test_split

# sklearn imports
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# other imports
from tensorflow.python.client import device_lib
import itertools

# warnings
import warnings

warnings.filterwarnings('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)

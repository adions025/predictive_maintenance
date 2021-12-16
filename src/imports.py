import pandas as pd
import numpy as np
import os

# plot imports
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import seaborn as sns

# keras imports
import keras
import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.optimizers import Adam, SGD, Adadelta, Adagrad
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

# sklearn imports
import sklearn
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# other imports
from tensorflow.python.client import device_lib
import itertools
from pylab import rcParams
import math
import xgboost
import time
from tqdm import tqdm
from xgboost import XGBClassifier

# Preprocesado y modelado Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.tree import export_text

# Preprocesado y modelado de Naive-Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest

# warnings
import warnings

warnings.filterwarnings('once')
warnings.filterwarnings("ignore", category=DeprecationWarning)

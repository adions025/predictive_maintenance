from keras.callbacks import ModelCheckpoint
from keras import optimizers
import tensorflow as tf
from imports import *
from metric_fuctions import *
from data_functions import *


def lstm_train(seq_array, label_array, sequence_length):
    nb_features = seq_array.shape[2]
    nb_out = label_array.shape[1]

    model = Sequential()
    model.add(LSTM(
        input_shape=(sequence_length, nb_features),
        units=100,
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        units=50,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=nb_out))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae', r2_keras])

    print(model.summary())
    history = model.fit(seq_array, label_array, epochs=60, batch_size=200, validation_split=0.05, verbose=2)
    print(history.history.keys())
    return model, history


def train_models(data, model='RF'):
    """
    Function for creating and training different models

    :param data:
    :param model:
    :return:
    """
    if model != 'LSTM':
        X = data.iloc[:, :14].to_numpy()
        Y = data.iloc[:, 14:].to_numpy()
        Y = np.ravel(Y)

    if model == 'RF':
        model = RandomForestRegressor(n_estimators=70, max_features=7, max_depth=5, n_jobs=-1, random_state=1)
        model.fit(X, Y)
        return model

    elif model == 'DT':
        model = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=0)
        # Entrenamiento del modelo
        model.fit(X, Y)
        return model

    elif model == 'XGB':
        model = xgboost.XGBRegressor(n_estimators=110, learning_rate=0.018, gamma=0, subsample=0.8,
                                     colsample_bytree=0.5, max_depth=3, silent=True)
        model.fit(X, Y)
        return model

    elif model == 'LSTM':
        seq_array, label_array, lstm_test_df, sequence_length, sequence_cols = lstm_data_preprocessing(data[0], data[1],
                                                                                                       data[2])
        model_instance, history = lstm_train(seq_array, label_array, sequence_length)
        return model_instance, history, lstm_test_df, seq_array, label_array, sequence_length, sequence_cols
    return


def create_model(neurons: list, dim: int, classes: int) -> Model:
    """
    Function to create a model, se necesita una lista de capas ocultas.
    Se necesita el tamaño de la entrada y la salida, por defecto la
    las capas ocultas se activan con relu, y la capa de salida softmax.
    Cause we have a multi-class classification problem = there is only
    one "right answer" = the outputs are mutually exclusive, then we
    use a softmax function. The softmax will enforce that the sum of
    the probabilities of  output classes are equal to one, so in order
    to increase the probability of a particular class.

    :param neurons: A list of hidden layers
    :param dim: A int value of input dimension
    :param classes: A int value of output dimension
    :return: A tf model
    """
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=dim, activation='relu'))
    [model.add(Dense(neurons[i], activation='relu')) for i in range(1, len(neurons))]
    model.add(Dense(classes, activation='softmax'))
    return model


LR = 0.001
BATCH_SIZE = 512


def compile_model(model: Model, optimizer: str, loss: str, lr: float = LR) -> Model:
    """
    Esta función recibe por parámetro el modelo a compilar y además el optimizador
    y la función de perdida, por defecto se definen ciertos parametros. Por ejemplo
    el learning rate aunque por defecto es 1-e2

    :param model: A tf model already created
    :param optimizer: A str optimizer
    :param loss: A str loss
    :param lr: A float
    :return: A compiled model
    """
    if optimizer == 'Adam':
        opt = optimizers.Adam(lr=lr)
    if optimizer == 'rms':
        opt = optimizers.RMSprop(lr=lr)
    if optimizer == 'SGD':
        opt = optimizers.SGD(lr=lr)
    if optimizer == 'Adadelta':
        opt = optimizers.Adadelta(lr=lr)
    if optimizer == 'Adagrad':
        opt = optimizers.Adagrad(lr=lr)
    if loss == 'sparse_c_c':
        los = 'sparse_categorical_crossentropy'
    if loss == 'categorial_c':
        los = tf.keras.losses.CategoricalCrossentropy()
    print(f'model compile using {opt, los, lr}')
    model.compile(optimizer=opt, loss=los, metrics=['accuracy'])
    return model


METRIC = "val_loss"
PATH = './model.h5'


def create_callbacks(metric: str = METRIC, path: str = PATH) -> list:
    """
    Function para crear la callbacks del modelo, permite pasar un filepath nuevo
    para cada modelo, por defecto la métrica de monitorización es la val_loss.

    :param metric: A str function loss name
    :param path: A str like /to/path/
    :return: a callback ModelCheckpoint
    """
    checkpoint = ModelCheckpoint(
        filepath=path,
        monitor=metric,
        mode='max',
        save_best_only=True,
        verbose=1)
    callbacks = [checkpoint]
    return callbacks

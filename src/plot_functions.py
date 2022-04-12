"""
Funciones necesarias para graficar
resultados.

@author: Adonis Gonzalez
"""

from imports import *


def plot_cm(cm: confusion_matrix, labels: list):
    """
    Function to plot a confusion matrix. Se necesita a
    computed confusion matrix object. By definition a
    confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in
    group :math:`i` and  predicted to be in group :math:`j`.

    :param cm: A computed matrix object
    :param labels: A list of unique target values
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white",
                 weight="bold",
                 fontsize=25)

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_accuracy(history):
    """
    Function to plot el accuracy del set de train y validation
    Is needed a `History` object. Its `History.history`
    attribute is a record of training loss values and metrics
    values at successive epochs, as well as validation loss
    values and validation metrics values (if applicable).

    :param history: a `History` object
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train val', 'Test val'], loc='upper right')
    plt.show()


def plot_loss(history):
    """
    Function to plot el loss del set de train y validation
    Is needed a `History` object. Its `History.history`
    attribute is a record of training loss values and metrics
    values at successive epochs, as well as validation loss
    values and validation metrics values (if applicable).

    :param history: a `History` object
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train loss', 'Test loss'], loc='upper right')
    plt.show()


def plot_matrix_correlations(correlations):
    """
    Plot rectangular data as a color-encoded matrix.
    This is an Axes-level function and will draw the heatmap.
    Axes space will be taken and used to plot a colormap, unless ``cbar``
    is False or a separate Axes is provided to ``cbar_ax``."""

    sns.heatmap(correlations, annot=True, cmap='YlGnBu', linewidths=.3)
    fig = plt.gcf()
    fig.set_size_inches(19, 19)
    plt.show()


def lstm_evaluation(model, history, seq_array, label_array):
    plt.plot(history.history['r2_keras'])
    plt.plot(history.history['val_r2_keras'])
    plt.title('model r^2')
    plt.ylabel('R^2')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model MAE')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
    print('\nMAE: {}'.format(scores[1]))
    print('\nR^2: {}'.format(scores[2]))
    y_pred = model.predict(seq_array, verbose=1, batch_size=200)
    test_set = pd.DataFrame(y_pred)
    test_set.head()

def plot_result(y_true, y_pred):
    rcParams['figure.figsize'] = 15, 9
    plt.plot(y_pred)
    plt.plot(y_true)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel('RUL')
    plt.xlabel('Observaciones de entramiento')
    plt.legend(('Predicted', 'True'), loc='upper right')
    plt.title('Comparasión de True values and Predicted values')
    plt.show()
    return

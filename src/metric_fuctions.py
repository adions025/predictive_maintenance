"""
Funciones necesarias para la evaluación
de modelos regresores.

@author: Adonis Gonzalez
"""

from imports import *


def score(y_true, y_pred, a1=10, a2=13):
    score = 0
    d = y_pred - y_true
    for i in d:
        if i >= 0:
            score += math.exp(i / a2) - 1
        else:
            score += math.exp(- i / a1) - 1
    return score


def score_func(y_true, y_pred):
    lst = [round(score(y_true, y_pred), 2),
           round(mean_absolute_error(y_true, y_pred), 2),
           round(mean_squared_error(y_true, y_pred), 2) ** 0.5,
           round(r2_score(y_true, y_pred), 2)]

    print(f' compatitive score {lst[0]}')
    print(f' mean absolute error {lst[1]}')
    print(f' root mean squared error {lst[2]}')
    print(f' R2 score {lst[3]}')
    return [lst[1], round(lst[2], 2), lst[3] * 100]


def r2_keras(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def lstm_test_eval(lstm_test_df, model, sequence_length, sequence_cols):
    # We pick the last sequence for each id in the test data
    seq_array_test_last = [lstm_test_df[lstm_test_df['id'] == id][sequence_cols].values[-sequence_length:]
                           for id in lstm_test_df['id'].unique() if
                           len(lstm_test_df[lstm_test_df['id'] == id]) >= sequence_length]

    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

    # Similarly, we pick the labels
    y_mask = [len(lstm_test_df[lstm_test_df['id'] == id]) >= sequence_length for id in
              lstm_test_df['id'].unique()]
    label_array_test_last = lstm_test_df.groupby('id')['RUL'].nth(-1)[y_mask].values
    label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)

    estimator = model

    # test metrics
    scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
    print('\nMAE: {}'.format(scores_test[1]))
    print('\nR^2: {}'.format(scores_test[2]))

    y_pred_test = estimator.predict(seq_array_test_last)
    y_true_test = label_array_test_last

    test_set = pd.DataFrame(y_pred_test)
    print(test_set.head())

    # Plot in blue color the predicted data and in green color the
    # actual data to verify visually the accuracy of the model.
    fig_verify = plt.figure(figsize=(10, 5))
    plt.plot(y_pred_test)
    plt.plot(y_true_test, color="orange")
    plt.title('prediction')
    plt.ylabel('value')
    plt.xlabel('row')
    plt.legend(['predicted', 'actual data'], loc='upper left')
    plt.show()
    return scores_test[1], scores_test[2]

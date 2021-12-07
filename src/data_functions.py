from imports import *


def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    id_df = df_zeros.append(id_df, ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)


# function to generate labels
def gen_label(id_df, seq_length, seq_cols, label):
    df_zeros = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    id_df = df_zeros.append(id_df, ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label = []
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label][stop])
    return np.array(y_label)


def prepare_train_data(df):
    """
    Crea una columna en el dataset recibido por parametro.
    Esta columna se obtiene de la resta del ultim ciclo de cada id,
    menos el ciclo actual.

    :param df: a pd Dataframe
    :return: a df
    """
    rul = pd.DataFrame(df.groupby('id')['cycle_time'].max().reset_index())
    rul.columns = ['id', 'max']
    df = df.merge(rul, on=['id'], how='left')
    df['RUL'] = df['max'] - df['cycle_time']
    df.drop(columns=['max'], inplace=True)
    return df
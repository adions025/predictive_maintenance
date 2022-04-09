from imports import *
from model_functions import *


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


def check_df_null(df):
    """
    Recive por parametro un objeto DataFrame, este se itera y se
    comprueba si hay nulos, si existen se imprime por pantalla.

    :param df: A DataFrame object to check
    """
    for key, value in df.isnull().sum().iteritems():
        if value > 0:
            print(key, " -->", value)


def add_labels_df(df, w1=30, w0=15):
    df['label1'] = np.where(df['RUL'] <= w1, 1, 0)  # se añade 1 si es menor o igual que w1
    df['label2'] = df['label1']
    df.loc[df['RUL'] <= w0, 'label2'] = 2
    return df


def normalize_df(df):
    """
    Funcion para normalizar
    """
    df['cycle_norm'] = df['cycle_time']
    cols_norm = df.columns.difference(
        ['id', 'cycle_time', 'RUL', 'label1', 'label2'])  # NORMALIZE COLUMNS except [id , cycle, rul ....]

    min_max_scaler = MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(df[cols_norm]),
                                 columns=cols_norm,
                                 index=df.index)

    join_df = df[df.columns.difference(cols_norm)].join(norm_train_df)
    train_df = join_df.reindex(columns=df.columns)
    return train_df


def gen_labels(id_df, seq_length, label):
    """Funcion para crear las secuencias de los labels a
       partir del tamaño definidio de seque."""

    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]


def gen_sequence(id_df, seq_length, seq_cols):
    """
    Funcion para crear las secuencias en del tamaño
    seq_length. Se itera se coge la seq de todas
    las columnas definidas.
    """

    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

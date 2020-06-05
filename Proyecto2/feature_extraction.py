import numpy as np
import tensorflow as tf
import csv
from sklearn import model_selection
from sklearn import svm
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential


def get_features(file_path):
    file = tf.io.read_file(file_path)
    content = file.numpy()
    values = content.decode()
    features_list = values.split('\r\n')
    data = []

    for row in features_list:
        if len(row) > 1:
            data.append(row.split(','))

    data = np.array(data)
    features = data[:, :-1]
    labels = data[:, -1]

    return features.astype(float), labels


def extract_data(ruta):
    dataset = tf.data.Dataset.list_files(ruta)
    datos = np.empty((0, 29))
    etiquetas = np.empty(0)

    for element in dataset:
        dato, etiqueta = get_features(element)
        datos = np.concatenate([datos, dato], axis=0)
        etiquetas = np.concatenate([etiquetas, etiqueta], axis=0)

    return datos, etiquetas


def separate_data(datos, etiquetas):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(datos,
                                                                        etiquetas,
                                                                        test_size=0.2)

    train_data_size = len(X_train)
    test_data_size = len(X_test)

    X_train = np.array(X_train)
    y_train = np.expand_dims(np.array(y_train), axis=1)

    train_data = np.concatenate([X_train, y_train], axis=1)

    X_test = np.array(X_test)
    y_test = np.expand_dims(np.array(y_test), axis=1)

    test_data = np.concatenate([X_test, y_test], axis=1)

    print('Datos entrenamiento: ' + str(train_data_size))
    print('Datos prueba: ' + str(test_data_size))

    train_file = open('Parcial2Machine/TrainData.csv', 'w', newline='')
    test_file = open('Parcial2Machine/TestData.csv', 'w', newline='')

    train_writer = csv.writer(train_file)
    test_writer = csv.writer(test_file)

    train_writer.writerows(train_data.tolist())
    test_writer.writerows(test_data.tolist())

    train_file.close()
    test_file.close()


def train_svm(X_train, y_train, X_validation, y_validation):
    C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]

    for constant in C:
        print('Entrenando con %d' % constant)
        model = svm.SVC(C=constant,
                        kernel='rbf')
        model.fit(X_train, y_train)
        print('Precision alcanzada: %d' % model.score(X_validation, y_validation))


def train_neuralnet(train_dataset, valid_dataset, name):
    layer_1 = Dense(18, input_shape=[29], activation='tanh')
    layer_2 = Dense(10, activation='tanh')
    layer_3 = Dense(4, activation='softmax')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model = Sequential([layer_1, layer_2, layer_3])

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.fit(x=train_dataset,
              epochs=250,
              verbose=2,
              steps_per_epoch=71,
              validation_data=valid_dataset,
              validation_steps=245)

    model.save('Parcial2Machine/' + name)


def preprocessed_datadict_extraction(data, label):
    features = tf.stack(list(data.values()), axis=1)

    if label.dtype == tf.string:
        label = tf.cast(label == 'Stage1', tf.int32) * 1 + \
                tf.cast(label == 'Stage2', tf.int32) * 1 + \
                tf.cast(label == 'Stage3', tf.int32) * 2 + \
                tf.cast(label == 'Stage4', tf.int32) * 2 + \
                tf.cast(label == 'StageR', tf.int32) * 3

    label = tf.one_hot(label, 4)
    return features, label


def get_preprocessed_datasets(number):
    dataset = tf.data.experimental.make_csv_dataset('Parcial2_Machine/CostedData/costed_data_' + str(number) + '.csv',
                                                    128,
                                                    ['f_' + str(i) for i in range(29)] + ['label'],
                                                    label_name='label',
                                                    shuffle=False,
                                                    header=False)

    dataset = dataset.map(preprocessed_datadict_extraction).cache().prefetch(20).repeat(250)

    valid = tf.data.experimental.make_csv_dataset('Parcial2_Machine/ValidationData.csv',
                                                  128,
                                                  ['f_' + str(i) for i in range(29)] + ['label'],
                                                  label_name='label',
                                                  shuffle=False,
                                                  header=False)

    valid = valid.map(preprocessed_datadict_extraction)

    return dataset, valid


def obtain_data_0(data):
    # features = tf.stack(data.values(), axis=1)
    return {'ch_0': tf.reshape(data['ch_0'], [3000, 1]),
            'ch_1': tf.reshape(data['ch_1'], [3000, 1]),
            'ch_2': tf.reshape(data['ch_2'], [3000, 1])}, tf.one_hot(0, 4)


def obtain_data_1(data):
    # features = tf.stack(list(data.values()), axis=1)
    return {'ch_0': tf.reshape(data['ch_0'], [3000, 1]),
            'ch_1': tf.reshape(data['ch_1'], [3000, 1]),
            'ch_2': tf.reshape(data['ch_2'], [3000, 1])}, tf.one_hot(1, 4)


def obtain_data_2(data):
    # features = tf.stack(list(data.values()), axis=1)
    return {'ch_0': tf.reshape(data['ch_0'], [3000, 1]),
            'ch_1': tf.reshape(data['ch_1'], [3000, 1]),
            'ch_2': tf.reshape(data['ch_2'], [3000, 1])}, tf.one_hot(2, 4)


def obtain_data_3(data):
    # features = tf.stack(list(data.values()), axis=1)
    return {'ch_0': tf.reshape(data['ch_0'], [3000, 1]),
            'ch_1': tf.reshape(data['ch_1'], [3000, 1]),
            'ch_2': tf.reshape(data['ch_2'], [3000, 1])}, tf.one_hot(3, 4)
            
def normalize(data, label):
    return {'ch_0': (data['ch_0'] + 250)/500,
            'ch_1': (data['ch_1'] + 250)/500,
            'ch_2': (data['ch_2'] + 1000)/2000}, label


def construct_cnn():

    imput_0 = tf.keras.Input(shape=(3000, 1), name='ch_0')
    imput_1 = tf.keras.Input(shape=(3000, 1), name='ch_1')
    imput_2 = tf.keras.Input(shape=(3000, 1), name='ch_2')

    chanel_0 = tf.keras.layers.Conv1D(3, 3, activation='relu')(imput_0)
    chanel_1 = tf.keras.layers.Conv1D(3, 3, activation='relu')(imput_1)
    chanel_2 = tf.keras.layers.Conv1D(3, 3, activation='relu')(imput_2)

    chanel_0 = tf.keras.layers.AveragePooling1D(3)(chanel_0)
    chanel_1 = tf.keras.layers.AveragePooling1D(3)(chanel_1)
    chanel_2 = tf.keras.layers.AveragePooling1D(3)(chanel_2)
    
    chanel_0 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_0)
    chanel_1 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_1)
    chanel_2 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_2)

    chanel_0 = tf.keras.layers.Conv1D(5, 3, activation='relu')(chanel_0)
    chanel_1 = tf.keras.layers.Conv1D(5, 3, activation='relu')(chanel_1)
    chanel_2 = tf.keras.layers.Conv1D(5, 3, activation='relu')(chanel_2)

    chanel_0 = tf.keras.layers.AveragePooling1D(3)(chanel_0)
    chanel_1 = tf.keras.layers.AveragePooling1D(3)(chanel_1)
    chanel_2 = tf.keras.layers.AveragePooling1D(3)(chanel_2)
    
    chanel_0 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_0)
    chanel_1 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_1)
    chanel_2 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_2)

    chanel_0 = tf.keras.layers.Conv1D(7, 3, activation='relu')(chanel_0)
    chanel_1 = tf.keras.layers.Conv1D(7, 3, activation='relu')(chanel_1)
    chanel_2 = tf.keras.layers.Conv1D(7, 3, activation='relu')(chanel_2)

    chanel_0 = tf.keras.layers.AveragePooling1D(3)(chanel_0)
    chanel_1 = tf.keras.layers.AveragePooling1D(3)(chanel_1)
    chanel_2 = tf.keras.layers.AveragePooling1D(3)(chanel_2)
    
    chanel_0 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_0)
    chanel_1 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_1)
    chanel_2 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_2)

    chanel_0 = tf.keras.layers.Conv1D(9, 3, activation='relu')(chanel_0)
    chanel_1 = tf.keras.layers.Conv1D(9, 3, activation='relu')(chanel_1)
    chanel_2 = tf.keras.layers.Conv1D(9, 3, activation='relu')(chanel_2)

    chanel_0 = tf.keras.layers.AveragePooling1D(3)(chanel_0)
    chanel_1 = tf.keras.layers.AveragePooling1D(3)(chanel_1)
    chanel_2 = tf.keras.layers.AveragePooling1D(3)(chanel_2)
    
    chanel_0 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_0)
    chanel_1 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_1)
    chanel_2 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_2)

    chanel_0 = tf.keras.layers.Conv1D(11, 3, activation='relu')(chanel_0)
    chanel_1 = tf.keras.layers.Conv1D(11, 3, activation='relu')(chanel_1)
    chanel_2 = tf.keras.layers.Conv1D(11, 3, activation='relu')(chanel_2)

    chanel_0 = tf.keras.layers.MaxPooling1D(3)(chanel_0)
    chanel_1 = tf.keras.layers.MaxPooling1D(3)(chanel_1)
    chanel_2 = tf.keras.layers.MaxPooling1D(3)(chanel_2)
    
    chanel_0 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_0)
    chanel_1 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_1)
    chanel_2 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_2)

    chanel_0 = tf.keras.layers.Conv1D(13, 3, activation='relu')(chanel_0)
    chanel_1 = tf.keras.layers.Conv1D(13, 3, activation='relu')(chanel_1)
    chanel_2 = tf.keras.layers.Conv1D(13, 3, activation='relu')(chanel_2)

    chanel_0 = tf.keras.layers.MaxPooling1D(3)(chanel_0)
    chanel_1 = tf.keras.layers.MaxPooling1D(3)(chanel_1)
    chanel_2 = tf.keras.layers.MaxPooling1D(3)(chanel_2)
    
    chanel_0 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_0)
    chanel_1 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_1)
    chanel_2 = tf.keras.layers.BatchNormalization(center=False, scale=False)(chanel_2)

    features = tf.keras.layers.concatenate([chanel_0, chanel_1, chanel_2])
    features = tf.keras.layers.Flatten()(features)

    #features = tf.keras.layers.Dense(40, activation='relu')(features)
    features = tf.keras.layers.Dense(20, activation='relu')(features)
    features = tf.keras.layers.Dense(10, activation='relu')(features)
    features = tf.keras.layers.Dense(4, activation='softmax')(features)

    model = tf.keras.Model(inputs=[imput_0, imput_1, imput_2],
                           outputs=[features])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model


if __name__ == "__main__":
    if __name__ == "__main__":
        train_data_1 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/TrainData/StageW/*.csv'],
                                                             3000,
                                                             ['ch_' + str(i) for i in range(3)],
                                                             shuffle=False,
                                                             header=False)

        train_data_2 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/TrainData/Stage1/*.csv'],
                                                             3000,
                                                             ['ch_' + str(i) for i in range(3)],
                                                             shuffle=False,
                                                             header=False)

        train_data_21 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/TrainData/Stage2/*.csv'],
                                                              3000,
                                                              ['ch_' + str(i) for i in range(3)],
                                                              shuffle=False,
                                                              header=False)

        train_data_3 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/TrainData/Stage3/*.csv'],
                                                             3000,
                                                             ['ch_' + str(i) for i in range(3)],
                                                             shuffle=False,
                                                             header=False)

        train_data_31 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/TrainData/Stage4/*.csv'],
                                                              3000,
                                                              ['ch_' + str(i) for i in range(3)],
                                                              shuffle=False,
                                                              header=False)

        train_data_4 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/TrainData/StageR/*.csv'],
                                                             3000,
                                                             ['ch_' + str(i) for i in range(3)],
                                                             shuffle=False,
                                                             header=False)

        train_data_1 = train_data_1.map(obtain_data_0, tf.data.experimental.AUTOTUNE).take(202082) \
            .cache('cache31').shuffle(1024).repeat()
        train_data_2 = train_data_2.map(obtain_data_1, tf.data.experimental.AUTOTUNE).take(11185) \
            .cache('cache32').shuffle(1024).repeat()
        train_data_21 = train_data_21.map(obtain_data_1, tf.data.experimental.AUTOTUNE).take(45223) \
            .cache('cache33').shuffle(1024).repeat()
        train_data_3 = train_data_3.map(obtain_data_2, tf.data.experimental.AUTOTUNE).take(3797) \
            .cache('cache34').shuffle(1024).repeat()
        train_data_31 = train_data_31.map(obtain_data_2, tf.data.experimental.AUTOTUNE).take(2334) \
            .cache('cache35').shuffle(1024).repeat()
        train_data_4 = train_data_4.map(obtain_data_3, tf.data.experimental.AUTOTUNE).take(17486) \
            .cache('cache36').shuffle(1024).repeat()

        data = tf.data.experimental.sample_from_datasets([train_data_1, train_data_2, train_data_3,
                                                          train_data_21, train_data_31, train_data_4])

        valid_data_1 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/ValidationData/StageW/*.csv'],
                                                             3000,
                                                             ['ch_' + str(i) for i in range(3)],
                                                             shuffle=False,
                                                             header=False)

        valid_data_2 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/ValidationData/Stage1/*.csv'],
                                                             3000,
                                                             ['ch_' + str(i) for i in range(3)],
                                                             shuffle=False,
                                                             header=False)

        valid_data_21 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/ValidationData/Stage2/*.csv'],
                                                              3000,
                                                              ['ch_' + str(i) for i in range(3)],
                                                              shuffle=False,
                                                              header=False)

        valid_data_3 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/ValidationData/Stage3/*.csv'],
                                                             3000,
                                                             ['ch_' + str(i) for i in range(3)],
                                                             shuffle=False,
                                                             header=False)

        valid_data_31 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/ValidationData/Stage4/*.csv'],
                                                              3000,
                                                              ['ch_' + str(i) for i in range(3)],
                                                              shuffle=False,
                                                              header=False)

        valid_data_4 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/ValidationData/StageR/*.csv'],
                                                             3000,
                                                             ['ch_' + str(i) for i in range(3)],
                                                             shuffle=False,
                                                             header=False)

        valid_data_1 = valid_data_1.map(obtain_data_0, tf.data.experimental.AUTOTUNE).take(22417)
        valid_data_2 = valid_data_2.map(obtain_data_1, tf.data.experimental.AUTOTUNE).take(1252)
        valid_data_21 = valid_data_21.map(obtain_data_1, tf.data.experimental.AUTOTUNE).take(5099)
        valid_data_3 = valid_data_3.map(obtain_data_2, tf.data.experimental.AUTOTUNE).take(414)
        valid_data_31 = valid_data_31.map(obtain_data_2, tf.data.experimental.AUTOTUNE).take(263)
        valid_data_4 = valid_data_4.map(obtain_data_3, tf.data.experimental.AUTOTUNE).take(1901)

        valid = valid_data_1.concatenate(valid_data_2).concatenate(valid_data_3).concatenate(valid_data_4)
        valid = valid.concatenate(valid_data_21).concatenate(valid_data_31)

        data = data.batch(256).map(normalize, tf.data.experimental.AUTOTUNE)
        valid = valid.cache('cache4').batch(128).map(normalize, tf.data.experimental.AUTOTUNE)

        data = data.prefetch(4)
        valid = valid.prefetch(8)
        
        
        model = construct_cnn()

        model = tf.keras.models.load_model('Parcial2Machine/CNNv4_normalizedbalanced.h5')

        model.summary()

        early_stop = tf.keras.callbacks.EarlyStopping('val_categorical_accuracy', patience=50, restore_best_weights=True, verbose=1)
        model.fit(data,
                  epochs=250,
                  verbose=2,
                  steps_per_epoch=4737,
                  validation_data=valid,
                  validation_steps=245,
                  callbacks=[early_stop])

        model.save('Parcial2Machine/CNNv4_normalizedbalanced.h5')

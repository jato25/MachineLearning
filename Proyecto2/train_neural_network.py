import tensorflow as tf
from feature_extraction import preprocessed_datadict_extraction
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential


if __name__ == '__main__':

    dataset1 = tf.data.experimental.make_csv_dataset('Parcial2Machine/TrainDataStageW.csv',
                                                    128,
                                                    ['f_' + str(i) for i in range(29)] + ['label'],
                                                    label_name='label',
                                                    shuffle=False,
                                                    header=False)

    dataset2 = tf.data.experimental.make_csv_dataset('Parcial2Machine/TrainDataStage1.csv',
                                                     128,
                                                     ['f_' + str(i) for i in range(29)] + ['label'],
                                                     label_name='label',
                                                     shuffle=False,
                                                     header=False)

    dataset3 = tf.data.experimental.make_csv_dataset('Parcial2Machine/TrainDataStage2.csv',
                                                     128,
                                                     ['f_' + str(i) for i in range(29)] + ['label'],
                                                     label_name='label',
                                                     shuffle=False,
                                                     header=False)

    dataset4 = tf.data.experimental.make_csv_dataset('Parcial2Machine/TrainDataStage3.csv',
                                                     128,
                                                     ['f_' + str(i) for i in range(29)] + ['label'],
                                                     label_name='label',
                                                     shuffle=False,
                                                     header=False)

    dataset5 = tf.data.experimental.make_csv_dataset('Parcial2Machine/TrainDataStage4.csv',
                                                     128,
                                                     ['f_' + str(i) for i in range(29)] + ['label'],
                                                     label_name='label',
                                                     shuffle=False,
                                                     header=False)

    dataset6 = tf.data.experimental.make_csv_dataset('Parcial2Machine/TrainDataStageR.csv',
                                                     128,
                                                     ['f_' + str(i) for i in range(29)] + ['label'],
                                                     label_name='label',
                                                     shuffle=False,
                                                     header=False)

    dataset1 = dataset1.map(preprocessed_datadict_extraction, tf.data.experimental.AUTOTUNE)\
        .unbatch().take(202288).cache().shuffle(1024)\
        .prefetch(20).repeat()

    dataset2 = dataset2.map(preprocessed_datadict_extraction, tf.data.experimental.AUTOTUNE) \
        .unbatch().take(11189).cache().shuffle(1024) \
        .prefetch(20).repeat()

    dataset3 = dataset3.map(preprocessed_datadict_extraction, tf.data.experimental.AUTOTUNE) \
        .unbatch().take(45123).cache().shuffle(1024) \
        .prefetch(20).repeat()

    dataset4 = dataset4.map(preprocessed_datadict_extraction, tf.data.experimental.AUTOTUNE) \
        .unbatch().take(3827).cache().shuffle(1024) \
        .prefetch(20).repeat()

    dataset5 = dataset5.map(preprocessed_datadict_extraction, tf.data.experimental.AUTOTUNE) \
        .unbatch().take(2333).cache().shuffle(1024) \
        .prefetch(20).repeat()

    dataset6 = dataset6.map(preprocessed_datadict_extraction, tf.data.experimental.AUTOTUNE) \
        .unbatch().take(17347).cache().shuffle(1024) \
        .prefetch(20).repeat()

    data = tf.data.experimental.sample_from_datasets([dataset1, dataset3, dataset6,
                                                      dataset2, dataset4, dataset5],[0.01,0.005,0.01,0.005,0.475,0.475]).batch(256)

    valid = tf.data.experimental.make_csv_dataset('Parcial2Machine/ValidationData.csv',
                                                  128,
                                                  ['f_' + str(i) for i in range(29)] + ['label'],
                                                  label_name='label',
                                                  shuffle=False,
                                                  header=False)

    valid = valid.map(preprocessed_datadict_extraction).unbatch().take(31346).cache().batch(128).repeat()

    layer_1 = Dense(18, input_shape=[29], activation='tanh')
    layer_2 = Dense(10, activation='tanh')
    layer_3 = Dense(4, activation='softmax')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model = Sequential([layer_1, layer_2, layer_3])

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
                  
    model = tf.keras.models.load_model('Parcial2Machine/Balanced_CostedNN.h5')

    model.fit(x=data,
              epochs=250,
              verbose=2,
              steps_per_epoch=20,
              validation_data=valid,
              validation_steps=245)

    model.save('Parcial2Machine/' + 'Balanced_CostedNN.h5')
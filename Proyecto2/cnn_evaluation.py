from feature_extraction import obtain_data_0, obtain_data_1, obtain_data_2, obtain_data_3, normalize
import tensorflow as tf

if __name__ == '__main__':
    test_data_1 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/TestData/StageW/*.csv'],
                                                        3000,
                                                        ['ch_' + str(i) for i in range(3)],
                                                        shuffle=False,
                                                        header=False)

    test_data_2 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/TestData/Stage1/*.csv'],
                                                        3000,
                                                        ['ch_' + str(i) for i in range(3)],
                                                        shuffle=False,
                                                        header=False)
                                                        
    test_data_21 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/TestData/Stage2/*.csv'],
                                                        3000,
                                                        ['ch_' + str(i) for i in range(3)],
                                                        shuffle=False,
                                                        header=False)

    test_data_3 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/TestData/Stage3/*.csv'],
                                                        3000,
                                                        ['ch_' + str(i) for i in range(3)],
                                                        shuffle=False,
                                                        header=False)
                                                        
    test_data_31 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/TestData/Stage4/*.csv'],
                                                        3000,
                                                        ['ch_' + str(i) for i in range(3)],
                                                        shuffle=False,
                                                        header=False)

    test_data_4 = tf.data.experimental.make_csv_dataset(['Parcial2Machine/TestData/StageR/*.csv'],
                                                        3000,
                                                        ['ch_' + str(i) for i in range(3)],
                                                        shuffle=False,
                                                        header=False)
    
    test_data_1 = test_data_1.map(obtain_data_0, tf.data.experimental.AUTOTUNE).take(56225)
    test_data_2 = test_data_2.map(obtain_data_1, tf.data.experimental.AUTOTUNE).take(3141)
    test_data_21 = test_data_21.map(obtain_data_1, tf.data.experimental.AUTOTUNE).take(12432)
    test_data_3 = test_data_3.map(obtain_data_2, tf.data.experimental.AUTOTUNE).take(1044)
    test_data_31 = test_data_31.map(obtain_data_2, tf.data.experimental.AUTOTUNE).take(634)
    test_data_4 = test_data_4.map(obtain_data_3, tf.data.experimental.AUTOTUNE).take(4888)

    test = test_data_1.concatenate(test_data_2).concatenate(test_data_3).concatenate(test_data_4)
    test = test.concatenate(test_data_21).concatenate(test_data_31).batch(128).map(normalize, tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.load_model('Parcial2Machine/CNNv4_normalizedbalanced.h5')

    predictions = []
    labels = []

    predictions_1 = []
    labels_1 = []

    for data, label in test:
        predictions += (tf.argmax(model.predict(data), axis=1).numpy().tolist())
        labels += (tf.argmax(label, axis=1).numpy().tolist())

        predictions_1 += model.predict(data).tolist()
        labels_1 += label.numpy().tolist()

    accuracy = tf.keras.metrics.CategoricalAccuracy()(labels_1, predictions_1)
    confusion = tf.math.confusion_matrix(labels, predictions)

    print('Presicion: ' + str(accuracy.numpy()))
    print(confusion)

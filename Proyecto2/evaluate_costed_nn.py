import tensorflow as tf
from feature_extraction import preprocessed_datadict_extraction


if __name__ == '__main__':

    models = [tf.keras.models.load_model('Parcial2Machine/CostedModel_' + str(i) + '.h5') for i in range(1,9)]

    test = tf.data.experimental.make_csv_dataset('Parcial2Machine/TestData.csv',
                                                 128,
                                                 ['f_' + str(i) for i in range(29)] + ['label'],
                                                 label_name='label',
                                                 shuffle=False,
                                                 header=False)

    test = test.map(preprocessed_datadict_extraction)        
    test = test.unbatch().take(78364).batch(128)

    predictions = []
    labels = []

    predictions_1 = []
    labels_1 = []

    for data, label in test:
        voting = tf.zeros((tf.shape(data)[0], 4))
        for model in models:
            voting += model.predict(data)

        voting = voting/8

        predictions += (tf.argmax(voting, axis=1).numpy().tolist())
        labels += (tf.argmax(label, axis=1).numpy().tolist())

        predictions_1 += voting.numpy().tolist()
        labels_1 += label.numpy().tolist()

    accuracy = tf.keras.metrics.CategoricalAccuracy()(labels_1, predictions_1)
    confusion = tf.math.confusion_matrix(labels, predictions)

    print('Presicion: ' + str(accuracy.numpy()))
    print(confusion)
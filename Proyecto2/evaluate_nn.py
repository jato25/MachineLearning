import tensorflow as tf
from feature_extraction import preprocessed_datadict_extraction


if __name__ == '__main__':

    model = tf.keras.models.load_model('Parcial2Machine/Balanced_CostedNN.h5')

    test = tf.data.experimental.make_csv_dataset('Parcial2Machine/TestData.csv',
                                                 128,
                                                 ['f_' + str(i) for i in range(29)] + ['label'],
                                                 label_name='label',
                                                 shuffle=False,
                                                 header=False)

    test = test.map(preprocessed_datadict_extraction).unbatch().take(78364).batch(128)

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
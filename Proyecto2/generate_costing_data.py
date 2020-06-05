from feature_extraction import preprocessed_datadict_extraction
import tensorflow as tf
import csv


def introduce_costs(data, label):
    return data, label, tf.reduce_sum(label * tf.constant([0.01, 0.01, 1, 0.01]), axis=1)


def save_costed_data(data, label, cost, writer):
    number = tf.random.uniform([], 0, 1)
    label = int(tf.reduce_sum(label * tf.constant([0.0, 1.0, 2.0, 3.0])).numpy())

    if number.numpy() < cost:
        row = data.numpy().tolist()
        row.append(label)
        writer.writerow(row)


if __name__ == '__main__':
    train = tf.data.experimental.make_csv_dataset('Parcial2_Machine/TrainData.csv',
                                                  128,
                                                  ['f_' + str(i) for i in range(29)] + ['label'],
                                                  label_name='label',
                                                  shuffle=False,
                                                  header=False)

    train = train.map(preprocessed_datadict_extraction)
    train = train.map(introduce_costs)

    train = train.unbatch().take(282107)
    train = train.shuffle(buffer_size=1024)

    file = open('Parcial2_Machine/CostedData/costed_data_1.csv', 'w', newline='')
    writer = csv.writer(file)

    for data, label, cost in train:
        save_costed_data(data, label, cost, writer)

    file.close()

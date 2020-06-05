from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
import tensorflow as tf
import os

IMAGE_SHAPE = (32, 32, 3)
CLASS_ONE = "Mundo1"

TRAIN_DATA_FILES = ["MarioData/TrainData/Mundo1/*.png", "MarioData/TrainData/Mundo2/*.png"]
VALIDATION_DATA_FILES = ["MarioData/ValidationData/Mundo1/*.png", "MarioData/ValidationData/Mundo2/*.png"]

CHECKPOINTS = "MarioLogs/Checkpoints/MarioLeNet/MarioLeNet_{epoch:04d}.h5"
TENSORBOARD = "MarioLogs/TensorBoard/MarioLeNet_0to60/"

BATCH_SIZE = 64


def process_dataset(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]
    label = label == CLASS_ONE
    label = tf.cast(label, dtype=tf.float32)

    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image)
    image = tf.image.resize(image, IMAGE_SHAPE[:2])
    image = (tf.image.convert_image_dtype(image, tf.float32) / 127.5) - 1

    return image, label


if __name__ == "__main__":
    MarioLeNet = Sequential(name='MarioLeNet')

    MarioLeNet.add(Conv2D(
        filters=6,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=IMAGE_SHAPE
    ))

    MarioLeNet.add(AveragePooling2D())

    MarioLeNet.add(Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation='relu'
    ))

    MarioLeNet.add(AveragePooling2D())

    MarioLeNet.add(Flatten())

    MarioLeNet.add(Dense(
        units=120,
        activation='relu'
    ))

    MarioLeNet.add(Dense(
        units=84,
        activation='relu'
    ))

    MarioLeNet.add(Dense(
        units=1,
        activation='sigmoid'
    ))

    train_dataset = tf.data.Dataset.list_files(TRAIN_DATA_FILES)
    train_dataset = train_dataset.map(process_dataset, tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    validation_dataset = tf.data.Dataset.list_files(VALIDATION_DATA_FILES)
    validation_dataset = validation_dataset.map(process_dataset, tf.data.experimental.AUTOTUNE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    Optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    Loss = tf.keras.losses.BinaryCrossentropy()

    MarioLeNet.compile(optimizer=Optimizer,
                       loss=Loss,
                       metrics=['binary_accuracy'])

    Checkpoints = tf.keras.callbacks.ModelCheckpoint(CHECKPOINTS,
                                                     save_best_only=True,
                                                     monitor='binary_accuracy')

    TensorBoard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD,
                                                 histogram_freq=2)

    tf.print("Iniciado entrenamiento")
    steps_per_epoch = 206

    MarioLeNet.fit(x=train_dataset,
                   shuffle
                   =True,
                   verbose=2,
                   callbacks=[Checkpoints, TensorBoard],
                   steps_per_epoch=steps_per_epoch,
                   epochs=60,
                   validation_data=validation_dataset,
                   initial_epoch=0)

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import tensorflow as tf
import os

IMAGE_SHAPE = (160, 160, 3)
CLASS_ONE = "Mundo1"

TRAIN_DATA_FILES = ["MarioData/TrainData/Mundo1/*.png", "MarioData/TrainData/Mundo3/*.png"]
VALIDATION_DATA_FILES = ["MarioData/ValidationData/Mundo1/*.png", "MarioData/ValidationData/Mundo3/*.png"]

CHECKPOINTS = "MarioLogs/Checkpoints/OneLayerModel/MarioMobileNet_{epoch:04d}.h5"
RUTA_CHECKPOINTS = "MarioLogs/Checkpoints/OneLayerModel/"
TENSORBOARD = "MarioLogs/TensorBoard/OneLayerModel_0to10/"

BATCH_SIZE = 64


def process_dataset(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    label = parts[-2]
    label = label == CLASS_ONE
    label = tf.cast(label, dtype=tf.float32)

    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image)
    image = tf.image.resize(image, (160, 160))
    image = (tf.image.convert_image_dtype(image, tf.float32) / 127.5) - 1

    return image, label


if __name__ == "__main__":
    MobileNet = InceptionV3(input_shape=IMAGE_SHAPE,
                            include_top=False,
                            weights='imagenet')

    MobileNet.trainable = False

    global_averge_pool = GlobalAveragePooling2D()

    dense_1 = Dense(units=1, activation='sigmoid', kernel_regularizer="l2")

    MarioMobileNet = Sequential([
        MobileNet,
        global_averge_pool,
        dense_1
    ])

    #MarioMobileNet = tf.keras.models.load_model("D:/MarioLogs/Checkpoints/OneLayerModel/MarioMobileNet_0003.h5")
    tf.print(MarioMobileNet.summary())

    train_dataset = tf.data.Dataset.list_files(TRAIN_DATA_FILES)
    train_dataset = train_dataset.map(process_dataset, tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    validation_dataset = tf.data.Dataset.list_files(VALIDATION_DATA_FILES)
    validation_dataset = validation_dataset.map(process_dataset, tf.data.experimental.AUTOTUNE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    Optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    Loss = tf.keras.losses.BinaryCrossentropy()

    MarioMobileNet.compile(optimizer=Optimizer,
                           loss=Loss,
                           metrics=['binary_accuracy'])

    Checkpoints = tf.keras.callbacks.ModelCheckpoint(CHECKPOINTS,
                                                     save_best_only=True,
                                                     monitor='binary_accuracy')

    TensorBoard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD,
                                                 histogram_freq=2)

    tf.print("Iniciado entrenamiento")
    steps_per_epoch = 80

    '''MarioMobileNet.fit(x=train_dataset,
                       shuffle=True,
                       verbose=2,
                       callbacks=[Checkpoints, TensorBoard],
                       steps_per_epoch=steps_per_epoch,
                       epochs=20,
                       validation_data=validation_dataset,
                       initial_epoch=0)'''

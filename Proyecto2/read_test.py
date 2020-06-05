import pyedflib as pyedf
import numpy as np
import csv
import os
import glob

CLASSES = {
    'Sleep stage W': 'StageW',
    'Sleep stage 1': 'Stage1',
    'Sleep stage 2': 'Stage2',
    'Sleep stage 3': 'Stage3',
    'Sleep stage 4': 'Stage4',
    'Sleep stage R': 'StageR',
    'Sleep stage ?': -1,
    'Movement time': -1
}

RUTA_ORIGEN = 'Parcial2Machine/Data/*Hypnogram.edf'
CARPETA_DESTINO_DATOS = 'Parcial2Machine/TrainData/'
s_WinSizeSec = 30


def process_file(ruta_archivo):
    name = ruta_archivo.split(os.path.sep)[-1]
    name = name.split('-')[0]
    name = name[:-1]
    # Lectura del archivo de estados de sueño (etiquetas)
    st_FileHypEdf = pyedf.EdfReader(ruta_archivo)

    # Datos en ventanas de 30 segundos,
    # v_HypTime es el tiempo de inicio, v_HypDur es la duración en un estado específico (pueden ser varias ventanas),
    # v_Hyp es la etiqueta.

    v_HypTime, _, v_Hyp = st_FileHypEdf.readAnnotations()

    # Lectura de las señales s_SigNum señales con nombres v_Signal_Labels
    st_FileEdf = pyedf.EdfReader('D:/Parcial2_Machine/Datos/' + name + '0-PSG.edf')

    # Conversion a segundos usando frecuencia de muestreo.
    SampleFrecuency_1 = st_FileEdf.getSampleFrequency(0)
    SampleFrecuency_2 = st_FileEdf.getSampleFrequency(5)

    Signal1 = st_FileEdf.readSignal(0)
    Signal2 = st_FileEdf.readSignal(1)
    Signal3 = st_FileEdf.readSignal(2)
    '''Signal4 = st_FileEdf.readSignal(3)
    Signal5 = st_FileEdf.readSignal(4)
    Signal6 = st_FileEdf.readSignal(5)
    Signal7 = st_FileEdf.readSignal(6)'''

    s_WinSizeSam_1 = np.round(SampleFrecuency_1 * s_WinSizeSec)
    #s_WinSizeSam_2 = np.round(SampleFrecuency_2 * s_WinSizeSec)

    # plot de señales en ventanas de 30s
    totalStates = len(v_Hyp)
    counter = 0

    #file = open(CARPETA_DESTINO_DATOS + name + '.csv', 'w', newline='')
    #writer = csv.writer(file)
    #list_to_save = []
    #labels = []

    for i in range(totalStates - 1):
        label = v_Hyp[i]
        label = CLASSES[label]

        if label == -1:
            continue

        StateStart_1 = int(v_HypTime[i]) * SampleFrecuency_1
        StateEnd_1 = int(v_HypTime[i + 1]) * SampleFrecuency_1

        '''StateStart_2 = int(v_HypTime[i]) * SampleFrecuency_2
        StateEnd_2 = int(v_HypTime[i + 1]) * SampleFrecuency_2'''

        Signal1Part = Signal1[StateStart_1:StateEnd_1]
        Signal2Part = Signal2[StateStart_1:StateEnd_1]
        Signal3Part = Signal3[StateStart_1:StateEnd_1]

        '''Signal4Part = Signal4[StateStart_2:StateEnd_2]
        Signal5Part = Signal5[StateStart_2:StateEnd_2]
        Signal6Part = Signal6[StateStart_2:StateEnd_2]
        Signal7Part = Signal7[StateStart_2:StateEnd_2]'''

        samples_1 = len(Signal1Part)

        initial_index_1 = 0
        final_index_1 = s_WinSizeSam_1

        '''initial_index_2 = 0
        final_index_2 = s_WinSizeSam_2'''

        while final_index_1 < samples_1:
            WindowSignal1 = Signal1Part[initial_index_1:final_index_1]
            WindowSignal2 = Signal2Part[initial_index_1:final_index_1]
            WindowSignal3 = Signal3Part[initial_index_1:final_index_1]

            '''WindowSignal4 = Signal4Part[initial_index_2:final_index_2]
            WindowSignal5 = Signal5Part[initial_index_2:final_index_2]
            WindowSignal6 = Signal6Part[initial_index_2:final_index_2]
            WindowSignal7 = Signal7Part[initial_index_2:final_index_2]'''

            signals_1 = np.array([WindowSignal1, WindowSignal2, WindowSignal3])
            #signals_2 = np.array([WindowSignal4, WindowSignal5, WindowSignal6, WindowSignal7])

            file_1 = open(CARPETA_DESTINO_DATOS + label + '/' + name + str(counter) + '.csv', 'w', newline='')
            #file_1 = open(name + str(counter) + '_1.csv', 'w', newline='')
            writer_1 = csv.writer(file_1)
            writer_1.writerows(signals_1.T)
            file_1.close()

            '''file_2 = open(CARPETA_DESTINO_DATOS + label + '\\' + name + str(counter) + '_2.csv', 'w', newline='')
            #file_2 = open(name + str(counter) + '_2.csv', 'w', newline='')
            writer_2 = csv.writer(file_2)
            writer_2.writerows(signals_2.T)
            file_2.close()'''

            '''signals_1 = tf.cast(signals_1, tf.complex64)
            fft_signal = tf.signal.fft(signals_1)

            power = tf.pow(tf.abs(fft_signal), 2)
            total_power = tf.reduce_sum(power, 1)
            delta_power = tf.reduce_sum(power[:, 0:120], 1) / total_power
            theta_power = tf.reduce_sum(power[:, 120:240], 1) / total_power
            alpha_power = tf.reduce_sum(power[:, 240:390], 1) / total_power
            beta_power = tf.reduce_sum(power[:, 390:660], 1) / total_power
            gamma_power = tf.reduce_sum(power[:, 660:1500], 1) / total_power
            mean_1, variance_1 = tf.nn.moments(tf.cast(signals_1, tf.float32), 1)
            mean_2, variance_2 = tf.nn.moments(tf.cast(signals_2, tf.float32), 1)

            features = tf.concat([mean_1,
                                  variance_1,
                                  delta_power,
                                  theta_power,
                                  alpha_power,
                                  beta_power,
                                  gamma_power,
                                  mean_2,
                                  variance_2], axis=0)

            labels.append(label)
            list_to_save.append(features.numpy().tolist())'''

            counter += 1
            initial_index_1 = final_index_1
            final_index_1 += s_WinSizeSam_1

            '''initial_index_2 = final_index_2
            final_index_2 += s_WinSizeSam_2'''

    '''labels = np.array(labels)
    labels = np.expand_dims(labels, axis=1)

    list_to_save = np.array(list_to_save)
    list_to_save = tf.cast(list_to_save, tf.float32)

    max_value = tf.reduce_max(list_to_save, 0)
    min_value = tf.reduce_min(list_to_save, 0)

    final_features = (list_to_save - min_value) / (max_value - min_value)

    final_features = final_features.numpy()
    final_features = np.concatenate([final_features, labels], axis=1)
    list_to_save = final_features.tolist()

    writer.writerows(list_to_save)
    file.close()'''
    return counter


if __name__ == "__main__":
    datos_totales = 0
    dataset = glob.glob(RUTA_ORIGEN)

    for element in dataset:
        print('Procesando: %s' % element)
        datos_totales += process_file(element)

    print('Total datos obtenidos: %d' % datos_totales)

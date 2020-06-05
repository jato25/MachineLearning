from feature_extraction import get_features
import csv
import numpy as np

if __name__ == '__main__':
    datos, etiquetas = get_features('Parcial2Machine/TrainData.csv')

    stages = ['StageW', 'Stage1', 'Stage2', 'Stage3', 'Stage4', 'StageR']
    files = [open('Parcial2Machine/TrainData' + stage + '.csv', 'w', newline='') for stage in stages]
    writers = [csv.writer(file) for file in files]
    total_datos = np.zeros(6)

    for dato, etiqueta in zip(datos, etiquetas):
        index = stages.index(etiqueta)
        writer = writers[index]
        total_datos[index] += 1
        
        etiqueta = np.array([etiqueta])
        dato = np.array(dato).astype(str)

        row = np.concatenate([dato, etiqueta])
        writer.writerow(row.tolist())

    for total, file, stage in zip(total_datos.tolist(), files, stages):
        print('Total datos %s: %d' % (stage, total))
        file.close()

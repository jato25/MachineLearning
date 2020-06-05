from feature_extraction import get_preprocessed_datasets, train_neuralnet


if __name__ == "__main__":
    for number in range(4,9):
        name = 'CostedModel_' + str(number) + '.h5'
    
        train, valid = get_preprocessed_datasets(number)
        print('Iniciando entrenamiento')
        train_neuralnet(train, valid, name)
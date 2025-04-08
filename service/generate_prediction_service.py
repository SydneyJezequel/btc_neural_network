import numpy as np
from service.prepare_dataset_service import PrepareDatasetService






class GeneratePredictionService:
    """ Affiche les résultats du modèle """


    def __init__(self):
        pass
        self.prepare_dataset_service = PrepareDatasetService()


    def predict_on_new_data(self, dataset_for_predictions, model, time_step=15):
        """ Génération de prédictions """

        # Préparation du dataset :
        dataset_for_predictions, scaler = self.prepare_dataset_to_predict(dataset_for_predictions, time_step)
        print("dataset for predictions : ", dataset_for_predictions)

        # Faire des prédictions :
        new_predictions = model.predict(dataset_for_predictions)
        print("new_predictions normalisées : ", new_predictions)
        new_predictions = scaler.inverse_transform(new_predictions)
        print("new_predictions finales : ", new_predictions)

        return new_predictions


    def prepare_dataset_to_predict(self, dataset, time_step):
        """ Préparation du dataset """

        # Préparation du dataset :
        tmp_dataset = self.prepare_dataset_service.format_dataset(dataset)
        tmp_dataset = self.prepare_dataset_service.delete_columns(tmp_dataset)

        # Normalisation :
        tmp_dataset_copy = tmp_dataset.copy()
        columns_to_normalize = ['Dernier']
        scaler = self.prepare_dataset_service.get_fitted_scaler(tmp_dataset_copy[columns_to_normalize])
        dataset = tmp_dataset
        print("dataset avant normalisation : ", dataset)
        normalized_datas = self.prepare_dataset_service.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
        dataset[columns_to_normalize] = normalized_datas
        print("dataset d'entrainement normalisé :", dataset)
        print("dataset shape : ", dataset.shape)

        # Suppression de la colonne date :
        del dataset['Date']

        # Création du dataset utilisé pour les prédictions :
        dataset = self.create_dataset_for_predictions(dataset, time_step)
        print("create_dataset_for_predictions : ", dataset)
        dataset = dataset.reshape(dataset.shape[0], dataset.shape[1], 1)
        print("after reshape : ", dataset)

        return dataset, scaler


    def create_dataset_for_predictions(self, dataset, time_step=1):
        """ Méthode qui génère le dataset utilisé pour les prédictions """
        dataX = []
        print("dataset before transforme : ", dataset)
        for i in range(len(dataset) - time_step):
            a = dataset.iloc[i:(i + time_step), 0].values
            dataX.append(a)
        return np.array(dataX)


    def insert_with_padding(self, values, start_index, total_length):
        """ Méthode pour insérer des prédictions avec padding """
        # Création d'un array avec des nan :
        arr = np.empty(total_length)
        arr[:] = np.nan

        end_index = start_index + len(values)
        if end_index > total_length:
            end_index = total_length
            values = values[:(end_index - start_index)]
        arr[start_index:end_index] = values.flatten()
        return arr

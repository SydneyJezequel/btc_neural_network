import numpy as np
from service.prepare_dataset_service import PrepareDatasetService






class GeneratePredictionService:
    """ Génère les prédictions du modèle """



    def __init__(self):
        pass
        self.prepare_dataset_service = PrepareDatasetService()



    # Génération des prédiction (V1) :
    """
    def predict_on_new_data(self, dataset_for_predictions, model, time_step=15):
        # Génération de prédictions
        # Préparation du dataset :
        dataset_for_predictions, scaler = self.prepare_dataset_to_predict(dataset_for_predictions, time_step)
        # Génération des prédictions :
        new_predictions = model.predict(dataset_for_predictions)
        new_predictions = scaler.inverse_transform(new_predictions)
        return new_predictions
    """



    def predict_on_new_data(self, dataset_for_predictions, model, scaler, time_step=15):
        """ Génération de prédictions """
        # Initialisation des variables :
        lst_output = []
        n_steps = time_step
        i = 0
        pred_days = 30

        # Préparation du dataset :
        # dataset_for_predictions, scaler = self.prepare_dataset_to_predict(dataset_for_predictions, time_step)

        # Convert DataFrame to NumPy array
        dataset_for_predictions = dataset_for_predictions.values

        # Préparation du dataset (2) :
        x_input = dataset_for_predictions[len(dataset_for_predictions) - time_step:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        print("x_input : ", x_input)

        # Génération des prédictions :
        while (i < pred_days):
            if (len(temp_input) > time_step):
                x_input = np.array(temp_input[1:])
                # Ensure x_input has the correct size
                x_input = x_input[-n_steps:]  # Take the last n_steps elements
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = np.array(temp_input)
                # Ensure x_input has the correct size
                x_input = x_input[-n_steps:]  # Take the last n_steps elements
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i = i + 1
        print("Output of predicted next days: ", len(lst_output))
        new_predictions = scaler.inverse_transform(lst_output)
        print("new_predictions: ", new_predictions)
        # new_predictions = lst_output
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
        normalized_datas = self.prepare_dataset_service.normalize_datas(tmp_dataset_copy[columns_to_normalize], scaler)
        dataset[columns_to_normalize] = normalized_datas
        # Suppression de la colonne date :
        del dataset['Date']

        # Ajustement dynamique de time_step
        if len(dataset) < time_step:
            time_step = len(dataset) - 1
            print(f"Ajustement de time_step à {time_step} car le dataset est trop petit.")

        # Création du dataset utilisé pour les prédictions :
        print("dataset : ", dataset)
        dataset = self.create_dataset_for_predictions(dataset, time_step)
        print("dataset_for_predictions : ", dataset)

        if dataset.size == 0:
            raise ValueError(
                "Le dataset pour les prédictions est vide. Assurez-vous que le dataset a suffisamment de lignes par rapport à time_step.")

        print("Shape before reshape:", dataset.shape)
        dataset = dataset.reshape(dataset.shape[0], dataset.shape[1], 1)
        return dataset, scaler



    def create_dataset_for_predictions(self, dataset, time_step=1):
        """ Méthode qui génère le dataset utilisé pour les prédictions """
        dataX = []
        for i in range(len(dataset) - time_step):
            a = dataset.iloc[i:(i + time_step), 0].values
            dataX.append(a)
        return np.array(dataX)



    def insert_with_padding(self, values, start_index, total_length):
        """ Méthode pour déplacer les prédictions avec du padding """
        # Création d'un array avec des nan :
        arr = np.empty(total_length)
        arr[:] = np.nan
        # Ajout du pading :
        end_index = start_index + len(values)
        if end_index > total_length:
            end_index = total_length
            values = values[:(end_index - start_index)]
        arr[start_index:end_index] = values.flatten()
        return arr

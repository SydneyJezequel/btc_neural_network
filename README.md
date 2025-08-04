PROJECT PRESENTATION :
* This project trains a neural network to predict the BTC price.
* It is trained with a dataset containing daily BTC closing prices from 2014 to 2025.
* The details of this reflection are summarized in the following documents in this repository :  Analyse - Modélisation du prix du Bitcoin avec un réseau de neurones.docx.pdf (french version), Reflection on Bitcoin price prediction modeling.docx.pdf (english version).


WHAT THE INDICATORS CORRESPOND TO : 
* Error metrics (RMSE, MSE, MAE) : They measure the difference between the values predicted by the model and the actual values. They are used to evaluate the precision of the model.
* Explained Variance Score (EVS) : It measures how much our predictions deviate on average from the actual values. An EVS close to 1 means that the model explains the data variance well.
* Coefficient of determination (R²) : It measures the proportion of the variance explained by the model in the actual data. A value close to 1 means that the relationships between the provided dataset and the obtained predictions are relevant.
* Mean Gamma Deviance (MGD) : It measures how the gaps between predictions and actual values vary according to the input data. This metric is useful for evaluating the stability of the model for different input data.
* Mean Poisson Deviance (MPD) : It indicates whether the model tends to overestimate or underestimate the actual values. A low MPD means the model makes precise predictions.


HOW TO ANALYZE THE INDICATORS :
* RMSE (Root Mean Squared Error), MSE (Mean Squared Error), MAE (Mean Absolute Error) : They should decrease during training but may be flat or increase in case of overfitting.
* EVS (Explained Variance Score) : It should approach 1 during training but may be flat and then decrease in case of overfitting.
* R² : It should approach 1 during training but may be flat and then decrease in case of overfitting.
* MGD (Mean Gamma Deviance) : Constant and low errors indicate a stable model.
* MPD (Mean Poisson Deviance) : It should decrease during training.


POINTS TO CONSIDER WHEN ANALYZING INDICATORS :
* Overfitting : If the model starts to overfit, performance metrics on the validation data may degrade. It's important to monitor these metrics and use techniques like early stopping to prevent overfitting.
* Plateau : The metrics may be flat. It indicates the model has reached its performance limits with the current data and configuration.
* Variability : Metrics can vary from one fold to another due to data variability. Cross-validation is useful for obtaining a more robust estimate of model performance.
* Disparity : A large disparity between the test and training results for RMSE, MSE, and MAE indicates that the model performs well on the training data but not on the test data.


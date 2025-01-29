PRESENTATION DU PROJET :
* Ce projet entraine un réseau de neurones pour prédire les cours du BTC.
* Il est entrainé sur un dataset qui contient les cours de clôture journalier 
du BTC pour les années de 2014 à 2025.


INDICATEURS UTILISES POUR EVALUER LA QUALITE DE L'ENTRAINEMENT :
* MSE, MSE, MAE : Plus les résultats des métriques RMSE, MSE et MAE seront proches entre les données de test et d'entraînement, moins il y aura de surajustement. Une grande disparité indique que le modèle réagit bien aux données d'entraînement mais moins aux données de test.
* VARIANCE REGRESSION : Le score de variance expliquée mesure la proportion de la variance dans les valeurs cibles expliquée par le modèle. Un score proche de 1 indique que le modèle explique bien la variance des données, tandis qu'un score proche de 0 indique le contraire.
* R2 : Le score R² mesure la proportion de la variance dans les valeurs cibles expliquée par le modèle. Un score R² proche de 1 indique que le modèle explique bien la variance des données, tandis qu'un score proche de 0 ou négatif indique que le modèle n'explique pas bien la variance ou est pire que la moyenne des valeurs cibles.
* MPD : La perte de déviance Gamma évalue à quel point le modèle prédit bien les temps de défaillance observés. Elle mesure la différence entre la déviance du modèle ajusté et la déviance d'un modèle nul. Le modèle ajusté utilise les variables indépendantes pour prédire la variable dépendante, tandis que le modèle nul prédit simplement la moyenne de la variable dépendante.


COMMENT ANALYSERS LES INDICATEURS :
* Validation RMSE (Root Mean Squared Error) : Mesure l'écart moyen entre les prédictions et les valeurs réelles. Elle devrait diminuer avec l'entraînement mais peut atteindre un plateau ou augmenter en cas de surapprentissage.
* Validation MSE (Mean Squared Error) : Mesure la moyenne des carrés des erreurs. Elle devrait également diminuer mais peut atteindre un plateau ou augmenter en cas de surapprentissage.
* Validation MAE (Mean Absolute Error) : Mesure l'erreur absolue moyenne entre les prédictions et les valeurs réelles. Elle devrait diminuer mais peut atteindre un plateau ou augmenter en cas de surapprentissage.
* Validation Explained Variance Score : Mesure la proportion de la variance dans les valeurs réelles expliquée par le modèle. Elle devrait se rapprocher de 1 mais peut atteindre un plateau puis diminuer en cas de surapprentissage.
* Validation R2 Score : Mesure la proportion de la variance dans les valeurs réelles prédite par le modèle. Elle devrait se rapprocher de 1 mais peut atteindre un plateau puis diminuer en cas de surapprentissage.
* Validation MGD (Mean Gamma Deviance) : Mesure la déviance moyenne pour une distribution gamma. Elle devrait diminuer mais peut atteindre un plateau ou augmenter en cas de surapprentissage.
* Validation MPD (Mean Poisson Deviance) : Mesure la déviance moyenne pour une distribution de Poisson. Elle devrait diminuer avec l'entraînement.


POINTS A CONSIDERER LORS DE L'ANALYSE DES INDICATEURS :
* Surapprentissage : Si le modèle commence à surapprendre, les métriques de performance sur les données de validation peuvent se dégrader. Il est important de surveiller ces métriques et d'utiliser des techniques comme l'arrêt précoce pour éviter le surapprentissage.
* Plateau : Les métriques peuvent atteindre un plateau, indiquant que le modèle a atteint ses limites de performance avec les données et la configuration actuelles.
* Variabilité : Les métriques peuvent varier d'un fold à l'autre en raison de la variabilité des données. La validation croisée est utile pour obtenir une estimation plus robuste de la performance du modèle. Plus les résultats des métriques RMSE, MSE et MAE seront proches entre les données de test et d'entraînement, moins il y aura de surajustement. Une grande disparité indique que le modèle réagit bien aux données d'entraînement mais moins aux données de test.











import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import parameters
from service.prepare_dataset_service import PrepareDataset






""" ****************************** Paramètres ****************************** """
DATASET_PATH = parameters.DATASET_PATH
DATASET_FILE = parameters.DATASET_FILE
# PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
# DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL
# SAVE_MODEL_PATH = parameters.SAVE_MODEL_PATH





print(" ************ Etape 1 : Loading dataset ************ ")
data = pd.read_csv(DATASET_PATH+DATASET_FILE)

# COLONNES DU DATASET :
# "Date","Dernier","Ouv."," Plus Haut","Plus Bas","Vol.","Variation %

# COLONNES QUI M'INTERESSENT :
# "Date", "Dernier", " Plus Haut", "Plus Bas"






print(" ************ Etape 2 : Preparation of the Dataset ************ ")


# Convertir la colonne "Date" au format datetime :
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')


# Formatage des colonnes numériques :
numeric_columns = ["Dernier", "Ouv.", " Plus Haut", "Plus Bas", "Variation %"]
for col in numeric_columns:
    data.loc[:, col] = data[col].str.replace('.', ' ').str.replace(' ', '').str.replace(',', '.')
    data[col] = pd.to_numeric(data[col], errors='coerce')


# Suppression des colonnes inutiles
data.drop(columns=['Vol.', 'Variation %'], inplace=True)


# Dataset pré-traitement :
print("dataset pré-traitement : ", data)


# Trier le DataFrame par la colonne 'Date'
data = data.sort_values(by='Date')


# Sélectionner les 300 derniers jours
last_300_days = data.tail(200)


# Création d'un DataFrame
df = pd.DataFrame(last_300_days)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# Conversion des données pour matplotlib
df['Date_Num'] = mdates.date2num(df.index)
ohlc = df[['Date_Num', 'Dernier', 'Ouv.', ' Plus Haut', 'Plus Bas']].values


# Création du graphique
fig, ax = plt.subplots()
candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r')


# Tracer une droite de tendance haussière
# ax.plot(df['Date_Num'], df['Plus Bas'].rolling(window=2).min(), color='blue', label='Tendance Haussière')


# Tracer une droite de tendance baissière
# ax.plot(df['Date_Num'], df[' Plus Haut'].rolling(window=2).max(), color='red', label='Tendance Baissière')


# Configuration du graphique
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.set_title('Graphique de Bougies avec Droites de Tendance')
ax.legend()
plt.xticks(rotation=45)
plt.show()

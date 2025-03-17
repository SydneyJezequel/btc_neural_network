import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import parameters

# Chargement et préparation des données
data = pd.read_csv(parameters.DATASET_PATH + parameters.DATASET_FILE)
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')

# Nettoyage des noms de colonnes
data.columns = data.columns.str.strip()

numeric_columns = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas", "Variation %"]
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col].str.replace('.', '').str.replace(',', '.'), errors='coerce')

data.drop(columns=['Vol.', 'Variation %'], inplace=True)
data = data.sort_values(by='Date')
last_300_days = data.tail(200)
df = pd.DataFrame(last_300_days)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Date_Num'] = mdates.date2num(df.index)
ohlc = df[['Date_Num', 'Dernier', 'Ouv.', 'Plus Haut', 'Plus Bas']].values

# Création du graphique
fig, ax = plt.subplots()
candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r')

# Fonction pour tracer les lignes de tendance hebdomadaires
def plot_weekly_trend_lines(df, ax):
    # Regrouper les données par semaine pour obtenir les points hauts et bas
    weekly_highs = df['Plus Haut'].resample('W').max()
    weekly_lows = df['Plus Bas'].resample('W').min()

    # Détecter les points de pivot hebdomadaires
    pivot_highs = weekly_highs.loc[(weekly_highs > weekly_highs.shift(1)) & (weekly_highs > weekly_highs.shift(-1))]
    pivot_lows = weekly_lows.loc[(weekly_lows < weekly_lows.shift(1)) & (weekly_lows < weekly_lows.shift(-1))]

    # Tracer les lignes de tendance haussière (supports) tous les trois points bas
    for i in range(0, len(pivot_lows) - 2, 3):
        if i + 2 < len(pivot_lows):
            # Prolonger la ligne de tendance
            start_idx, mid_idx, end_idx = pivot_lows.index[i], pivot_lows.index[i + 1], pivot_lows.index[i + 2]
            start_val, mid_val, end_val = pivot_lows.iloc[i], pivot_lows.iloc[i + 1], pivot_lows.iloc[i + 2]
            ax.plot([start_idx, end_idx, df.index.max()], [start_val, end_val, end_val], color='blue')

    # Tracer les lignes de tendance baissière (résistances) tous les trois points hauts
    for i in range(0, len(pivot_highs) - 2, 3):
        if i + 2 < len(pivot_highs):
            # Prolonger la ligne de tendance
            start_idx, mid_idx, end_idx = pivot_highs.index[i], pivot_highs.index[i + 1], pivot_highs.index[i + 2]
            start_val, mid_val, end_val = pivot_highs.iloc[i], pivot_highs.iloc[i + 1], pivot_highs.iloc[i + 2]
            ax.plot([start_idx, end_idx, df.index.max()], [start_val, end_val, end_val], color='red')

# Tracer les lignes de tendance hebdomadaires
plot_weekly_trend_lines(df, ax)

# Configuration du graphique
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.set_title('Graphique de Bougies avec Droites de Tendance Hebdomadaires')
plt.xticks(rotation=45)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import parameters
from scipy.stats import linregress

# Chargement et préparation des données
data = pd.read_csv(parameters.DATASET_PATH + parameters.DATASET_FILE)
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')
numeric_columns = ["Dernier", "Ouv.", " Plus Haut", "Plus Bas", "Variation %"]
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col].str.replace('.', '').str.replace(',', '.'), errors='coerce')
data.drop(columns=['Vol.', 'Variation %'], inplace=True)
data = data.sort_values(by='Date')
last_300_days = data.tail(200)
df = pd.DataFrame(last_300_days)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Date_Num'] = mdates.date2num(df.index)
ohlc = df[['Date_Num', 'Dernier', 'Ouv.', ' Plus Haut', 'Plus Bas']].values

# Création du graphique
fig, ax = plt.subplots()
candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r')

# Fonction pour tracer les lignes de tendance
def plot_trend_lines(df, ax):
    # Détecter les points de pivot
    pivot_highs = df.loc[(df[' Plus Haut'] > df[' Plus Haut'].shift(1)) & (df[' Plus Haut'] > df[' Plus Haut'].shift(-1))]
    pivot_lows = df.loc[(df['Plus Bas'] < df['Plus Bas'].shift(1)) & (df['Plus Bas'] < df['Plus Bas'].shift(-1))]

    # Tracer les lignes de tendance haussière
    for i in range(len(pivot_lows) - 1):
        slope, intercept, r_value, p_value, std_err = linregress(
            [pivot_lows.index[i].timestamp(), pivot_lows.index[i + 1].timestamp()],
            [pivot_lows['Plus Bas'].iloc[i], pivot_lows['Plus Bas'].iloc[i + 1]]
        )
        ax.plot(pivot_lows.index[i:i + 2], [pivot_lows['Plus Bas'].iloc[i], pivot_lows['Plus Bas'].iloc[i + 1]], color='blue')

    # Tracer les lignes de tendance baissière
    for i in range(len(pivot_highs) - 1):
        slope, intercept, r_value, p_value, std_err = linregress(
            [pivot_highs.index[i].timestamp(), pivot_highs.index[i + 1].timestamp()],
            [pivot_highs[' Plus Haut'].iloc[i], pivot_highs[' Plus Haut'].iloc[i + 1]]
        )
        ax.plot(pivot_highs.index[i:i + 2], [pivot_highs[' Plus Haut'].iloc[i], pivot_highs[' Plus Haut'].iloc[i + 1]], color='red')

# Tracer les lignes de tendance
plot_trend_lines(df, ax)

# Configuration du graphique
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.set_title('Graphique de Bougies avec Droites de Tendance')
plt.xticks(rotation=45)
plt.show()

import pandas as pd
import numpy as np






class TechnicalIndicatorsService:
    """ Calcul des indicateurs techniques """



    @staticmethod
    def ma(df, n):
        """ Calcul de la moyenne mobile """
        return pd.Series(df['Dernier'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))



    @staticmethod
    def rsi(df, period):
        """ Calcul du RSI """
        delta = df['Dernier'].diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        u[u.index[period-1]] = np.mean(u[:period])
        u = u.drop(u.index[:(period-1)])
        d[d.index[period-1]] = np.mean(d[:period])
        d = d.drop(d.index[:(period-1)])
        rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
        return 100 - 100 / (1 + rs)



    @staticmethod
    def calculate_signal(dataset, taille_sma1, taille_sma2):
        """ Calcul du signal croisement des sma """
        # Calcul des moyennes mobiles :
        sma1_col = 'MA_' + str(taille_sma1)
        sma2_col = 'MA_' + str(taille_sma2)
        signal_col = sma1_col +  '_supérieure_'  + sma2_col
        dataset[sma1_col] = TechnicalIndicatorsService.ma(dataset, taille_sma1)
        dataset[sma2_col] = TechnicalIndicatorsService.ma(dataset, taille_sma2)
        # Calcul des signaux :
        dataset[signal_col] = (dataset[sma1_col] > dataset[sma2_col]).astype(int)
        return dataset

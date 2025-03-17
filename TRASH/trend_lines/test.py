# Download Apple price history and save adjusted close prices to numpy array
# import pandas.io.data as pd
import pandas as pd
import parameters

""" ****************************** Paramètres ****************************** """
DATASET_PATH = parameters.DATASET_PATH
PATH_TRAINING_DATASET = parameters.PATH_TRAINING_DATASET
DATASET_FILE = parameters.DATASET_FILE
DATASET_FOR_MODEL = parameters.DATASET_FOR_MODEL

print("TEST : " + DATASET_PATH)



x = pd.read_csv(DATASET_PATH+DATASET_FILE)

# Make some trendlines
import trendy

# Generate general support/resistance trendlines and show the chart
# winow < 1 is considered a fraction of the length of the data set
trendy.gentrends(x, window =1.0 / 3, charts = True)

# Generate a series of support/resistance lines by segmenting the price history
trendy.segtrends(x, segments = 2, charts = True)  # equivalent to gentrends with window of 1/2
trendy.segtrends(x, segments = 5, charts = True)  # plots several S/R lines

# Generate smaller support/resistance trendlines to frame price over smaller periods
trendy.minitrends(x, window = 30, charts = True)

# Iteratively generate trading signals based on maxima/minima in given window
trendy.iterlines(x, window = 30, charts = True)  # buy at green dots, sell at red dots
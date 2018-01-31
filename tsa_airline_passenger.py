import pandas as pd 
from matplotlib import pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose
from pprint import pprint 
plt.style.use('fivethirtyeight')

# input
file = "international_airline_passengers.csv"
df = pd.read_csv(file)

# info
df.shape # (145, 2)
pprint(df.columns)
df.head()

# remove NAs
df.dropna(inplace=True)
df.shape # (144, 2)

# for tsa, we need a DateTimeIndex
df['Month'] = df['Month'].str[::]+"-01"
df['Month'] = pd.DatetimeIndex(df['Month'])
df.set_index('Month', inplace=True)

# rename column
df.columns = ["num_passengers"]

# tsa
result = seasonal_decompose(df.iloc[:,0], model='multiplicative', freq=1)
tsa_plot = result.plot()
plt.show

jupyter nbconvert --to html "Explainable AI - Questions.ipynb"
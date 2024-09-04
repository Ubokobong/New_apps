import warnings
warnings.filterwarnings("ignore")
from fastapi import FastAPI
from fastapi.responses import StreamingResponse  import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neural_network import MLPRegressor 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
app = FastAPI()

@app.get("/iris")
def get_iris():
  df = pd.read_csv('/Users/mac/NEW_aps/New_apps/HSBA Historical Data.csv')
  df['Vol.'] = df['Vol.'].replace('[\$,M]', '', regex=True).astype(float)
  df['Vol.'] = df['Vol.'].fillna(0).astype(int)
  df.duplicated()
  df['Date'] = pd.to_datetime(df['Date'])
  plt.figure(figsize=(14,5))
  time = np.array(df.Date)
  time = pd.to_datetime(time)
  sns.set_style("ticks")
  sns.lineplot(data=df,x= time,y='Price',color='firebrick')
  sns.despine()
  plt.title("HSBA Stock Price over the years",size='x-large',color='blue')
  fig.save('tix.png')
  file = open('tix.png, mode="rb")
  return StreamingResponse(file, media_type = "image/png")
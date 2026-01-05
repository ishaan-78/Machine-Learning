import pandas as pd
import plotly.express as px
from prophet import Prophet
import plotly.io as pio
pio.renderers.default='colab'

df = pd.read_csv('TSLA.csv', names=['Date','Open','High','Low','Close','Adj Close','Volume'],header=None)

px.area(df, x='Date',y='Close')
px.line(df, x='Date',y='Close')
px.area(df, x='Date',y='Volume')
px.bar(df, y="Volume")

columns=['Date','Close']
ndf = pd.DataFrame(df,columns=columns)
prophet_df = ndf.rename(columns={'Date':'ds', 'Close':'y'})
m = Prophet()
m.fit(prophet_df)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)
# forecast
px.line(forecast, x='ds', y='yhat')
figure = m.plot(forecast, xlabel='ds', ylabel='y')
figure2 = m.plot_components(forecast)


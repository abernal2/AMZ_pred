# AMZ_pred

Applied long-short-term-memory (LSTM) to predict amazon stock prices using 60 days of historical data to predict the next days closing price. The input features consisted of a total of 4: High, Low, Open, and Close prices instead of just a single feature.

The objective was to minimize the root mean squared error (RMSE) and after manually changing the learning rates, the predictions curve was a smoothed and lagged version of the true curve. It does not do great when we use the learned model with the previous predictions. Therefore, we would need to retrain the model every week. This might be because the dynamics are chaning within this time frame. The next step is to use other features like the 10 year treasury rate, LIFOR rate, and the company specific market and/or accounting features, which we can get from Yahoo Finance and SEC filings, respectively. 

# stock_market_analyzer

FEATURES Predicts the next day's stock price using 60-day historical data. Integrates with Yahoo Finance via yfinance for accurate and up-to-date market data. Incorporates sentiment analysis of recent news and social media to refine predictions.

INSTALL REQUIRED PACKAGES pip install -r requirements.txt

USAGE Download stock data and train the model: Modify the ticker_symbol variable in main.py to the desired stock ticker symbol (e.g., 'AAPL' for Apple). Run the model: python main.py View predictions: The model will output the predicted stock price for the next day based on the past 60 days of data.

DATA SOURCES Stock Data: Powered by Yahoo Finance using the yfinance library. Sentiment Data: Market sentiment is extracted from news sources and social media using an NLP-based sentiment analysis tool.

MODEL ARCHITECTURE Our model utilizes an LSTM neural network for time-series forecasting. The model is trained on a rolling window of 60-day stock price data, which captures temporal dependencies in the data to improve the prediction accuracy. LSTM Model Configuration Input Shape: (60 days, 1 feature) Layers: LSTM layers to capture time-series dependencies Dense layers for final prediction output Loss Function: Mean Squared Error (MSE)

SENTIMENT ANALYSIS INTEGRATION Incorporating sentiment analysis allows the model to account for public mood and market sentiment, providing a more comprehensive view of factors that may impact stock prices. We retrieve sentiment scores from recent articles and social media posts and adjust the prediction based on these scores. Sentiment analysis is currently powered by VADER (Valence Aware Dictionary and sEntiment Reasoner).

FUTURE WORK Enhance sentiment analysis: Use transformer-based NLP models for more accurate sentiment scores. Increase prediction horizon: Explore multi-day forecasting for 5- and 10-day horizons. Add feature engineering: Include additional financial indicators like Moving Average (MA) and RSI (Relative Strength Index) to enrich the input data.

# CS4641 - New COVID-19 Case Prediction
## Team Members: Matt Chen, Chima Okechukwu, Trevor Pope, and Sho Szczepaniuk
## Introduction and Background
Our society is facing an unprecedented pandemic, and we are interested in applying machine learning techniques to predict future COVID-19 cases in counties across the US. The United States currently has over 200,000 deaths and over 7.5 million coronavirus cases [1]. We will use unsupervised and supervised learning to make county's 2-day, 5-day, and 1-week forecasts of COVID-19 cases. 

## Methods
### Datasets
We plan on using a few key datasets in order to predict future COVID-19 cases. For COVID-19 data, we will be using [John Hopkin's COVID-19 Dataset](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data). It contains time series data for both COVID-19 Deaths and COVID-19 Cases at a county level and is updated daily. We plan on using [Google's Population Dataset](https://www.google.com/publicdata/explore?ds=kf7tgg1uo9ude_&hl=en&dl=en) as we believe that population statistics and economic indicators such as the unemployment rate will be good features for our model.

### Data Sanitization and Preprocessing
The datasets are uniform, and so the amount of sanitization will be minimal. However, since we are using multiple datasets as features, we will have to use the intersect of our databases that we're using. For time series analysis we can format the data so that we use prior time steps to predict next time steps ([sliding window method](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/) ).

### Feature Selection (maybe with Unsupervised Learning)
Tbh we don't have many features tho do we need unsupervised learning?

### Forecasting with Supervised Learning
Maybe Recurrent Neural Networks (like LSTM)
We can even use simpler Supervised Learning if we use sliding window.

## References
[1] "United States Coronavirus Cases," WorldoMeters, Accessed Oct 1, 2020. [Online] Available: https://www.worldometers.info/coronavirus/country/us/

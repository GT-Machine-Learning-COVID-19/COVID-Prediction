# CS4641 - New COVID-19 Case Prediction
## Team Members: Matt Chen, Chima Okechukwu, Trevor Pope, and Sho Szczepaniuk
## Introduction and Background
Our society is facing an unprecedented pandemic, and we are interested in applying machine learning techniques to predict future COVID-19 cases in counties across the US. The United States currently has over 200,000 deaths and over 7.5 million coronavirus cases [1]. We will use unsupervised and supervised learning to predict a county's 2-day, 5-day, and 1-week COVID-19 cases. 

## Summary Figure
Image: courtesy Urban Footprint
![prediction of hotspots](https://images.fastcompany.net/image/upload/w_1153,ar_16:9,c_fill,g_auto,f_auto,q_auto,fl_lossy/wp-cms/uploads/2020/08/p-1-90540870-these-states-are-likely-to-see-the-next-surge-in-covid-cases.jpg)
## Methods
### Unsupervised
We aim to find the most impactful features using Principal Component Analysis, in order to get a smaller, more powerful model [2]. Beacuse we are dealing with a problem on the scale on the entire United States, there will be a large multitude of features across counties that need to be pared down into only the most critial ones. In addition to the Google Population dataset listed below, we are also exploring the usage of various US Census issued datasets, as well as unemployment by county, which is tracked by the month. The biggest challege will be to migrate these geographical datasets into one comprehensive, geographical dataset that is able to map numerous features to the United States and allow us to compare COVID cases to them.

### Supervised
Usually, most Covid-19 predictions are based entirely on time series techniques such as Exponential Smoothing or ARIMA models, and don't bring in other features except the number of cases over time. Additionally, these models must be precisely tuned and can be difficult to get a very accurate prediciton. While we will experiment with these techniques, we ultimately want to get a more robust prediction based on several features (specified below). We want to try some multilinear regression methods, as well as some deep learning methods such as RNN and feedback networks like LSTM, to give us more predictive power and find complex relationships among features [3]. Since most Covid-19 data is presented as a timeseries, we would also like to try some sliding-window methods, where we consider several windows of time as discrete data points (i.e. days 1-7, 2-8, 3-9, and so on), which tend to produce good results when using in combination with neural network models [4].

### Datasets
We plan on using a few key datasets in order to predict future COVID-19 cases. For COVID-19 data, we will be using [John Hopkin's COVID-19 Dataset](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data). It contains time series data for both COVID-19 Deaths and COVID-19 Cases at a county level and is updated daily. We plan on using Google's [Population Dataset](https://www.google.com/publicdata/explore?ds=kf7tgg1uo9ude_&hl=en&dl=en) and [Economic/Unemployment Datasets](https://www.google.com/publicdata/directory#) as we believe that population statistics and economic indicators such as the unemployment rate will be good features for our model.

### Data Sanitization and Preprocessing
We're going to have to get the intersect of our databases that we're using. For time series analysis we can format the data so that we use prior time steps to predict next time steps ([sliding window method](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/) ).

### Feature Selection 
Within the various Google Datasets, at least 50 different domestic features are covered by the dataset. We anticipate that many of these features have little to do with COVID 19 and can be thrown away. To determine which features will be core to the creation of our model, we will be using Principal Compnent Analysis to determine which features have the highest relation to COVID case growth in the United States. Using the results from this portion of the project, we will be able to be more flexible with the methods we use as well as create a higher efficiency model when all is said and done. 

## Results
We're hoping to create a model that has >80% accuracy at predicting the amount of COVID cases 2 days, 5 days, and 1 week into the future for a given specific county. 

## Discussion 
Ideally, we would create a model that allows for communities around the United States to better prepare for COVID outbreaks given the current status of the country. We believe that our model would be best suited in serving marginalized groups, in particular those who may not have the means to properly prepare for COVID during all stages of the upcoming pandemic. With a model like ours, we would be able to forecast risk for these types of individuals and allow them to make educated decisions regarding the pandemic. From an institutional standpoint, a model like ours would allow for government at all levels to better equip communities in fighting the spread of the virus. If our model properly maps the next big potential hotspots, ideally, there is ample time for these types of institutions to issue warnings, stockpile PPE, or do any number of functions to lessen the severity of the outbreak. Past COVID, we believe our model may be extended to other health crisis, such as food or water shortages, as well as other pandemics [5].
 
## References
[1] "United States Coronavirus Cases," WorldoMeters, Accessed Oct 1, 2020. [Online] Available: https://www.worldometers.info/coronavirus/country/us/

[2] M. R. Mahmoudi, M. H. Heydari, S. N. Qasem, A. Mosavi, S. S. Bandh, "Principal component analysis to study the relations between the spread rates of COVID-19 in high risks countries" ScienceDirect, Sept 14, 2020. Available: https://doi.org/10.1016/j.aej.2020.09.013

[3] F. Shahid, A. Zameer, M. Muneeb, "Predictions for COVID-19 with deep learning models of LSTM, GRU and Bi-LSTM." ScienceDirect, Aug 19, 2020. Available: 
https://doi.org/10.1016/j.chaos.2020.110212 

[4] Hota, H. S., Richa Handa, and A. K. Shrivas. "Time series data prediction using sliding window based rbf neural network." International Journal of Computational Intelligence Research 13.5 (2017): 1145-1156. Available: http://www.ripublication.com/ijcir17/ijcirv13n5_46.pdf

[5] S. Hussein, "A Two-Layer Water Demand Prediction System in Urban Areas Based on Micro-Services and LSTM Neural Networks" ResearchGate, Aug 2020, DOI: 10.1109/ACCESS.2020.3015655 Available: https://www.researchgate.net/publication/343560899_A_Two-Layer_Water_Demand_Prediction_System_in_Urban_Areas_Based_on_Micro-Services_and_LSTM_Neural_Networks

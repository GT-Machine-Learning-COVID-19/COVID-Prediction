# Introduction and Background
The goal of this project is to be able to predict COVID-19 spikes or hotspots across the various counties in the State. The model makes the prediction using population data, social determinants, and current COVID-19 spread data. This report details the work on this project. The following sections discuss the datasets used, the unsupervised learning methods, the supervised learning methods, and closes with a conclusion.

Our focus on unsupervised learning was to gain as much insight into the underlying distribution and structure of COVID-19 cases in different counties. We analyzed the dataset and identified redundant features using PCA and a correlation matrix. This was to improve the accuracy of our model in the second part of our project, in which we will use different supervised learning methods to determine the amount of new COVID-19 cases in a specific county. Additionally, we separated similar data points into clusters and used the elbow method to find the optimal clusters. 

For the supervised portion, we compared the performance of ARIMA, LSTM, and Gradient Boosting in predicting the COVID cases for the past two weeks (11/22/2020 - 12/05/2020) and comparing those against the true number of cases. 

Being able to predict future COVID spikes would ideally facilitate an improvement in the response for affected areas. And, similar models could be used to predict other major population-specific events, such as another pandemic, shortages in food or water, or similar occurrences, and better prepare for those when the time comes. 

## Data Management
We primarily used two different types of data: a county feature dataset (e.g. demographics) and a cumulative COVID case dataset (i.e. number of total cases per county per day)
### Feature Dataset
The first step for us was to create a complete dataset of potentially useful features for every county. Our goal was to include as many relevant features that we could find. We used Census data for county level information about the population at the county level. We determined that labor statistics about counties could potentially be a useful feature so we used Bureau of Labor Statistics data. We hypothesized that mask mandates could impact COVID-19 cases, so we used a publicly available mask mandate dataset. The dataset consists of short descriptions of all mandates in different regions. Finally, we used a dataset about general socioeconomic information about counties that we found publically available on Github. We then combined all of the features from the data found for our dataset, which organized all this data on a ocunty-by-county basis.

![std_features df](/std_feat.png)

### COVID Case Dataset
The second primary dataset we used was the Johns Hopkins CSSE COVID-19 Dataset. From this dataset, we were able to derive day-by-day cumulative COVID-19 cases from 3132 (all but 11) counties of the United States, dating back to the first US COVID case on 01/22/2020 to the present day.

This dataset was primarily used for the supervised portion of the project, where we began tackling the regression problem of predicting future COVID cases based upon past data. To transform this dataset into a usable form for our supervised models, we created 10-day sliding windows. Essentially, a sliding window uses the previous n days as a feature space and the n+1th day as the label. The number of days was decided on a model-by-model basis

![covid df](/covid_df.png)

### Data Cleaning
We removed features from the datasets that were duplicates of each other after combining them, removed aggregations (i.e. many datasets had state level aggregations), and removed geographic information about each county. We also removed features or combined features that we knew were similar. For example, we combined multiple features about population composition into percent ethnic. After combining the different datasets, we ended up with a dataset with 17 features. For the mask mandate, it was very unstructured so we changed it to a binary encoding of 0 for no mask mandate and 1 for mask mandate. Finally, we removed any countries that had NaN as a value for any of the remaining features. This removal removed 11 of the 3243 counties in the United States, and we determined that this was an acceptable amount of counties to drop. Alternatively, we considered setting the NaNs to the mean value for each feature, but the 11 counties varied by a large margin in terms of population, so we decided against this.

# Unsupervised Learning
## Clustering
**Motivations**

We want to cluster similar counties together based on the features we gathered. As there are over 3,000 counties in the United States, being able to cluster similar counties into groups reduces the amount of models we will need to build during the next phase. Instead of predicting each county individually, we will predict cases on a cluster by cluster basis.. Our reasoning is that similar counties will have similar predictions (assuming that their cases are similar), and so it is viable to group them. Our aim is to cast a wide net of groups in order to capture a lot of types of counties, that way the predictions are more accurate later on.

**Results and Implications**

For clustering, we decided to cluster with the K-Means algorithm. The elbow method suggests that around 20 or so clusters would be “optimal”, but in our case, we aren’t necessarily trying to minimize clusters. Too few clusters and we run the risk of grouping together dissimilar counties, so we decided to go with a safer estimate of around 28 counties, as at this point the change in distortion became less than 100 per additional cluster.

![clustering](/elbow.png)

We found that after K=6, the K-Means algorithm will isolate singular counties into their own clusters. For example, after K=7, New York City is always placed into its own cluster. This brings some challenges for PCA, which requires N > 1 to run, but on its own isn’t a bad thing. New York being as big as it is likely deserves its own model in order to get an accurate prediction. 

These clusters would prove to be incredibly useful moving into the supervised learning portion, as they allowed us to group similar counties and create models based upon individual clusters, rather than every county at once.

## Data Analysis & Dimensionality Reduction
### Correlation Matrix

We calculated the standard correlation coefficient (also referred to as Pearson’s r) between all the pairs of attributes using scikit-learn’s corr() method. The figure below summarises the result. 

Overall, we were mostly interested with how much each attribute correlates with the mean of the time series data in the last 2 weeks. A summary of that result is shown below.

![correlation between covid and features](/covid_corr.png)

As shown, the population in a county and the amount of international migration show a strong positive correlation with the average number of reported cases. This suggests that the average number of covid cases tends to go up in more populated areas. On the other hand, some features as the ‘Percent Under Diploma’ have almost no effect on covid reporting. Removing features as this might improve the accuracy of the future model. We repeated the process with standardised data. The correlation matrix is unaffected by standardization of the data. 

### Principal Component Analysis
**Motivations**

Although there were 17 features in the socio-economic features dataset, we believed that Principal Component Analysis would serve as a useful tool to reduce the overall dimensionality of our dataset.

Because many of our features are intuitively and mathematically related to one other, we felt that PCA would be a valuable form of unsupervised learning to run on our dataset. For example, logically speaking, population, migration, and birth rates may possibly be collapsed into one dimension, and mathematically speaking, there is almost a perfect correlation between housing density and population density (as seen in the correlation matrix between the features below). 

![correlation matrix features](/feature_corr.png)

Therefore, there was value in running PCA to reduce the dimensionality of our overall dataset, which may improve the efficiency of our overall model during the supervised portion.

Prior to running PCA, we standardized the data set (as unscaled data will likely be dominated by a single component). To run PCA, we used the built in functions included in the sklearn library. From there, we can determine the number of principal components compared to their respective explained variance. We can determine the relative importance of the features in each principal component using the magnitude of the corresponding values in the eigenvectors. The heatmaps are created using the seaborn library.

**Results**

![% variance preserved](/variancemaintained.png)

% Variance preserved by Principal Component
1. *24.97*531181876283% of variance explained by 1 principal component
2. *43.81*650907804699% of variance explained by 2 principal components
3. *53.10i*025449664428% of variance explained by 3 principal components
4. *61.77*9260440099335% of variance explained by 4 principal components
5. *68.39*923510837932% of variance explained by 5 principal components
6. *74.63*354311471619% of variance explained by 6 principal components
7. *80.59*120966493887% of variance explained by 7 principal components
8. *85.30*547132180237% of variance explained by 8 principal components
9. *89.09*401930472255% of variance explained by 9 principal components
10. *92.56*520391781005% of variance explained by 10 principal components
11. *95.34*320685151206% of variance explained by 11 principal components
12. *97.20*373271702995% of variance explained by 12 principal components
13. *98.73*608659408484% of variance explained by 13 principal components
14. **99.93**06372785805% of variance explained by 14 principal components
15. **99.99**017280188369% of variance explained by 15 principal components
16. **99.99**989896978461% of variance explained by 16 principal components

95% of the variance is maintained at 11 components, 99% is maintained at about 13 components (minimum 14 components to maintain exactly 99% variance), and 99.99% is maintained at 15 components.

Feature Contribution to each Principal Component

![feature to each pc](/feature_contr.png)

By principal component, the most important feature is:
1. Percent Poverty
2. Percent Bachelors
3. Crime Rate Per 100k/Population Density
4. Death Rate
5. Percent with Only Diploma
6. Percent Poverty
7. Percent Under Diploma
8. Unemployment Rate
9. Percent Ethnic
10. Percent Ethnic
11. Mask Mandate
11. Mask Mandate
12. Labor Force
13. Mask Mandate
14. Percent with Only Diploma
15. Percent Some College
16. Death Rate

It can be observed that while population doesn’t define any single principal component, it heavily contributes to the first two principal components, which explain ~43% of the variance of the data set.

**Implications of PCA on only the feature dataset**

Running PCA allows us to reduce the dimensionality from 17 to 13, a reduction of 23.5% of dimensions while still maintaining 98.73% of the variance of the dataset.
The first principal component was primarily explained by percent poverty levels, implying that percent poverty is a distinguishing factor between counties. Similarly, the population was a large factor in contributing to the first and second principal components, which makes sense, as more populous counties will naturally have a different makeup than smaller counties.

After completing PCA on only the feature dataset, we realized that we had little intuition into why clusters were clustered as they were. So, to determine what features were most important to each cluster, we decided to run PCA once more on each cluster out of the 28. Since PCA only works when run on more than one county, the following results are for all of the clusters which contain multiple counties. We decided to employ a similar strategy as to how we interpreted the principal components for the overall feature dataset by listing the feature that contributed most to the first three principal components of each cluster.

![top 3 per pc per cluster](/top_3.png)

Overall, only 4 features on this chart show up multiple times, which are: Mask Mandate, Percent Poverty, Percent Ethnic, and Population, implying that these four features are likely the most important ones in the dataset when it comes to contributing to the variance within clusters of counties. While we originally anticipated that clusters would be defined by unique and different features, this result makes sense as well — from PCA run on all of the counties, it was evident that most of the feature space was relevant somehow to the variance of the dataset (99% variance was not attained until 14 features). Therefore, while these four features showed up the most in terms of being the highest contributor to the first three principal components, it must be acknowledged that the other features also play a significant role in the variance of the dataset and our clusters as a whole.
 
It was mentioned earlier that some clusters were skipped as they were singleton clusters (clusters with only one county) that could not support PCA. Those counties revolved around the regions of:

11: Los Angeles, CA

19: New York City, NY

20: Maricopa, AZ (Phoenix + Tucson)

24: Cook County, IL (Chicago)

From a pure qualitative standpoint, these four counties are defined by their status as major metropolitan areas. In fact, LA County, Maricopa County, and Cook County are the 1st, 2nd, and 4th largest counties by population in the entire United States.

Moving into the supervised learning portion, we kept this knowledge in mind and created cluster-specific models in some instances, which improved performance most of the time.

# Supervised Learning Methods

## ARIMA Modeling
**Introduction and Motivations**

ARIMA modeling is a classic technique for time series based data. These models rely on autocorrelations in the data to make a strong prediction for the data. It is always an excellent starting point because the models are simple to construct, but can be tricky to get an optimal model due to stationarity and autocorrelation constraints. 

**Data Usage and Methods**

ARIMA models require stationary data, which is data that has no cyclic behavior or trend and seasonality. Covid cases are acyclic in that there aren’t any major patterns of rising and falling cases within any of the clusters. There is however a strong upwards trend across all clusters, and so to remove this we differenced the data at various lags to lower the ACF (autocorrelation) plots. Below is the ACF for the first cluster before (on the left) and after differencing (on the right). 
![ACF](/sup1.png) ![ACF2](/sup2.png)

Before differencing, there is an extreme correlation at nearly every lag, implying our data is not stationary. After differencing the data twice, the ACF plot looks much better. You will notice a very large spike at lag 1 in the ACF plot, which is due to the fact that there are a significant number of days with zero cases at the beginning of the time series, and also indicates that there still might be a slight trend. However, this isn’t as much of a problem because ARIMA models of first-order and above are able to handle slight trends in the data as long as there isn’t any seasonality. And, a Box-Ljung test gives a good p-value which indicates the residuals are independent.

After differencing each cluster twice, we made 28 ARIMA models using mean COVID cases for each cluster. The process is as follows:

We used the Hyndman-Khandakar algorithm for fitting each ARIMA model, which is available in the Forecast package for R. Essentially, after differencing, we tested various candidate models that minimize the Akaike Information Criterion (AICc) score. After finding the one with the lowest AICc score, we checked our assumptions on residuals and autocorrelations again by checking the ACF and residual plots. For example, for the fourth cluster, the ARIMA parameters were (3, 2, 2), and the residual analysis was the following:

![residuals](/sup3.png)

The residuals are centered around zero which is good, and the variance is relatively stable, though slightly skewed towards the right. We think this is due to the large number of days with zero cases at the beginning of the time series. Overall, however, the residual assumptions are decent and the model is successful.

**Results and implications**

The model parameters for each cluster are as follows:

![params](/sup4.png)

You will notice that the d parameter, the number of lagged differences required for stationarity, is 2 for every model, which makes sense since we differenced the data twice before building the models.

In regards to accuracy, we tested the models on the last 14 days of data. On average, across all clusters, the RMSE was 750.62. Below a box plot showing the distribution of RMSE.

![boxplot](/sup5.png)

You will notice that there are 4 outliers with quite large RMSE. These were clusters 8, 7, 22, 5, and 11, which were larger clusters with few counties (such as New York, LA, etc). The models performed poorly for these due to extreme stationarity concerns. We first noticed that, for these counties, the trend was exponential rather than linear, and so the differencing techniques weren’t as effective as we had hoped. We tried to fix these by applying Box-Cox transformations (essentially, taking the log of the data), and then differencing the data, but the variance was not stable across the residuals, and the RMSE was even worse (around 1.5 times higher for most clusters). We conclude that ARIMA models aren’t as effective for time series large changes in trends (as is the case for curves with varying exponential growth across time), and so might not be the best choice for larger clusters. However, overall, the ARIMA models proved effective for most clusters. Below is a sample prediction for the next 14 days (as of 12/6/2020), with 95% confidence intervals.

![intervals](/sup6.png)

As the average RMSE per cluster for a baseline of predicting the previous day’s COVID cases for all 14 days was 4,600 (using the same methodology as outlined in the conclusion per cluster), the ARIMA models were effective in predicting COVID Cases.

## LSTM
**Introduction and Motivations**

LSTM is a modified version of Recurrent Neural Network that resolves the vanishing gradient problem by allowing the possibility for the gradient to flow back unchanged. LSTM is capable of doing so by replacing the hidden layers of a Recurrent Neural Network with memory cells. The LSTM cells have an input gate (It), an output gate (Ot), a forget gate (Ft), and an activation function that allow it to learn the behavior of temporal correlations.

![activation func](/sup6.5.png)

This allows for the LSTM to be well suited for sequential data such as video and speech, and as a result can be used for tasks such as speech recognition or handwriting recognition.

We realized that COVID-19 cases and deaths are sequence dependent, and researched different supervised learning techniques for time series data. We found that the LSTM RNN was a very effective and frequently used way to take into consideration previous timesteps, so we decided on trying to use it to predict future COVID-19 deaths.

**Data Usage and Methods**
Originally, we had planned on creating a multivariate LSTM RNN as we believed this would be the most effective way to capture . However, we realized that for the multivariate LSTM we would need time series data for all of our features. We ultimately decided on creating a univariate LSTM model as we concluded that trying to find time series data for all of the features would be very difficult. Additionally, we knew that the features would be reflected in our clustering, and so we decided we would create a model for each cluster instead, in addition to creating a model for all the data. 

For the LSTM, in addition to the data cleaning and sanitization that was done previously within the unsupervised learning portion, we normalized the data between 0 and 1. We realized that as opposed to creating a regular LSTM or a bi-directional LSTM, that a stacked LSTM would be the most optimal, as having multiple layers vertically stacked would allow for more model complexity.

**Hyperparameter Tuning**

When creating the LSTM, we had read online that adding dropout to our LSTM layers would help prevent overfitting. Typically, the dropout amount for each layer is 0.1 or 0.2 based on literature that we found online. However, when we added dropout, the RMSE scores more than tripled on average so we opted to remove the dropout. We decided on having four layers for our LSTM, as having fewer than four layers increased the error scores, and above four the improvements are nonexistent.


**Results and implications**

The results of the LSTM RMSE and MAE data indicate that our LSTM Recurrent Neural Network model is ineffective or inaccurate. Considering that the LSTM is a very common and effective RNN for time-series data, it is indicative that we did not tune the hyperparameters correctly, or did not have adequate data. A graph of the LSTM model and its prediction that uses the whole dataset is pictured below.

![activation func](/LSTM.png)

Similar to the ARIMA model, we predicted the past 14 days with this model, and observed a RMSE of 40169.25 for the general model, and an average RMSE of 36706.69 for models created for every one of the 28 clusters generated with the KMeans algorithm used in the unsupervised learning portion. The MAE for the general model was 30959.34 and the average for the cluster models was 25432.09.  It is important to note though that it was an improvement from the baseline RMSE of almost 129,000, but is significantly more than the RMSEs for the ARIMA models and Gradient Boosting models.

## Gradient Boosting
**Introduction and Motivations**

Gradient boosting is an iterative machine learning algorithm used for regression and classification tasks (prediction of future COVID cases given current COVID cases is a regression problem). Its strength lies in iteratively creating weak learners (often decision trees) and tuning feature importance after each iteration to create stronger and stronger learners. The way the additive model does this is by greedily minimizing the loss function of a base learner on a training set. Eventually, all trees are combined into one complete strong model.

We felt gradient boosting would be a good option for our problem space not only because of its ability to solve regression-type problems but also due to the results of data analysis PCA in the previous section. Based on the results of these sections, it could be concluded that several of the features in our feature space would be less relevant in creating our supervised models. Thus, by using gradient boosting, we would be able to phase out several of these less relevant features and create a more pure and accurate model. At the same time, this ensured that we would not need to cut out features or dimensions completely, losing out on information on our data.

While this reasoning ended up not being incorrect (see Data Usage and Methods section), gradient boosting ended up being a good choice regardless, due to its high level of flexibility when applied to different types of counties. For example, some models derived extremely high feature importance from the latest day (‘TS10’ >= 90%), while others had more balanced feature importances (note scale).

![feature_imp1](/sup7.png)

![feature_imp2](/sup8.png)

**Data Usage and Methods**

Since this is a regression type problem, the first step for preparing the data was the addition of sliding windows as described in part 1 of this paper. A window size of 10 was selected as the best window size after iteratively testing window sizes of 5 through 15. There is no point in standardizing the data, as gradient boosting uses trees and trees are agnostic to standardized values.

In an early iteration of our model, we discovered that the county demographic features had little to no impact on the model. In the feature importance plots (examples of which are displayed above), county demographic features were consistently ranked as the bottom features in predicting the next day’s COVID cases. So, we moved forward with our model with only the sliding window data.

After completing this ‘standard’ gradient boosting model that generally works for all US counties, we further subdivided counties in two different methods and created models for each division. 

For the first method, we created a set of 6 models, each of which predicts future COVID cases based upon the 6 National Center for Health Statistics 2013 Urban-Rural Classification codes. The primary motivation for doing so was due to the model not working very well for small-population, incredibly rural counties.

![maui county](/sup9.png)

Maui County (FIPS = 150009) COVID-19 cases with the Urban Rural code of 4

For the second method, we created a set of 28 models, each of which predicts future COVID cases for one of the 28 clusters from the unsupervised learning portion. Both methods led to an improvement in the predictive power of the model on a day-by-day predictive basis.

**Hyperparameter Tuning**

To find hyperparameters for the standard model, we employed an iterative method of testing values for hyperparameters within a range until the one with the best was selected, similar to how the best window size was selected for.

Before employing this method, to come up with initial approximate hyperparameters that were close to the ideal values for gradient boosting, we used hyperopt, which is essentially an optimizer that minimizes/maximizes any function. How it works it that it takes an input space of hyperparameters and gradually optimizes based on past results towards more ideal hyperparameters. Using this package provided a good baseline for us to start from.

![maui county](/sup10.png)
From there, we simply tested the hyperparameters in a range around the hyperopt outputs until the sufficiently best value was found.

For the urban-rural grouping and the cluster groupings, we used the same hyperparameter set for each method, and primarily focused on reducing the randomness of the hyperparameters. This primarily included adjusting the subsample and the learning rate. The reasoning behind this was because due to these county-level divisions, each model is dealing with less, more purified data, meaning that subsample, the proportion of number of features used for a given iteration, should be more, as there is a higher chance of losing out on critical information (as opposed to a higher chance of overfitting for the standard model). 

While for the urban-rural grouping we only adjusted subsample, for the cluster groupings, we took it a step further and increased the learning rate as well. As the sets of counties are even smaller and purer now, we felt more secure increasing this hyperparameter as the risk of overfitting is less.

![standard](/sup11.png)
![ur](/sup12.png)
![clusters](/sup13.png)

*70 estimators was chosen with the elbow rule: on the standard mode, the test RMSE began flattening out and having negligible change around 70 estimators.*

**Results and implications**

Overall, the results indicated great promise for our gradient boosting model. While we used RMSE as the objective function evaluation method for the creation of the model, we did not use this as the primary metric to determine the efficacy of the model. Instead, for each county for each day, we input the past 10-day COVID data and compared it against the true value of the COVID cases for that day. With the standard model, we were able to achieve a RMSE of 1586 and a MAE of 5.50 over 300+ days of COVID data. With the urban rural models and cluster models, we were able to achieve a RMSE of 1305 and 1463 and a MAE of 4.14 and 3.03, respectively. In context, this indicates that our model is only off by ~1500 cases for the entire lifespan of COVID per county, and is on average off by ~5 cases per county when predicting one day ahead. (*this metric is different than the standard metric used to measure all of our supervised models*)

Identical to the ARIMA model, we predicted the past 14 days on this model (11/22 - 12/05) and observed a RMSE of 872.6351131275891, 962.2865918394436, and 1081.2541614145837 for the respective standard model, urban rural models, and cluster models. Contextualized, this indicates that the gradient boosting models were off by ~900 cases on average in America across 14 days.

While the above results should be sufficient evidence that the model performs at a solid level, to confirm that these results are truly representative, we created our model using 10-fold validation (splitting the data up into 10 sections, and training on 9 parts and testing on the remaining part 10 times over), a method that was corroborated by other COVID predictive models [8]. These results are only for the standard model, but as the urban rural and cluster set models were both loosely based around the standard model, these results confirm that our model in particular does not experience abnormal results. There may be a worry with overfitting, as the model training score is much lower than the corresponding cross validation score.

![kfold](/sup14.png)

To help visualize these predictions and have an actual deliverable product for our project, we also created a static site that displays the 10 COVID case input for each county and the projected 7 days based upon those cases, and based on anecdotal experience, the results seem to match up relatively well with reality. https://gradient-boost-covid.herokuapp.com/. There are several counties in which the cluster models seem to have broken, but we have confirmed that the standard and urban-rural model data are all up to date. The screenshots on this report are using covid data up to 12/01/2020, but after completion of this report, the site will be updated with more modern data.

![site](/sup15.png)

Several limitations of this model include the data it was trained on. The metric with which we are measuring our supervised models is the RMSE for the next 14 days, indicating that the model’s labels should have been set for the 14 days after the window. However, we experimented a larger label size of 7-day predictions, and it simply took too long for the model to be created, and we were unable to allocate the necessary resources in tuning hyperparameters and refining this model. We are curious as to if this changed label would improve this model.

Off of this, while we attempted to adjust hyperparameters to minimize overfitting as much as possible, overfitting is likely still an issue on our model, especially for the urban-rural and cluster models. As the 14-day projection indicates, the standard model performs better than either, which likely means that the higher specificity models overfit. Additionally, while literature supported the use of k fold cross-validation in this type of problem, it would also be valuable to use time-sensitive forms of validation, like forward-chaining.

# Conclusion
In the unsupervised portion, we analyzed the feature space of our county datasets, creating clusters that would ultimately fine-tune and improve the efficacy of our supervised models.

In the supervised portion, we began tackling the problem space of predicting future COVID-19 cases, applying the three methods of LSTM, ARIMA, and Gradient Boosting and predicted the cases of the past two weeks as a point of comparison.

To compare our models against a baseline, we simply assumed that the next two week’s COVID cases would remain the same. In this case, the mean RMSE was approximately 4600.59 cases over every cluster.

Here are the overall results:

![results](/results.png)

On a per-cluster basis, ARIMA models had the lowest RMSE overall, However, there were several outliers for RMSE, particularly for the larger clusters with larger counties, which are arguably the most important ones. For clusters with smaller populations, i.e. smaller counties, the ARIMA model suggests promising results with an RMSE of 750.62. The LSTM models performed the worst for the per cluster average RMSE, with an average of 36706.69. This is indicative that the hyperparameters were not tuned correctly or the model was deployed incorrectly. As it was our first time using Keras and utilizing an LSTM model the latter is the more likely case.

The ARIMA models worked well over clusters but often fell short for individual counties due to stationarity constraints. Therefore, overall, gradient boosting was the best model for predicting COVID cases on a per-county basis with an average RMSE of approximately 900 for the models generated there, however, there is value in using ARIMA models for clusters that include many of the smaller counties of the United States. 

# Works Cited
TODO LATER

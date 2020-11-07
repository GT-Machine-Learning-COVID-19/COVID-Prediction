# CS4641 - New COVID-19 Case Prediction Midterm Report -- Unsupervised Learning
## Team Members: Matt Chen, Chima Okechukwu, Trevor Pope, and Sho Szczepaniuk

## Introduction and Background
This report and the corresponding work done represents the first half of our overall project. Our focus was on using unsupervised learning to learn more about the underlying distribution and structure of COVID-19 cases in different counties. Our goal consisted of gaining insight on the different features and reducing the number of unnecessary features using PCA and creating a correlation matrix.This will improve the accuracy of our model in the second part of our project, in which we will use different supervised learning methods to determine the amount of new COVID-19 cases in a specific county. Additionally, we wanted to cluster similar data together so that for the supervised learning
## Data Management
### Data Used
The first step for us was to create a complete dataset of potentially useful features for every county. Our goal was to include as many relevant features that we could find. We used Census data for county level information about the population at the county level. We determined that labor statistics about counties could potentially be a useful feature so we used Bureau of Labor Statistics data. For data about COVID-19 cases both currently and previously, we used data from  John Hopkins University’s CSSE Github. We hypothesized that mask mandates could impact COVID-19 cases, so we used a publicly available mask mandate dataset. The dataset consists of short descriptions of all mandates in different regions. Finally, we used a dataset about general socioeconomic information about counties that we found publically available on Github. We then combined all of the features from the data found for our dataset.
### Data Cleaning
We removed features from the datasets that were duplicates of each other after combining them, removed aggregations, and removed geographic information about each county. After combining the different datasets, we ended up with a dataset with 17 features. For the mask mandate, it was very unstructured so we changed it to a binary encoding of 0 for no mask mandate and 1 for mask mandate. Finally, we removed any countries that were missing any of the remaining features, which are the ones we deemed as potentially relevant. This removal removed 11 of the 3243 counties in the United States. In order to avoid the issue of scale and magnitude for PCA, we also standardized all of the data.
## Clustering
*Motivations*
We want to cluster similar counties together based on the features we gathered. As there are over 3,000 counties in the United States, being able to cluster similar counties into groups reduces the amount of models we will need to build during the next phase. Instead of predicting each county individually, we will predict cases on a cluster by cluster basis.. Our reasoning is that similar counties will have similar predictions (assuming that their cases are similar), and so it is viable to group them. Our aim is to cast a wide net of groups in order to capture a lot of types of counties, that way the predictions are more accurate later on.

*Results and Implications*
For clustering, we decided to cluster with the K-Means algorithm. The elbow method suggests that around 20 or so clusters would be “optimal”, but in our case, we aren’t necessarily trying to minimize clusters. Too few clusters and we run the risk of grouping together dissimilar counties, so we decided to go with a safer estimate of around 28 counties, as at this point the change in distortion became less than 100 per additional cluster.

We found that after K=6, the K-Means algorithm will isolate singular counties into their own clusters. For example, after K=7, New York City is always placed into its own cluster. This brings some challenges for PCA, which requires N > 1 to run, but on its own isn’t a bad thing. New York being as big as it is likely deserves its own model in order to get an accurate prediction. 

## Data Analysis & Dimensionality Reduction
### Correlation Matrix
We calculated the standard correlation coefficient (also referred to as Pearson’s r) between all the pairs of attributes using scikit-learn’s corr() method. The figure below summarises the result. 

Overall, we were mostly interested with how much each attribute correlates with the mean of the time series data in the last 2 weeks. A summary of that result is shown below.

As shown, the population in a county and the amount of international migration show a strong positive correlation with the average number of reported cases. This suggests that the average number of covid cases tends to go up in more populated areas. We repeated the process with standardised data. The correlation matrix is unaffected by standardisation of the data. 
### Principal Component Analysis
*Motivations*
Although there were 17 features in the socio-economic features dataset, we believed that Principal Component Analysis would serve as a useful tool to reduce the overall dimensionality of our dataset.
Because many of our features are intuitively and mathematically related to one other, we felt that PCA would be a valuable form of unsupervised learning to run on our dataset. For example, logically speaking, population, migration, and birth rates may possibly be collapsed into one dimension, and mathematically speaking, there is almost a perfect correlation between housing density and population density (as seen in the correlation matrix between the features below).

Therefore, there was value in running PCA to reduce the dimensionality of our overall dataset, which may improve the efficiency of our overall model during the supervised portion.
Prior to running PCA, we standardized the data set (as unscaled data will likely be dominated by a single component).
*Results*
% Variance preserved by Principal Component
1. 24.97531181876283% of variance explained by 1 principal component
2. 43.81650907804699% of variance explained by 2 principal components
3. 53.10025449664428% of variance explained by 3 principal components
4. 61.779260440099335% of variance explained by 4 principal components
5. 68.39923510837932% of variance explained by 5 principal components
6. 74.63354311471619% of variance explained by 6 principal components
7. 80.59120966493887% of variance explained by 7 principal components
8. 85.30547132180237% of variance explained by 8 principal components
9. 89.09401930472255% of variance explained by 9 principal components
10. 92.56520391781005% of variance explained by 10 principal components
11. 95.34320685151206% of variance explained by 11 principal components
12. 97.20373271702995% of variance explained by 12 principal components
13. 98.73608659408484% of variance explained by 13 principal components
14. 99.9306372785805% of variance explained by 14 principal components
15. 99.99017280188369% of variance explained by 15 principal components
16. 99.99989896978461% of variance explained by 16 principal components

95% of the variance is maintained at 11 components, 99% is maintained at about 13 components (minimum 14 components to maintain exactly 99% variance), and 99.99% is maintained at 15 components.

Feature Contribution to each Principal Component

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
17. An observation is that while population doesn’t define any single principal component, it heavily contributes to the first two principal components, which explain ~43% of the variance of the data set.

*Implications of PCA on only the feature dataset*
Running PCA allows us to reduce the dimensionality from 17 to 13, a reduction of 23.5% of dimensions while still maintaining 98.73% of the variance of the dataset.
The first principal component was primarily explained by percent poverty levels, implying that percent poverty is a distinguishing factor between counties. Similarly, the population was a large factor in contributing to the first and second principal components, which makes sense, as more populous counties will naturally have a different makeup than smaller counties.


## Conclusion
What did we learn from the unsupervised learning portion?

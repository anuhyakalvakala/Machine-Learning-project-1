# Machine-Learning-project-1
Projects on ML
Used scikit-learn (sklearn) library to generate and evaluate decision trees and their ensembles.

**Dataset Description:**

•	Data is from the EEG Neuroheadset from one continuous EEG measurement where each measurement was for 117 seconds. 
•	The targets are mentioned as 1 and 2 which are eye state of the user, and it was included in the dataset manually detected by the camera during EEG measurement
•	'1' indicates the eye-closed and '0' the eye-open state.
•	The features correspond to 14 EEG measurements from the headset, originally labeled AF3, F7, F3, FC5, T7, P, O1, O2, P8, T8, FC6, F4, F8, AF4, in that order.
•	Data has 14980 rows and 14 columns
•	Columns are named as ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V1','V12','V13','V14']

**Description:**

•	We are using sklearn library for importing datasets, trees, metrics, model selection and ensembles
•	Datasets was fetched and stored in a variable data and its features names are stored in features as a list, data values are stored in data_val with pandas.core.frame.Dataframe and its target values are stored in data_target as pandas.core.series.series
•	The tree was classified using decision tree classifier using default criterion as Gini and stored into mytree and with criterion as entropy and stored in mytree_gini for calculating information gain.

**Training and Testing Data:**

•	The Data was trained using the fit method on both criterions.
•	The data was predicted with the values of the data and predictions will be as 1 and 2

**Evaluation:**

•	Data has been evaluated and values of accuracy, F1- measure ,Precision and Recall has been measured using two criterions which resulted 1
•	As data was given 1 which shows it is overfitting the data. To fix this we have cross validated the data with the value of 9.
•	We got the mean scores of 0.53 with Gini and 0.54 with entropy.
• We have tunned our parameters with 10,20,30,40,50 to optimize our model architecture.
•	The optimization of model was done with GridsearchCV and cross validated with value of 9 and mean of 0.53 for Gini and was tuned on 10 and for entropy mean was 0.53 and tuned on 50.

**Ensemble:**
•	Ensembelling was done using Bagging, Random Forest and Adaboost to remove bias and variances in individual models
•	Bagging was done using bagging Classifier and a mean of 0.54 was observed with criterion as Gini.
•	Random forest is done using random forest classifier with default criterion as Gini and observed a mean of 0.55
•	Boosting was done in a sequential model learning from mistakes from the previous models.
•	Adaboost was done using ada-boost classifier with cross validation 0f 9 and observed a mean score of 0.53

**Evaluation measures:**
•	Basic decision tree using accuracy, F1 measure, precision, and recall which all are equal to one
•	After cross validation we can find accuracy and roc_auc measures.
•	During parameter tunning we did roc_auc for tuned trees, Random Forest, Bagging, AdaBoost and found their mean scores


**Cross validation-Mean scores**

Normal DT	         0.53
Normal DT	         0.54
Tuned DT -Gini	   0.53
Tuned DT -entropy  0.53
Random Forest	     0.55
Bagging	           0.54
Ada Boost	         0.53

**Results and conclusions:**

The data was trained and cross validated on different algorithms and found a mean score for all algorithms which are all almost same value near 0.5 which better after ensemble than the value at tunning which was 0.55





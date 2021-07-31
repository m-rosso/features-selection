## Empirical tests

This folder contains codes and datasets for testing among different features selection methods. The main file is the notebook "Features Selection - Empirical Tests", which brings codes for implementing the tests together with their final outcomes. Below, the objective of these tests, besides of their methodology and main findings are reproduced from that notebook.

--------------------------------------------------------------------------------------------------------------------------------------------------------------
Once features selection has been discussed and given the developed class *FeaturesSelection*, which groups alternative methods, this notebook tries to assess which approach is the most adequate for the **regression problem** provided by the [Communities and Crime Unnormalized](https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized) dataset obtained from the UCI Machine Learning Repository.
<br>
<br>
This dataset has 18 potential (continuous) target variables and 125 original features (one of which is a categorical variable that gives rise to additional input variables). In this empirical application, the variable chosen as output is "ViolentCrimesPerPop", the total number of violent crimes per 100.000 population. Each row of the dataset represent a unique instance, which consists of a community from US cities. Crime data refers to 1995 and comes from the FBI, while demographics refers to the 1990 Census. The main advantages of this dataset are its limited amount of observations, simplifying tests as estimations require less amount of memory and time, and the moderately high number of features.
<br>
<br>
When it comes to the **methodology of tests**, the following procedures are executed in order to produce, from raw data, outcomes that may help pointing to the most adequate approach of features selection for the regression problem at hand:
1. *Train-test split:* data is shuffled and, then, 25% is kept held out as test data, while the first 75% of the data is used not only for training models, but also for any calculation needed during data preparation.
2. *Data pre-processing:* features are classified (continuous, binary, categorical) and an early selection is implemented in order to drop variables with an excessive number of missings (more than 95% of instances of training data) and variables with no variance. Then, missing values are assessed and transformations take place: logarithmic transformation and standard scaling for numerical features, besides of the outcome variable. Missing values are treated as follows: a new category is created for missings in the categorical variable, while 0 is imputed for missings in numerical variables, plus the creation of binary variables indicating the existence of missings. Finally, the categorical variable is transformed as one-hot encoding is applied.
3. *Features selection:* all methods presented in notebook "Features Selection - Tutorials" are covered here: variance and correlation screening, supervised learning selection and exaustive methods (RFE, RFECV, SequentialFeatureSelector, random selection). Below, we find the complete grid of approaches to be tested (which may involve two or more methods sequentially).
4. *Model training:* two learning algorithms were picked for training models, Lasso (a linear, regularized method) and XGBoost (more flexibly, boosted models). Their hyper-parameters (regularization parameter for Lasso; subsample parameter, learning rate, maximum depth and number of estimators in the ensemble for XGBoost) are defined using K-folds cross-validation over the training data.
5. *Model evaluation:* the following performance metrics are calculated on the test data so the best approach can be identified: RMSE, R2, MAE, MSLE.

Is crucial to notice that features selection is inserted into each iteration of the K-folds CV estimation. When the final model is trained using the best hyper-parameters, a new selection of features takes place using the entire training data. Consequently, the *FeaturesSelection* class is not directly used here. Instead, the *KfoldsCV_fit* class (available in my [Github](https://github.com/m-rosso/validation)) proceeds to an aggregation of classes, since it initializes an object of that class previously to the model training based on train-validation split at each iteration of K-folds, and a final initialization previously to the training of the final model.
<br>
<br>
This is important to avoid the [Freedman paradox](https://www.alexejgossmann.com/Freedmans_paradox/), which would occur if only one features selection was implemented using the entire training data: when training models at each K-folds iteration, information referring to the validation data would improperly be used for hyper-parameters definition.

The following **approaches for features selection** are tested in this notebook of empirical tests:
* Single methods:
    * Variance thresholding.
    * Correlation thresholding.
    * Supervised learning selection (using a linear estimator).
    * RFE.
    * RFECV.
    * Sequential selection (only forward-stepwise selection).
    * Random selection (for each model size, a random set of features is selected; then, the best model is defined).


* Combined methods:
    * Variance or correlation thresholding and supervised learning selection.
    * Variance or correlation thresholding and RFE.
    * Variance or correlation thresholding and RFECV.
    * Variance or correlation thresholding and sequential selection (forward-stepwise selection).
    * Variance or correlation thresholding and random selection.

By implementing these empirical tests, we find that features selection has not a strong impact in predictive performance for this learning task. Even so, at least competitive metrics are obtained with shorter computing times and with less complex models. Therefore, features selection was able to reduce complexity of models while preserving generalization capacity. The **main conclusions** from the tests are summarized below:
* *Performance metrics*: even that very similar results are found, supervised learning selection seems to be the best choice for this learning task given both algorithms used during tests.
	* Besides, no selection of features has only the 9th highest R2 for XGBoost.

* When total elapsed time is taken into account, this is even more evident. Although having no absolute meaning, the *ratio between R2 and running time* shows the superiority of first screening features based on the correlation among them, and finally selecting features through supervised learning methods. If no selection has very poor relative performance (e.g., excessive running for lasso estimation), sequential features selection requires a prohibitive computing time for a performance not better than that for more simple alternatives.

* When explicitly relating performance metric with *running time*, the extent to which sequential selection is not appropriate for this learning task is strengthen. If sequential selection is disregarded, a light positive association is found between performance and running time, although some cost-effective alternatives are available, such as the above mentioned supervised learning selection with a relatively strong regularization whose small subset of selected features allows a good performance with just a few computational complexity.

* A similar conclusion can be drawn from the relationship between *performance and the number of selected features* performance and the number of selected features.

It is important to notice that results derived, presented and discussed here do not hold for any supervised learning task. However, some notes may help choosing a features selection for a given setting. Supervised learning selection seems adequate for a first approach when trying to reduce complexity of models. Methods such as RFE and RFECV are more robust techniques with good balance between performance and running time. Both forward and backward-stepwise selection may only be considered when performance is expected to be highly optimized, since they have extremely high computational costs. Finally, unsupervised features screening, either by variance or correlation thresholding, should always be considered, since may help dropping irrelevant features at a very low computational cost.

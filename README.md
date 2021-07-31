## Features selection

This project has the objective of exploring and testing alternative methods of features selection. It has started with the reading of articles, books on machine learning fundamentals, and the documentation of libraries, which has led to the notes found in the Wiki of this repository, where the relevance, approaches and methods of features selection are presented. The two main objectives of selecting features are reducing model complexity (thus saving memory and time) and eventually improving model performance.
<br>
<br>
The Wiki organizes popular methods based on three different classes of methods: *analytical methods*, which focus on the relationship between two variables (different inputs or an input and the output) or even consider only one variable at a time; *supervised learning selection*, which makes use of statistical learning methods that rank input variables according to their importance while training a model; and *exaustive methods*, which explore several distinct subsets of the entire set of available features.
<br>
<br>
In order to explore and test alternative methods of features selection, the development of this project has led to four major contents: first, the already mentioned notes; second, a Python class providing a unified API for implementing multiple methods from those three classes mentioned above (module "features_selection.py"); third, a notebook which illustrates how to use the most relevant methods of features selection, by using either the native classes and functions or the developed class with a unifed API (folder "Tutorials"); and finally, a notebook ("Features Selection - Empirical Tests") implementing tests for assessing the most adequate method for a given regression problem (folder "Empirical Tests").

--------------------------------------------------------------------------------------------------------------------------------------------------------------
FeaturesSelection is a Python class (module "features_selection.py") that can implement three different classes of features selection methods: analytical methods, supervised learning selection and exaustive methods.

*Analytical methods* refer to approaches where only two variables the most are taken at the same time: here, variance selection (variance thresholding to exclude variables with too few variability) and correlation selection (correlation thresholding to exclude variables with excessive correlation with variables with more variability) are explored by the class as initialization parameter "method" is declared equal to "variance" or "correlation", respectively. Additional arguments needed during initialization:
* threshold: reference value for variance or correlation thresholding.

*Supervised learning selection* means that only those features whose importance is greater than some threshold are selected, where this importance is calculated as a model is trained under a given supervised learning method. Can be used as initialization parameter "method" is defined equal to "supervised". Additional arguments needed:
* estimator: (declared into "features_select" method) machine learning algorithm containing either a "coef_" or a "feature_importances_" attribute.
* threshold: (declared during initialization) importance value above which features are selected.

*Exaustive methods* evaluate several distinct subsets of features with different lengths. Methods covered by the FeaturesSelection class are: RFE (method="rfe"), RFECV (method="rfecv"), SequentialFeatureSelector (method="sequential") and random selection (method="random_selection"). Except for the last method, all the others are extracted from sklearn. Besides of a unified approach, the FeaturesSelection class introduces the choice of the best number of features for RFE and SequentialFeatureSelector, since sklearn implementation does not cover this.
* RFE: given an initialized estimator, at each step a predefined number of the least relevant features are dropped until a given number is reached. Arguments (all declared during initialization, except for "estimator"):
    * estimator: machine learning algorithm.
    * num_folds: number of folds of K-folds CV for selecting final model.
    * metric: performance metric for selecting final model.
    * max_num_feats: maximum number of features to be tested.
    * step: number of features to be dropped at each iteration.


* RFECV: selected features are defined according to the optimization of some performance metric that is calculated using K-folds cross-validation. Consequently, at each step the least important features are dropped, and from the final collection of models where each has a different number of features the best model is chosen through cross-validation. Arguments (all declared during initialization, except for "estimator"):
    * estimator: machine learning algorithm.
    * num_folds: number of folds of K-folds CV for selecting final model.
    * metric: performance metric for selecting final model.
    * min_num_feats: minimum number of features to be selected.
    * step: number of features to be dropped at each iteration.


* SequentialFeatureSelector: depending on the "direction" initialization parameter, a version of forward-stepwise selection or a version of backward-stepwise selection can be implemented. As with RFE, the number of features to be selected is another parameter that should ultimately be defined in order to optimize model performance. Arguments (all declared during initialization, except for "estimator"):
    * estimator: machine learning algorithm.
    * num_folds: number of folds of K-folds CV for selecting final model.
    * metric: performance metric for selecting final model.
    * max_num_feats: maximum number of features to be tested.
    * direction: indicates whether forward or backward-stepwise selection should be implemented.


* Random selection of features: defines a collection of models with different numbers of features (all randomly picked), and then chooses the best model using K-folds CV. Arguments (all declared during initialization, except for "estimator"):
    * estimator: machine learning algorithm.
    * num_folds: number of folds of K-folds CV for selecting final model.
    * metric: performance metric for selecting final model.
    * max_num_feats: maximum number of features to be tested.
    * step: number of features to be randomly included at each iteration.

The usage of the FeaturesSelection class requires the initialization of an object, when all relevant arguments should be declared depending on the method chosen during the definition of "method" argument. Then, inputs and output are passed (together with "estimator" parameter whenever needed) in the "select_features" method, which then produces a list with the names of selected features.

Note in the FeaturesSelection class that "select_features" method is constructed upon three static methods (*analytical_methods*, *supervised_selection* and *exaustive_methods*) that may be used as a function without the need of creating an object from FeaturesSelection class. Then, inputs and output (together with "estimator" parameter whenever needed) should also be declared when executing the static method.

The main advantages of making use of the FeaturesSelection class are:
* Unified and straightforward API: at least 7 distinct methods of features selection are available within the same class.
* Flexibility of use: one can either initialize an object from FeaturesSelection class or call static methods as functions.
* Ready-to-use outcomes: irrespective of the method chosen, the output of the features selection will be a list with names of selected features. Thus, it is not necessary to define the best number of features for a model.

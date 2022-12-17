# Cora Classification
Classification of scientific publications using various classifiers.

## Project Description

The purpose of this project is to make a comparison between the classification performance of different machine learning algorithms in the categorization of the scientific articles using the words availabe in them articles as features.

### Dataset
The Cora dataset consists of 2708 scientific publications classified into one of seven classes (`Case_Based`, `Genetic_Algorithms`, `Neural_Networks`, `Probabilistic_Methods`, `Reinforcement_Learning`, `Rule_Learning`, `Theory`). The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download Link: https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

Related Papers:
- [Qing Lu, and Lise Getoor. "Link-based classification." ICML, 2003.](https://linqspub.soe.ucsc.edu/basilic/web/Publications/2003/lu:icml03/)
- [Prithviraj Sen, et al. "Collective classification in network data." AI Magazine, 2008.](https://linqspub.soe.ucsc.edu/basilic/web/Publications/2008/sen:aimag08/)


## Applied Classifiers

We can divide the classifiers used in this project into two categories of classical and deep learning models.

### Classical Machine Learning Models
We have applied the [scikit-learn library](https://scikit-learn.org/stable/) for the classical machine learning models.
Mainly, we have used two categories of classical machine learning models

**Parametric**
*	[Linear Support Vector Machine (SVM)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html);
* [Linear Discriminat Analysis (LDA)](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html);
* [Quadratic Discriminat Analysis (QDA)](https://scikit-learn.org/0.16/modules/generated/sklearn.qda.QDA.html);
* [Gaussian Naive Bayes (GNB)](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html);
* [Multinomial Naive Bayes (MNB)](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html);
* [Logistic Regression (LR)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

**Nan-parametric**
* [SVM with Radial Basis Function (RBF) kernel](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
* [Random Forest (RF)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html);
* [eXtreme Gradient Boosting (XGBoost)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html);
* [K-Nearest Neighbors (KNN)](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).

#### Execution of the Classical Machine Learning Models

The code to execute the above mensioned methods applied on the Cora data, can be found [here](https://github.com/splendidcomputer/cora_project/blob/main/Cora_Classification_ClassicML.ipynb).

### Deep Learning Model
In this project we have used the [MultiLayer Perceptron (MLP)](MultiLayer Perceptron (MLP) with five fully connected layers which itslef is a parametric model.

Neural networks do the feature extraction internally and achieve higher accuracy.
Deep learning model architecture used in this project has the following architecture:
Input	512	256	dropout	128	dropout	64	10	7

 ![alt text](https://github.com/splendidcomputer/cora_project/blob/main/model_plot.png)
 
 
 #### Execution of the Deep Learning Model
 The code to execute deepl learning model applied on the Cora data, can be found [here](https://github.com/splendidcomputer/cora_project/blob/main/Cora_Classification_NN.ipynb).
 
## Results

For this problem, in general, among classical machine learning models, non-parametric models perfrom slightly better than the parametric.

You can see the whole prediction results in TSV files, containing all classifiction results in this [directory](https://github.com/splendidcomputer/cora_project/tree/main/Prediction_Results).

### Classical Machine Learning approaches

The comparison of the prediction results using classical machine learning modles can be found through the following link:

* [Comparsion of the Classical Machine Learning results](https://github.com/splendidcomputer/cora_project/blob/main/Prediction_Results/ML_Test_ACC.tsv)

As you can see the among classical machine learning models, non-parametric models perfrom slightly better than the parametric.

### Deepl Learning

The classification Result using deep neural networks is also shown in the table below:

* [Classification results using Deep Neural Netwroks](https://github.com/splendidcomputer/cora_project/blob/main/Prediction_Results/NN_Test_ACC.tsv)

## Conclusion

As it can be seen from the classification performance results, the deep learning model acheives more promising results as it comes internally with feature extraction layers which provide the classification layer with a well classwisely seperated space and therefore it acheives a higher accuracy in comparison with the classical models that do not have the feature extraction innately.

# Cora Classification
Classification of scientific publications using various classifiers.

## Project Description

The purpose of this project is to find the best machine learning algorithm to predict the scientific article classifications using the words availabe in the articles as features.

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
Mainly, we have two categories of classical machine learning models

**Parametric**
*	Linear Support Vector Machine (SVM);
* Linear Discriminat Analysis (LDA);
* Quadratic Discriminat Analysis (QDA);
* Gaussian Naive Bayes (GNB);
* Multinomial Naive Bayes (MNB);
* Logistic Regression (LR).

**Nan-parametric**
* SVM with Radial Basis Function (RBF) kernel 
* Random Forest (RF);
* eXtreme Gradient Boosting (XGBoost);
* K-Nearest Neighbors (KNN).

#### Execution of the Classical Machine Learning Models
The code to execute the above mensioned methods applied on the Cora data, can be found [here](https://github.com/splendidcomputer/cora_project/blob/main/Cora_Classification_ClassicML.ipynb).

### Deep Learning Model
In this project we have used the MultiLayer Perceptron (MLP) with five fully connected layers which itslef is a parametric model.

Neural networks do the feature extraction internally and achieve higher accuracy.
Deep learning model architecture used in this project has the following architecture:
Input	512	256	dropout	128	dropout	64	10	7

 ![alt text](https://github.com/splendidcomputer/cora_project/blob/main/model_plot.png)
 
 
 #### Execution of the Deep Learning Model
 The code to execute deepl learning model applied on the Cora data, can be found [here](https://github.com/splendidcomputer/cora_project/blob/main/Cora_Classification_NN.ipynb).
 
## Results

For this problem, in general, among classical machine learning models, non-parametric models perfrom slightly better than the parametric.

The comparison of the prediction results using classical machine learning modles can be found through the following link:

* [Comparsion of the Classical Machine Learning results](https://github.com/splendidcomputer/cora_project/blob/main/Prediction_Results/ML_Test_ACC.tsv)

As you can see the among classical machine learning models, non-parametric models perfrom slightly better than the parametric.

The classification Result using deep neural networks is also shown in the table below:

* [Classification results using Deep Neural Netwroks](https://github.com/splendidcomputer/cora_project/blob/main/Prediction_Results/NN_Test_ACC.tsv)

As you can see the deep learning model acheives more promising results as it comes internally with feature extraction layers which provide the classification layer with a well classwisely seperated space and therefore it acheives a higher accuracy in comparison with the classical models that do not have the feature extraction innately.


### Parametric vs non-parametric
QDA: Features are colinear

##

### Feature Selection methods

Feature Extraction methods can be divied into two main categories:

* Wrapper;
* Filter.

#### Wrapper
In warper models, we use the ML model and add/remove features until we achieve the desired accuracy.

•	Sequential Forward Feature Selection (SFFS)  Has a high computational time
•	Sequential Backward Feature Selection (SBFS)

We could use these approches

#### Filter
In filter methods we neglect the ML model and we only check if the features have linear dependence/relation
•	Pearson correlation
Filter methods are more appropriate 
Wrapper methods have higher computational cost
Wrapper methods have higher chacnes of overfitting



Parametric Machine Learning Algorithms
Assumptions can greatly simplify the learning process but can also limit what can be learned. Algorithms that simplify the function to a known form are called parametric machine learning algorithms.
The algorithms involve two steps:
1.	Select a form for the function.
2.	Learn the coefficients for the function from the training data.

Assuming the functional form of a line greatly simplifies the learning process. Now, all we need to do is estimate the coefficients of the line equation and we have a predictive model for the problem.
Often the assumed functional form is a linear combination of the input variables and as such parametric machine learning algorithms are often also called “linear machine learning algorithms “.
The problem is the actual unknown underlying function may not be a linear function like a line. It could be almost a line and require some minor transformation of the input data to work right. Or it could be nothing like a line in which case the assumption is wrong, and the approach will produce poor results.
Some more examples of parametric machine learning algorithms include:
•	Logistic Regression
•	Linear Discriminant Analysis
•	Perceptron
•	Naive Bayes
•	Simple Neural Networks
Benefits of Parametric Machine Learning Algorithms:
•	Simpler: These methods are easier to understand and interpret results.
•	Speed: Parametric models are very fast to learn from data.
•	Less Data: They do not require as much training data and can work well even if the fit to the data is not perfect.
Limitations of Parametric Machine Learning Algorithms:
•	Constrained: By choosing a functional form these methods are highly constrained to the specified form.
•	Limited Complexity: The methods are more suited to simpler problems.
•	Poor Fit: In practice the methods are unlikely to match the underlying mapping function.

Nonparametric Machine Learning Algorithms
Algorithms that do not make strong assumptions about the form of the mapping function are called nonparametric machine learning algorithms. By not making assumptions, they are free to learn any functional form from the training data.
Nonparametric methods seek to best fit the training data in constructing the mapping function, whilst maintaining some ability to generalize to unseen data. As such, they are able to fit a large number of functional forms.

An easy-to-understand nonparametric model is the k-nearest neighbors algorithm that makes predictions based on the k most similar training patterns for a new data instance. The method does not assume anything about the form of the mapping function other than patterns that are close and are likely to have a similar output variable.

Some more examples of popular nonparametric machine learning algorithms are:

•	k-Nearest Neighbors
•	Decision Trees like CART and C4.5
•	Support Vector Machines
Benefits of Nonparametric Machine Learning Algorithms:

•	Flexibility: Capable of fitting a large number of functional forms.
•	Power: No assumptions (or weak assumptions) about the underlying function.
•	Performance: This can result in higher performance models for prediction.
Limitations of Nonparametric Machine Learning Algorithms:

•	More data: Require a lot more training data to estimate the mapping function.
•	Slower: A lot slower to train as they often have far more parameters to train.
•	Overfitting: More of a risk to overfit the training data and it is harder to explain why specific predictions are made.

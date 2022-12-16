# cora_project
Classification of scientific publications using various classifiers.


Classical Machine Learning Models
We have applied the sklearn library for the classical Machine Learning models.
We have two categories of classical machine learning models
•	Parametric
o	SVM, LDA, QDA	
•	Nan-parametric 
RF, GNB, MNB, Bayes

Parametric vs non-parametric
QDA: Features are colinear

Feature Extraction methods

Supervised: LDA
Unsupervised: PCA
Wrapper
In warper models, we use the ML model and add/remove features until we achieve the desired accuracy.

•	Sequential forward feature selection (SFFS)  Has a high computational time
•	Sequential backward feature selection (SBFS)

Filter
In filter methods we neglect the ML model and we only check if the features have linear dependence/relation
•	Pearson correlation
Filter methods are more appropriate 
Wrapper methods have higher computational cost
Wrapper methods have higher chacnes of overfitting

Neural networks
Do the feature extraction internally and achieve higher accuracy.
Deep learning model:
Input	512	256	dropout	128	dropout	64	10	7

 ![alt text](http://url/to/img.png)

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


Random Forest

Random Forest is an easy-to-use machine learning algorithm that often provides excellent results even without adjusting its meta-parameters. This algorithm is considered one of the most used machine learning algorithms due to its simplicity and usability, both for "Classification" and "Regression". Here, how the random forest works and other important issues around it will be examined.

To understand how the random forest works, you must first learn the "Decision Tree" algorithm, which is the building block of the random forest. Humans use decision trees for their decisions and choices every day, even if they do not know what they are using is a machine learning algorithm. To clarify the concept of the decision tree algorithm, an everyday example is used, i.e., forecasting the maximum air temperature of the city for the next day (tomorrow).

Here it is assumed that the desired city of Seattle is located in the state of Washington (this example can be extended to various other cities as well). To answer the simple question "What is the temperature tomorrow?", it is necessary to work on a series of queries. This is done by creating an initially suggested temperature range selected based on domain knowledge.

In this matter, if it is not clear at the beginning of the work what time of the year "tomorrow" (the temperature of which is supposed to be guessed) corresponds to, the initially suggested range can be between 30 and 70 degrees (Fahrenheit). Gradually, through a series of questions and answers, this interval is reduced to ensure that a sufficiently confident prediction can be made.

How do you prepare good questions that can be answered correctly to narrow down the available space? If the goal is to limit the range as much as possible, the smart solution is to address queries that are relevant to the problem under investigation. Since temperature is highly dependent on the time of year, a good place to start would be to ask, "What season is tomorrow?"

LDA
Linear Discriminant Analysis as its name suggests is a linear model for classification and dimensionality reduction.  Most commonly used for feature extraction in pattern classification problems. This has been here for quite a long time. First, in 1936 Fisher formulated linear discriminant for two classes, and later on, in 1948 C.R Rao generalized it for multiple classes. LDA projects data from a D dimensional feature space down to a D’ (D>D’) dimensional space in a way to maximize the variability between the classes and reducing the variability within the classes.
Why LDA?:
•	Logistic Regression is one of the most popular linear classification models that perform well for binary classification but falls short in the case of multiple classification problems with well-separated classes. While LDA handles these quite efficiently.
•	LDA can also be used in data preprocessing to reduce the number of features just as PCA which reduces the computing cost significantly.
•	LDA is also used in face detection algorithms. In Fisherfaces LDA is used to extract useful data from different faces. Coupled with eigenfaces it produces effective results.
Shortcomings:
•	Linear decision boundaries may not effectively separate non-linearly separable classes. More flexible boundaries are desired.
•	In cases where the number of observations exceeds the number of features, LDA might not perform as desired. This is called Small Sample Size (SSS) problem. Regularization is required.
We will discuss this later.
Assumptions:
LDA makes some assumptions about the data:
•	Assumes the data to be distributed normally or Gaussian distribution of data points i.e. each feature must make a bell-shaped curve when plotted. 
•	Each of the classes has identical covariance matrices.
However, it is worth mentioning that LDA performs quite well even if the assumptions are violated.

QDA
This operator performs a quadratic discriminant analysis (QDA). QDA is closely related to linear discriminant analysis (LDA), where it is assumed that the measurements are normally distributed. Unlike LDA however, in QDA there is no assumption that the covariance of each of the classes is identical. To estimate the parameters required in quadratic discrimination more computation and data is required than in the case of linear discrimination. If there is not a great difference in the group covariance matrices, then the latter will perform as well as quadratic discrimination. Quadratic Discrimination is the general form of Bayesian discrimination.

Discriminant analysis is used to determine which variables discriminate between two or more naturally occurring groups. For example, an educational researcher may want to investigate which variables discriminate between high school graduates who decide (1) to go to college, (2) NOT to go to college. For that purpose the researcher could collect data on numerous variables prior to students' graduation. After graduation, most students will naturally fall into one of the two categories. Discriminant Analysis could then be used to determine which variable(s) are the best predictors of students' subsequent educational choice. Computationally, discriminant function analysis is very similar to analysis of variance (ANOVA). For example, suppose the same student graduation scenario. We could have measured students' stated intention to continue on to college one year prior to graduation. If the means for the two groups (those who actually went to college and those who did not) are different, then we can say that the intention to attend college as stated one year prior to graduation allows us to discriminate between those who are and are not college bound (and this information may be used by career counselors to provide the appropriate guidance to the respective students). The basic idea underlying discriminant analysis is to determine whether groups differ with regard to the mean of a variable, and then to use that variable to predict group membership (e.g. of new cases).

Discriminant Analysis may be used for two objectives: either we want to assess the adequacy of classification, given the group memberships of the objects under study; or we wish to assign objects to one of a number of (known) groups of objects. Discriminant Analysis may thus have a descriptive or a predictive objective. In both cases, some group assignments must be known before carrying out the Discriminant Analysis. Such group assignments, or labeling, may be arrived at in any way. Hence Discriminant Analysis can be employed as a useful complement to Cluster Analysis (in order to judge the results of the latter) or Principal Components Analysis.
XGB
XGBoost is a popular and efficient open-source implementation of the gradient boosted trees algorithm. Gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target variable by combining the estimates of a set of simpler, weaker models.

When using gradient boosting for regression, the weak learners are regression trees, and each regression tree maps an input data point to one of its leafs that contains a continuous score. XGBoost minimizes a regularized (L1 and L2) objective function that combines a convex loss function (based on the difference between the predicted and target outputs) and a penalty term for model complexity (in other words, the regression tree functions). The training proceeds iteratively, adding new trees that predict the residuals or errors of prior trees that are then combined with previous trees to make the final prediction. It's called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.
SVM (Linear / RBF)
KNN
Linear Regression (LR)
GNB
MNG


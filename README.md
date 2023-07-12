# data-mining
This project was created as part of "Data Mining" subject in Computer Engineering &amp; Informatics Department (CEID) of University of Patras. It involves the use of 2 ML models, namely Support Machine Vectors (SVM) and Artificial Neural Networks (ANN). The project was implemented using `Python`.

## Support Machine Vectors
This task is implemented in python file `task1.py` using the winequality-red.csv Dataset. For this file we make use of the `scikit-learn` library for the necessary models.

The first part involved the simple use of a SVC (Support Vector Classifier) for the classification of the wine quality. 
In the second part, we were asked to remove 33% of the `ph` feature data of the training set. Then we had to refill the missing values with four distinct methods.
1. Remove the entire column.
2. Fill the missing rows with the mean of the entire column.
3. Fill the missing values using Logistic Regression.
4. Fill the missing values, using K-means to create groups and then calculating each group's mean of the `ph` column. Then we fill each group's NaN value with the group's mean.

## Artificial Neural Networks
This task is implemented in python file `task2.py` using the onion-or-not.csv Dataset. For this file we make use of the `NLTK` library for the natural language processing and MLPClassifier from `scikit-learn` for the ANN. With the help of NLTK we transform the input strings to vector space where we can feed them to the ANN and make some predictions on whether a given fake title was uploaded in __[theonion.com](https://www.theonion.com/)__.

Both tasks are measured using metrics such as accuracy, precision, recall and f1-score.

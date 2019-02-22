# Kernel SVM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
# Importing the dataset
col_names= ['Diagnosis ','RT3U','T4','T3','TSH','DTSH']
dataset = pd.read_table('thy.txt',sep=',', names= col_names)
X = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier1 = SVC(kernel = 'rbf', random_state = 0)
classifier1.fit(X_train, y_train)

# Fitting Kernel SVM to the Training set
from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(X_train, y_train)

# Fitting Kernel SVM to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier3 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier3.fit(X_train, y_train)


# Predicting the Test set results(SVC)
y_pred1 = classifier1.predict(X_test)

# Predicting the Test set results(LOGISTIC)
y_pred2 = classifier2.predict(X_test)

# Predicting the Test set results(DECISION TREE)
y_pred3 = classifier3.predict(X_test)

# Making the Confusion Matrix for test data
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score

#confusion matrix for SVC
cm1 = confusion_matrix(y_test, y_pred1)

#confusion matrix for LOGISTIC
cm2 = confusion_matrix(y_test, y_pred2)

#confusion matrix for DECISION TREE
cm3 = confusion_matrix(y_test, y_pred3)

# heatmap for SVC
sb.heatmap(cm1, annot= True)

# heatmap for LOGISTIC
sb.heatmap(cm2, annot= True)

# heatmap for DECISION TREE
sb.heatmap(cm3, annot= True)


#svc
accuracy_score(y_test, y_pred1)

#logistic
accuracy_score(y_test, y_pred2)

#decision tree
accuracy_score(y_test, y_pred3)



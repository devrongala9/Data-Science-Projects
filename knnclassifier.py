#!/usr/bin/env python
# coding: utf-8

# In[516]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load the Dataset
data = pd.read_csv(r"C:\Users\komma\Downloads\P2\P2\nba2021.csv")


# In[517]:


data.shape


# In[518]:


# The dataset contains attributes such as player name and team name, age
# We know that they are not useful for classification and thus do not
# include them as features.
columns_to_remove = ["Player", "Age", "Tm"]
data = data.drop(columns=columns_to_remove)


# In[519]:


data.head()


# In[520]:


data = data[data.MP >= 5]
data = data[data.PTS >= 5]
data.shape


# In[521]:


summary_df = data.groupby('Pos').mean()
summary_df


# In[522]:


# Encode the 'Pos' column to convert position labels to numeric labels
label_encoder = LabelEncoder()
data['Pos'] = label_encoder.fit_transform(data['Pos'])


# In[523]:


# Create an empty list to store DataFrames
p_values_dataframes = []
target_variable = 'Pos'

# Iterate through all columns except the target variable
for column in data.columns:
    if column != target_variable:
        # Create a contingency table
        crosstab = pd.crosstab(data[target_variable], data[column])
        
        # Perform the chi-squared test
        chi2, p, _, _ = chi2_contingency(crosstab)
        
        # Create a DataFrame for the results and append it to the list
        result_df = pd.DataFrame({'Feature': [column], 'Chi-squared': [chi2], 'P-value': [p]})
        p_values_dataframes.append(result_df)

# Concatenate the list of DataFrames into a single DataFrame
p_values_df = pd.concat(p_values_dataframes, ignore_index=True)

# Sort the DataFrame by p-value in ascending order
p_values_df.sort_values(by='P-value', ascending=True, inplace=True)

# Show the top 15 features with the lowest p-values
top_features = p_values_df.nsmallest(15, 'P-value')
#print(top_features)


# In[524]:


#using above analysis and domain knowledge removing following columns

columns_to_remove = ['G', 'GS', 'FG%', '3P%', '2P%','FT%', 'MP','FT', 'FG','eFG%']
data = data.drop(columns=columns_to_remove)


# In[525]:


data.loc[:, 'Pos'].value_counts()
#checking if target variable is balanced or not


# In[526]:


# Define features and target variable
X = data[['FGA', '3PA', '2PA', 'FTA', 'ORB', 'DRB', 'TRB', 'AST','STL','BLK','PF']]
y = data['Pos']

# Encode the 'Pos' column to convert position labels to numeric labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Initialize KNN classifier with different values of n_neighbors
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # Build the KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,p=2)#p=2 indicates euclidean distance
    knn.fit(X_train, y_train)
    
    # Record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    
    # Record generalization accuracy
    test_accuracy.append(knn.score(X_test, y_test))

# Plot the training and test accuracy for different values of n_neighbors
plt.plot(neighbors_settings, training_accuracy, label="Training Accuracy")
plt.plot(neighbors_settings, test_accuracy, label="Test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
#plt.show()

# Print the test set accuracy for the best-performing KNN model
best_n_neighbors = neighbors_settings[np.argmax(test_accuracy)]
best_knn = KNeighborsClassifier(n_neighbors=best_n_neighbors,p=2)
best_knn.fit(X_train, y_train)
test_accuracy_best = best_knn.score(X_test, y_test)
print("Best Test Set Accuracy: {:.2f}".format(test_accuracy_best))
print(best_knn)



# In[527]:


# Train the KNN model with the best n_neighbors
best_knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
best_knn.fit(X_train, y_train)

# Predict on the training and test data
y_train_pred = best_knn.predict(X_train)
y_test_pred = best_knn.predict(X_test)

# Calculate the training and test set scores
train_accuracy = best_knn.score(X_train, y_train)
test_accuracy = best_knn.score(X_test, y_test)

# Calculate the confusion matrix for training data
conf_matrix_train = pd.crosstab(y_train, y_train_pred, rownames=['True'], colnames=['Predicted'], margins=True)

# Calculate the confusion matrix for test data
conf_matrix_test = pd.crosstab(y_test, y_test_pred, rownames=['True'], colnames=['Predicted'], margins=True)

# Print training and test set scores
print("Training set score for KNN: {:.3f}".format(train_accuracy))
print("Test set score for KNN: {:.3f}".format(test_accuracy))

# Print confusion matrices
print("Confusion matrix for training data KNN model:")
print(conf_matrix_train)

print("Confusion matrix for test data KNN model:")
print(conf_matrix_test)


# In[510]:


from sklearn.model_selection import cross_val_score, StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Calculate cross-validation scores
cv_scores = cross_val_score(best_knn, X, y, cv=kfold)

# Print cross-validation scores
print("Cross-validation scores KNN: {}".format(cv_scores))
print("Average cross-validation score KNN: {:.2f}".format(np.mean(cv_scores)))


# In[511]:


# Define features and target variable
X = data[['BLK','ORB','3PA','TRB','DRB','PF','STL','AST','PTS','2PA','FTA']]
y = data['Pos']
# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Create and train the LinearSVC model
linearsvc = LinearSVC(dual=True,max_iter=10000)
linearsvc.fit(X_train, y_train)

# Evaluate the model using test data
y_pred = linearsvc.predict(X_test)


# In[512]:


# Calculate accuracy on the training set
train_accuracy = linearsvc.score(X_train, y_train)
print("Training Set Accuracy for SVC first model: {:.2f}".format(train_accuracy))

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy for SVC first model: {:.2f}".format(accuracy))

# Step 5: Confusion Matrix
# Calculate the confusion matrix for training data
conf_matrix = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
print("Confusion Matrix SVC1:")
print(conf_matrix)


# In[513]:


from sklearn.model_selection import cross_val_score, StratifiedKFold
# Define 10-fold stratified cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
# Perform cross-validation and print accuracy for each fold
accuracies = cross_val_score(linearsvc, X, y, cv=cv, scoring='accuracy')
for fold, accuracy in enumerate(accuracies, start=1):
    print(f"Fold {fold} Accuracy: {accuracy:.2f}")
# Calculate and print the average accuracy
average_accuracy = np.mean(accuracies)
print(f"Average Accuracy Across All Folds for SVC first model: {average_accuracy:.2f}")


# In[514]:


data = data[data.PTS >= 8]
# Define features and target variable
X = data[['BLK','ORB','3PA','TRB','DRB','PF','STL','AST','PTS','2PA','FTA']]
y = data['Pos']
# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
svm = SVC(kernel='linear',random_state=0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
train_accuracy = svm.score(X_train, y_train)
print("Training Set Accuracy for SVC Second model: {:.2f}".format(train_accuracy))
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy for SVC second model: {:.2f}".format(accuracy))
conf_matrix = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
print("Confusion Matrix SVC2:")
print(conf_matrix)


# In[515]:


from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
accuracies = cross_val_score(svm, X, y, cv=cv, scoring='accuracy')
for fold, accuracy in enumerate(accuracies, start=1):
    print(f"Fold {fold} Accuracy: {accuracy:.2f}")
average_accuracy = np.mean(accuracies)
print(f"Average Accuracy Across All Folds for svc second model: {average_accuracy:.2f}")


# In[ ]:





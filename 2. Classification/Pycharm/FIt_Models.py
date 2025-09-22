from Preprocessing import *


# # Preprocess Train Data
# In[30]:


# Apply Sentiment Analysis
X_train = apply_sentiment(X_train)

# Clean data and Handle missing values
X_train,mean_list = data_cleaning_X_Train(X_train)
save_model(mean_list, 'Mean List.pkl')

# Encoding Some Features With Label Encoding
X_train,le = data_encoding_X_Train(X_train)
save_model(le, 'Label Encoder.pkl')

#Drop 'Positive_Review' & 'Negative_Review' In X_Train
X_train = X_train.drop(['Positive_Review'], axis=1)
X_train = X_train.drop(['Negative_Review'], axis=1)

# Handle Missing Value of The Reviewer_Score (y_train)  Classifier with Random Forset
X_train,y_train,rf = data_cleaning_Y_Train(X_train,y_train)
save_model(rf, 'Filling Missing Values_RF.pkl')

# Encoding The Reviewer_Score (y_train)  With Ordinal Encoder
y_train,ordinal_encoder = data_encoding_Y_Train(y_train)
save_model(ordinal_encoder, 'Ordinal Encoder.pkl')

# Handle Outliers
X_train = handle_outliers(X_train, 'iqr', 1)

#Scaling X_Train
X_train,scaler=scaling(X_train,IsTrain=True, IsStandard=True) # First: True is X_train ,, Second: True -> is (Standard) or not -> (Norm)
save_model(scaler, 'Standard Scaler.pkl')

# select the most effiective Features
X_train, y_train = feature_selection(X_train, y_train)
y_train=y_train.values
# reshape a 1D numpy array into a 2D numpy array.
y_train = y_train[:, np.newaxis]
save_model(X_train.columns, 'Train_Cols_Feature_Selection.pkl')


# # Models

# ## SVM Model

# In[31]:


'''
from sklearn import svm

# Create an SVM classifier
c = 0.001; #Svm Regularzition Parameter 
svm = svm.SVC(kernel='poly',degree=5)

# Fit the classifier to the training data
start = time.perf_counter()
svm.fit(X_train, y_train.ravel())
end = time.perf_counter()
print("SVM Model Time  : " + str(end - start) + " sec")

# Save model to file
save_model(svm, 'SVM_Model.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(svm,X_train,y_train,data_name=train)

# Compute the confusion matrix (Test) 
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "SVM_Model")
'''



# ## Logistic Regression Model

# In[32]:


# Create a Logistic Regression classifier
lr = LogisticRegression(max_iter=1000)

# Fit the classifier to the training data
start = time.perf_counter()
lr.fit(X_train, y_train.ravel())
end = time.perf_counter()
print("Logistic Regress Model Time  : " + str(end - start) + " sec")

# Save model to file
save_model(lr, 'Logistic_Regress.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(lr,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "Logistic Regress Model")


# ## Decision Tree Model

# In[33]:


from sklearn.tree import DecisionTreeClassifier

# Create an instance of the DecisionTreeClassifier class
dt = DecisionTreeClassifier(max_depth=10, min_samples_split=500)

# Fit the classifier to the training data
start = time.perf_counter()
dt.fit(X_train, y_train)
end = time.perf_counter()
print("Decision Trees Model Time  : " + str(end - start) + " sec")

# Save model to file
save_model(dt, 'Decision_Tree.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(dt,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "Decision_Tree Model")


# ## RandomForestClassifier

# In[34]:


# Define the hyperparameters to try
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Use grid search to find the best hyperparameters
#grid_search = GridSearchCV(rf, param_grid, cv=5)
#grid_search.fit(X_train, y_train.ravel())

# Print the best hyperparameters and the corresponding score
#print('Best hyperparameters:', grid_search.best_params_)
#print('Best score:', grid_search.best_score_)

# Use the best model to make predictions on the testing data
#best_model = grid_search.best_estimator_
#predictions_test = best_model.predict(X_test)

# Create the random forest classifier with some hyperparameters
#rf = RandomForestClassifier(max_depth= 10, min_samples_leaf= 4, min_samples_split= 10, n_estimators= 100)
rf = RandomForestClassifier(n_estimators=100, max_depth=10)

# Fit the random forest model to the training data
start = time.perf_counter()
rf.fit(X_train, y_train.ravel())
end = time.perf_counter()
print("RandomForest Classifier Model Time  : " + str(end - start) + " sec")

# Save model to file
save_model(rf, 'RandomForestClassifier.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(rf,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "RandomForestClassifier Model")

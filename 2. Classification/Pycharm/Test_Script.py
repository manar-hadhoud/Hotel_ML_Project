from Preprocessing import *
# # Test Script

# ## Preparing The Test

# In[35]:
# Read The Test Data
#test_data = pd.read_csv('hotel-classification-dataset.csv')

# set test_X & test_Y
#y_test = data.iloc[:, 16]
#X_test = data.iloc[:, 0:16]


# Apply Sentiment Analysis
X_test = apply_sentiment(X_test)

# Clean data and Handle missing values
load_mean_list= load_model('Mean List.pkl')
X_test = data_cleaning_X_Test(X_test,load_mean_list)


# Encoding Some Features With Label Encoding
load_le= load_model('Label Encoder.pkl')
X_test = data_encoding_X_Test(X_test,load_le)


#Drop 'Positive_Review' & 'Negative_Review' In X_Test
X_test = X_test.drop(['Positive_Review'], axis=1)
X_test = X_test.drop(['Negative_Review'], axis=1)

# Handle Missing Value of The Reviewer_Score  (y_test) Classifier with
load_missing_rf= load_model('Filling Missing Values_RF.pkl')
X_test,y_test = data_cleaning_Y_Test(X_test,y_test,load_missing_rf)

# Encoding The Reviewer_Score  (y_test) With Ordinal Encoder
load_ordinal_encoder= load_model('Ordinal Encoder.pkl')
y_test = data_encoding_Y_Test(y_test,load_ordinal_encoder)

# Handle Outliers
X_test = handle_outliers(X_test, 'iqr', 1)

#Scaling X_Test
load_scaler= load_model('Standard Scaler.pkl')
X_test=scaling(X_test,IsTrain= False,scaler= load_scaler) # False is X_test..

# get column names of X_train
#train_cols = X_train.columns
train_cols = load_model('Train_Cols_Feature_Selection.pkl')

# select only the columns present in X_train in X_test
X_test = X_test.loc[:, train_cols]


# ## Calculate Accuracy On (Test)

# In[36]:


# Load the models
loaded_svm = load_model('SVM_Model.pkl')
load_lr = load_model('Logistic_Regress.pkl')
loaded_dt= load_model('Decision_Tree.pkl')
loaded_rf = load_model('RandomForestClassifier.pkl')

# Calculate accuracy on test data && Confusion_Matrix

#predictions_test=Print_Accuracy(loaded_svm,X_test,y_test,data_name="test")
#Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "SVM_Model")

predictions_test=Print_Accuracy(load_lr,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "Logistic Regress Model")

predictions_test=Print_Accuracy(loaded_dt,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "Decision_Tree Model")

predictions_test=Print_Accuracy(loaded_rf,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "RandomForestClassifier Model")

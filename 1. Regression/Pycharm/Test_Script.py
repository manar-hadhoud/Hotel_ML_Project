# # Test Script
from Preprocessing import *

# ## Preparing The Test

# In[ ]:


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
load_missing_mean= load_model('Filling Missing Values_Mean.pkl')
y_test = data_cleaning_Y_Test(y_test,load_missing_mean)

# Encoding The Reviewer_Score  (y_test) With Ordinal Encoder
#load_ordinal_encoder= load_model('Ordinal Encoder.pkl')
#y_test = data_encoding_Y_Test(y_test,load_ordinal_encoder)

# Handle Outliers
X_test = handle_outliers(X_test, 'iqr', 1)
#y_test = handle_outliers(y_test, 'iqr', 1)

#Scaling X_Test
load_scaler_x= load_model('Standard Scaler_x.pkl')
X_test=scaling(X_test,IsTrain= False,scaler= load_scaler_x,cols= X_test.columns) # False is X_test..
#Scaling y_Test
load_scaler_y= load_model('Standard Scaler_y.pkl')
y_test=scaling(y_test.values.reshape(-1, 1),IsTrain= False,scaler= load_scaler_y) # False is y_test..
y_test['Reviewer_Score']= y_test.iloc[:,0]
y_test = y_test.iloc[:, 1]


# get column names of X_train
#train_cols = X_train.columns
train_cols = load_model('Train_Cols_Feature_Selection.pkl')

# select only the columns present in X_train in X_test
X_test = X_test.loc[:, train_cols]


# ## Calculate Accuracy On (Test)

# In[ ]:


# Load the models
loaded_lr = load_model('Linear_Regression.pkl')
load_lasso = load_model('Lasso.pkl')

loaded_poly= load_model('Polynomial.pkl')
load_poly_features= load_model('poly_features.pkl')
X_test_poly= load_poly_features.transform(X_test)

loaded_ridge = load_model('ridge.pkl')
loaded_dt = load_model('DecisionTreeRegressor_model.pkl')
load_rf = load_model('RandomForestRegressor.pkl')
loaded_xgb= load_model('XGBRegressor.pkl')
loaded_kn = load_model('KNeighborsRegressor.pkl')
loaded_lgb = load_model('LGBMRegressor.pkl')
load_er = load_model('ElasticNet_regressor.pkl')
loaded_adb= load_model('adaboost_regressor.pkl')
loaded_qr = load_model('quantile_reg.pkl')
loaded_br = load_model('Bayesian_regressor.pkl')


# Calculate accuracy on test data && Confusion_Matrix

#predictions_test=Print_Accuracy(loaded_svm,X_test,y_test,data_name="test")
#Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "SVM_Model")

predictions_test=Print_Accuracy(loaded_lr,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "Linear_Regression Model")

predictions_test=Print_Accuracy(load_lasso,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "Lasso Model")

predictions_test=Print_Accuracy(loaded_poly,X_test_poly,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "Polynomial Model")

predictions_test=Print_Accuracy(loaded_ridge,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "ridge Model")

predictions_test=Print_Accuracy(loaded_dt,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "DecisionTreeRegressor Model")

predictions_test=Print_Accuracy(load_rf,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "RandomForestRegressor Model")

predictions_test=Print_Accuracy(loaded_xgb,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "XGBRegressor Model")

predictions_test=Print_Accuracy(loaded_kn,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "KNeighborsRegressor Model")

predictions_test=Print_Accuracy(loaded_lgb,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "LGBMRegressor Model")

predictions_test=Print_Accuracy(load_er,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "ElasticNet_regressor Model")

predictions_test=Print_Accuracy(loaded_adb,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "adaboost_regressor Model")

predictions_test=Print_Accuracy(loaded_qr,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "quantile_reg Model")

predictions_test=Print_Accuracy(loaded_br,X_test,y_test,data_name="test")
Display_Confusion_Matrix(y_data= y_test, predictions_data= predictions_test ,model_name= "Bayesian_regressor Model")



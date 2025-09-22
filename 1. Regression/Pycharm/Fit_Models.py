from Preprocessing import *
# # Preprocess Train Data

# In[ ]:


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

# Handle Missing Value of The Reviewer_Score (y_train)  Regressor with Random Forset
y_train,mean = data_cleaning_Y_Train(y_train)
save_model(mean, 'Filling Missing Values_Mean.pkl')

# Encoding The Reviewer_Score (y_train)  With Ordinal Encoder
#y_train,ordinal_encoder = data_encoding_Y_Train(y_train)
#save_model(ordinal_encoder, 'Ordinal Encoder.pkl')

# Handle Outliers
X_train = handle_outliers(X_train, 'iqr', 1)
#y_train = handle_outliers(y_train.values.reshape(-1, 1), 'iqr', 1)

#Scaling X_Train
X_train,scaler_x=scaling(X_train,IsTrain=True, IsStandard=True,cols=X_train.columns) # First: True is X_train ,, Second: True -> is (Standard) or not -> (Norm)
save_model(scaler_x, 'Standard Scaler_x.pkl')
#Scaling y_Train
y_train,scaler_y=scaling(y_train.values.reshape(-1, 1),IsTrain=True, IsStandard=True) # First: True is X_train ,, Second: True -> is (Standard) or not -> (Norm)
y_train['Reviewer_Score']= y_train.iloc[:,0]
y_train = y_train.iloc[:, 1]
save_model(scaler_y, 'Standard Scaler_y.pkl')

# select the most effiective Features
X_train, y_train = feature_selection(X_train, y_train)
y_train=y_train.values
# reshape a 1D numpy array into a 2D numpy array.
y_train = y_train[:, np.newaxis]
save_model(X_train.columns, 'Train_Cols_Feature_Selection.pkl')


# # Models

# ## Linear Model

# In[ ]:


start = t.perf_counter()
linear = linear_model.LinearRegression().fit(X_train,y_train)
end = t.perf_counter()
print("Linear_Regression Model Time  : " + str(end - start) + " sec")


# Save model to file
save_model(linear, 'Linear_Regression.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(linear,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "Linear_Regression Model")


# ## Lasso model

# In[ ]:


start = t.perf_counter()
lasso = Lasso(alpha=0.01).fit(X_train, y_train)
#lasso = load_model('Lasso.pkl')
end = t.perf_counter()
print("Lasso Model Time  : " + str(end - start) + " sec")


# Save model to file
save_model(lasso, 'Lasso.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(lasso,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "Lasso Model")


# ## Polynomial Model

# In[ ]:


poly_features = PolynomialFeatures(degree=8)
X_train_poly = poly_features.fit_transform(X_train)
# Save model to file
save_model(poly_features, 'poly_features.pkl')

start = t.perf_counter()
Polynomial_model = linear_model.LinearRegression().fit(X_train_poly, y_train)
#Polynomial_model = load_model('Polynomial.pkl')
end = t.perf_counter()
print("Polynomial Model Time  : " + str(end - start) + " sec")


# Save model to file
save_model(Polynomial_model, 'Polynomial.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(Polynomial_model,X_train_poly,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "Polynomial Model")


# ## Ridge Model

# In[ ]:


start = t.perf_counter()
ridge = Ridge(alpha=0.01).fit(X_train, y_train)
end = t.perf_counter()
print("Ridge Model Time  : " + str(end - start) + " sec")


# Save model to file
save_model(ridge, 'ridge.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(ridge,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "Ridge Model")


# ## DecisionTreeRegressor Model

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

start = t.perf_counter()
DecisionTreeRegressor_model = DecisionTreeRegressor(max_depth=100).fit(X_train, y_train)
end = t.perf_counter()
print("DecisionTreeRegressor Model Time  : " + str(end - start) + " sec")


# Save model to file
save_model(DecisionTreeRegressor_model, 'DecisionTreeRegressor_model.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(DecisionTreeRegressor_model,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "DecisionTreeRegressor Model")


# ## RandomForestRegressor Model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

star = t.perf_counter()
RandomForestRegressor_model = RandomForestRegressor(n_estimators=50, max_depth=100).fit(X_train, y_train.ravel())
end = t.perf_counter()
print("RandomForestRegressor Model Time  : " + str(end - start) + " sec")


# Save model to file
save_model(RandomForestRegressor_model, 'RandomForestRegressor.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(RandomForestRegressor_model,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "RandomForestRegressor Model")


# ## XGBRegressor Model

# In[ ]:


from xgboost import XGBRegressor
star = t.perf_counter()
xgb = XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,random_state=42).fit(X_train, y_train)
end = t.perf_counter()
print("XGBRegressor Model Time  : " + str(end - start) + " sec")

# Save model to file
save_model(xgb, 'XGBRegressor.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(xgb,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "XGBRegressor Model")


# ## KNeighborsRegressor

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

star = t.perf_counter()
knn_regressor = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
end = t.perf_counter()

print("KNeighborsRegressor Model Time  : " + str(end - start) + " sec")

# Save model to file
save_model(knn_regressor, 'KNeighborsRegressor.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(knn_regressor,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "KNeighborsRegressor Model")


# ## LGBMRegressor Model

# In[ ]:


import lightgbm as lgb
# Create a LightGBM Regression model
params = {
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}
star = t.perf_counter()
LightGBM_regressor = lgb.LGBMRegressor(**params).fit(X_train, y_train.ravel())
end = t.perf_counter()

print("LGBMRegressor Model Time  : " + str(end - start) + " sec")

# Save model to file
save_model(LightGBM_regressor, 'LGBMRegressor.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(LightGBM_regressor,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "LGBMRegressor Model")


# ## ElasticNetElasticNet Model

# In[ ]:


from sklearn.linear_model import ElasticNet
star = t.perf_counter()
ElasticNet_regressor = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)
end = t.perf_counter()

print("ElasticNet_regressor Model Time  : " + str(end - start) + " sec")

# Save model to file
save_model(ElasticNet_regressor, 'ElasticNet_regressor.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(ElasticNet_regressor,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "ElasticNet_regressor Model")


# ## AdaBoostRegressor Model

# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

#Create a Decision Tree Regressor as the base estimator
base_estimator = DecisionTreeRegressor(max_depth=3)

#Create an AdaBoost Regression model with 50 estimators
star = t.perf_counter()
adaboost_regressor = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=50, random_state=42).fit(X_train, y_train.ravel())
end = t.perf_counter()

print("adaboost_regressor Model Time  : " + str(end - start) + " sec")

# Save model to file
save_model(adaboost_regressor, 'adaboost_regressor.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(adaboost_regressor,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "adaboost_regressor Model")


# ## QuantRegQuantReg Model

# In[ ]:


import statsmodels.api as sm
import numpy as np

#Create a Quantile Regression model with alpha=0.5
star = t.perf_counter()
quantile_reg = sm.QuantReg(y_train, X_train).fit(q=0.5)
end = t.perf_counter()

print("quantile_reg Model Time  : " + str(end - start) + " sec")

# Save model to file
save_model(quantile_reg, 'quantile_reg.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(quantile_reg,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "quantile_reg Model")


# ## BayesianRidge Model

# In[ ]:


from sklearn.linear_model import BayesianRidge
star = t.perf_counter()
Bayesian_regressor = BayesianRidge().fit(X_train, y_train.ravel())
end = t.perf_counter()

print("Bayesian_regressor Model Time  : " + str(end - start) + " sec")

# Save model to file
save_model(Bayesian_regressor, 'Bayesian_regressor.pkl')

# Calculate accuracy on train data
predictions_train=Print_Accuracy(Bayesian_regressor,X_train,y_train,data_name="train")

# Compute the confusion matrix (Test)
Display_Confusion_Matrix(y_data= y_train, predictions_data= predictions_train ,model_name= "Bayesian_regressor Model")

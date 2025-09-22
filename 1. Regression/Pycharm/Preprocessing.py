#!/usr/bin/env python
# coding: utf-8

# # Import Libraries & Intial x,y

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import datetime as dt
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import time as t
import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# Read The Data
data = pd.read_csv('hotel-regression-dataset.csv')

# set X & Y
y = data.iloc[:, 16]
x = data.iloc[:, 0:16]


# # Split Train_Test & Sentiment Analysis

# In[2]:


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


# In[3]:


from textblob import TextBlob

# Define a function to analyze the sentiment of a sentence
def get_sentiment(sentence):
    blob = TextBlob(sentence)
    return blob.sentiment.polarity

# Apply the sentiment analysis function to each row in the positive and negative columns
def apply_sentiment(data):
    data['positive_sentiment'] = data['Positive_Review'].apply(get_sentiment)
    data['negative_sentiment'] = data['Negative_Review'].apply(get_sentiment)
    return data


# # Preprocessing

# ## Missing Vlaues && Endcoding

# In[4]:


def split_Review_Date(data):
    # split Review_Date
    # df = pd.DataFrame()
    data['Review_Date'] = pd.to_datetime(data['Review_Date'])
    data['Year'] = data['Review_Date'].dt.year
    data['Month'] = data['Review_Date'].dt.month
    data['Day'] = data['Review_Date'].dt.day
    data = data.drop(["Review_Date"], axis=1)
    return data

def split_Tags_days_since(data):
    # split Tags
    Trip = []
    Members = []
    Room_Kind = []
    Nights = []
    the_way_of_submission = []
    Tags = list(data['Tags'])
    tag = []
    days = list(data['days_since_review'])
    days_final = []
    for i in range(len(Tags)):
        Tags[i] = Tags[i].replace("[", "")
        Tags[i] = Tags[i].replace("]", "")
        Tags[i] = Tags[i].replace("'", "")
        tag.append(Tags[i].split(","))
        # clean days_since review
        days[i] = str(days[i]).replace("days", "")
        days[i] = str(days[i]).replace("day", "")
        days_final.append(int(days[i]))
    for j in tag:
        trip = False
        mem = False
        room = False
        night = False
        submission = False
        for k in j:
            if k.__contains__("trip"):
                Trip.append(k)
                trip = True
            elif k.__contains__('room') or k.__contains__('Room'):
                Room_Kind.append(k)
                room = True
            elif (k.__contains__('Stayed')):
                result = ''.join(char for char in k if char.isdigit())
                Nights.append(int(result))
                night = True
            elif k.__contains__('Submitted'):
                the_way_of_submission.append(k)
                submission = True
            elif k.__contains__('Couple') or k.__contains__('Group') or k.__contains__('children') or k.__contains__(
                    'traveler'):
                Members.append(k)
                mem = True
        if not trip:
            Trip.append('no trip')
        if not room:
            Room_Kind.append('no room')
        if not night:
            Nights.append(0)
        if not submission:
            the_way_of_submission.append('not Submitted')
        if not mem:
            Members.append('no one')
    data['Trip'] = Trip
    data['Members'] = Members
    data['Room'] = Room_Kind
    data['Nights'] = Nights
    data['Submission'] = the_way_of_submission
    data = data.drop(['Tags'], axis=1)
    data['days_since_review'] = days_final
    return data

def data_cleaning_X_Train(X_train):

    X_train['Positive_Review'] = X_train['Positive_Review'].replace('No Positive', np.nan)
    X_train['Negative_Review'] = X_train['Negative_Review'].replace('No Negative', np.nan)

    # drop columns with all nan values
    X_train = X_train.dropna(axis=1, how="all")


    mean_list = []
    mean_list.append(X_train['Average_Score'].mean())
    mean_list.append(X_train['Total_Number_of_Reviews'].mean())  # ask
    mean_list.append(X_train['lat'].mean())
    mean_list.append(X_train['lng'].mean())


    # data with filling missing values (X_train)
    X_train = X_train.fillna({'Hotel_Address': 'no address', 'Additional_Number_of_Scoring': 0.0,
                              'Review_Date': "00/00/0000", 'Average_Score': mean_list[0], 'Hotel_Name': 'no name',
                              'Reviewer_Nationality': 'no info', 'Negative_Review': 'positive',
                              'Review_Total_Negative_Word_Counts': 0.0, 'Total_Number_of_Reviews': mean_list[1],
                              'Positive_Review': 'negative', 'Review_Total_Positive_Word_Counts': 0.0,
                              'Total_Number_of_Reviews_Reviewer_Has_Given': 0.0,
                              'Tags': "[' no trip ', ' no ', ' no Room ', ' 0 nights ', 'not Submitted']",
                              'days_since_review': '0 day', 'lat': mean_list[2], 'lng': mean_list[3]}) #, 'Reviewer_Score': 0.0})


    # split Review_Date
    X_train = split_Review_Date(X_train) # For X_train

    # split Tags
    X_train = split_Tags_days_since(X_train) # For X_train

    return X_train,mean_list

def data_cleaning_X_Test(X_test,mean_list):

    X_test['Positive_Review'] = X_test['Positive_Review'].replace('No Positive', np.nan)
    X_test['Negative_Review'] = X_test['Negative_Review'].replace('No Negative', np.nan)


    # drop columns with all nan values
    X_test = X_test.dropna(axis=1, how="all")


    # data with filling missing values (X_test)
    X_test = X_test.fillna({'Hotel_Address': 'no address', 'Additional_Number_of_Scoring': 0.0,
                            'Review_Date': "00/00/0000", 'Average_Score': mean_list[0], 'Hotel_Name': 'no name',
                            'Reviewer_Nationality': 'no info', 'Negative_Review': 'positive',
                            'Review_Total_Negative_Word_Counts': 0.0, 'Total_Number_of_Reviews': mean_list[1],
                            'Positive_Review': 'negative', 'Review_Total_Positive_Word_Counts': 0.0,
                            'Total_Number_of_Reviews_Reviewer_Has_Given': 0.0,
                            'Tags': "[' no trip ', ' no ', ' no Room ', ' 0 nights ', 'not Submitted']",
                            'days_since_review': '0 day', 'lat': mean_list[2], 'lng': mean_list[3]}) #, 'Reviewer_Score': 0.0})


    # split Review_Date
    X_test = split_Review_Date(X_test) # For X_train

    # split Tags
    X_test = split_Tags_days_since(X_test) # For X_train

    return X_test

# Handle The Missing Values In The Reviewer_Score With Tec: Predictive imputation

def data_cleaning_Y_Train(y_train):

    mean = y_train.mean()
    y_train = y_train.fillna({'Reviewer_Score': mean})

    return y_train , mean

def data_cleaning_Y_Test(y_test, mean):

    y_test = y_test.fillna({'Reviewer_Score': mean})

    return y_test


from collections import defaultdict
class Label_Encoder:
        def __init__(self):
            self.label_dict = defaultdict(int)
            self.label_count = 0

        def Fit_Or_Transform(self,labels):

            transformed = np.zeros(len(labels), dtype=int)

            for i, label in enumerate(labels):
                if label in self.label_dict:
                    transformed[i] = self.label_dict[label]
                else:
                    self.label_dict[label] = self.label_count
                    self.label_count += 1
                    transformed[i] = self.label_dict[label]

            return transformed



def data_encoding_X_Train(X_train):
    # Creating a instance of label Encoder.
    le = Label_Encoder()
    # printing label For (X_train) "Fit" and "Transform"
    X_train['Hotel_Name'] = le.Fit_Or_Transform(X_train['Hotel_Name'])
    X_train['Hotel_Address'] = le.Fit_Or_Transform(X_train['Hotel_Address'])
    X_train['Reviewer_Nationality'] = le.Fit_Or_Transform(X_train['Reviewer_Nationality'])
    X_train['Trip'] = le.Fit_Or_Transform(X_train['Trip'])
    X_train['Members'] = le.Fit_Or_Transform(X_train['Members'])
    X_train['Submission'] = le.Fit_Or_Transform(X_train['Submission'])
    X_train['Room'] = le.Fit_Or_Transform(X_train['Room'])

    return X_train,le


def data_encoding_X_Test(X_test,le):

    # printing label For (X_test) "Already Fitted" just "Transform"
    X_test['Hotel_Name'] = le.Fit_Or_Transform(X_test['Hotel_Name'])
    X_test['Hotel_Address'] = le.Fit_Or_Transform(X_test['Hotel_Address'])
    X_test['Reviewer_Nationality'] = le.Fit_Or_Transform(X_test['Reviewer_Nationality'])
    X_test['Trip'] = le.Fit_Or_Transform(X_test['Trip'])
    X_test['Members'] = le.Fit_Or_Transform(X_test['Members'])
    X_test['Submission'] = le.Fit_Or_Transform(X_test['Submission'])
    X_test['Room'] = le.Fit_Or_Transform(X_test['Room'])

    return X_test

def data_encoding_Y_Train(y_train):
    #the predict Column

    # Oridinal Encoding
    categories_order = ['Low_Reviewer_Score', 'Intermediate_Reviewer_Score','High_Reviewer_Score']
    # create an instance of OrdinalEncoder with the defined categories order
    ordinal_encoder = OrdinalEncoder(categories=[categories_order])
    # encode the data using the ordinal encoder (y_Train) "Fit" and "Transform"
    y_train_encoded = ordinal_encoder.fit_transform(y_train.values.reshape(-1, 1))
    y_train_encoded = pd.Series(y_train_encoded.reshape(-1), index= y_train.index)

    return y_train_encoded,ordinal_encoder

def data_encoding_Y_Test(y_test,ordinal_encoder):

    # encode the data using the ordinal encoder (y_Test) "already Fitted" just "Transform"
    y_test_encoded = ordinal_encoder.transform(y_test.values.reshape(-1, 1))
    y_test_encoded = pd.Series(y_test_encoded.reshape(-1), index= y_test.index)

    return y_test_encoded


# ## Scaling

# In[5]:


def standard(x,cols):
    scaler = StandardScaler()
    #cols=x.columns
    x = scaler.fit_transform(x)  # calc mean and standard division
    x = pd.DataFrame(x, columns= cols)
    return x,scaler

def norm(x,cols):
    scaler = MinMaxScaler()
    #cols=x.columns
    scaler.fit_transform(x)  # calc X_max and X_min
    # # X_norm = (X_old - X_min) / (X_max - X_min)
    x = pd.DataFrame(x, columns= cols)
    return x,scaler

def scaling(x, IsTrain,IsStandard=None,scaler=None,cols=None):
    if (IsTrain == True):
        if (IsStandard == True):
            x,scaler = standard(x,cols)
        else:
            x,scaler = norm(x,cols)
        return x,scaler

    else: # this is test
        #cols = x.columns
        x = scaler.transform(x)  # calc mean and standard division (Standard) Or calc X_max and X_min (Norm)
        x = pd.DataFrame(x, columns= cols)
        return x


# ## Handle Outliers

# In[6]:


def trimming(data, name, upper_limit, lower_limit):
    update_data = data.loc[(data[name] <= upper_limit) & (data[name] >= lower_limit)]
    return update_data

def capping(data, name, upper_limit, lower_limit):
    new_data = data.copy()
    new_data.loc[data[name] > upper_limit, name] = upper_limit
    new_data.loc[data[name] < lower_limit, name] = lower_limit
    return new_data

def IQR(data, type):
    # method2 : IQR
    n = len(data.columns)
    for i in range(n):
        name = data.columns[i]
        q1 = data[name].quantile(0.25)
        q3 = data[name].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr
        # rmoval function
        if (type == 1):
            data = capping(data, name, upper_limit, lower_limit)
        elif (type == 2):
            data = trimming(data, name, upper_limit, lower_limit)
    return data

def z_score(data, type):
    n = len(data.columns)
    for i in range(n):
        name = data.columns[i]
        upper_limit = data[name].mean() + 3 * data[name].std()
        lower_limit = data[name].mean() - 3 * data[name].std()
        # rmoval function
        if (type == 1):
            data = capping(data, name, upper_limit, lower_limit)
        elif (type == 2):
            data = trimming(data, name, upper_limit, lower_limit)
    return data

def percentile(data, type):
    # method2 : IQR
    n = len(data.columns)
    for i in range(n):
        name = data.columns[i]
        upper_limit = data[name].quantile(0.99)
        lower_limit = data[name].quantile(0.01)
        # rmoval function
        if (type == 1):
            data = capping(data, name, upper_limit, lower_limit)
        elif (type == 2):
            data = trimming(data, name, upper_limit, lower_limit)
    return data

def handle_outliers(data, method, type):
    if (method == 'iqr'):
        data = IQR(data, type)
    elif (method == 'p'):
        data = percentile(data, type)
    elif (method == 'z'):
        data = z_score(data, type)
    return data


# ## Feature Selection

# In[7]:


from sklearn.feature_selection import VarianceThreshold

def ANOVA_ftest(x, y):
    fs = SelectKBest(score_func=f_classif, k=4)
    # fs = SelectKBest(score_func=mutual_info_classif, k=4)
    fs.fit(x, y.values.ravel())
    # transform train input data
    X_train_fs = fs.transform(x)
    np.seterr(invalid='ignore')
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    selected_features = x.columns[fs.get_support()]
    x = x[selected_features]
    return x, y

def feature_selection(x, y):
    constant_filter = VarianceThreshold(threshold=0)

    # Fit and transforming on train data
    data_constant = constant_filter.fit_transform(x)

    # Extracting all constant columns using get support function of our filter
    constant_columns = [column for column in x.columns if column not in x.columns[constant_filter.get_support()]]
    x = x.drop(constant_columns, axis=1)
    qcons_filter = VarianceThreshold(threshold=0.01)

    # Fit and transforming on train data
    data_qcons = qcons_filter.fit_transform(x)

    # Extracting all Quasi constant columns using get support function of our filter
    qcons_columns = [column for column in x.columns if column not in x.columns[qcons_filter.get_support()]]
    x = x.drop(qcons_columns, axis=1)

    # check duplicated
    '''data_qcons_t = x.T
    data_qcons_t.shape
    x = data_qcons_t.drop_duplicates(keep='first').T'''
    # x = x.astype('int')
    x, y = ANOVA_ftest(x, y)
    return x, y


# # Save & Load Model

# In[8]:


def save_model(model, filename):
    filename ='Models'+'/'+ filename
    # Check if the file exists
    if os.path.exists(filename):
        # oberwrite the model to file
        joblib.dump(model, filename)
        print(f"Model Overwrited to '{filename}'.")
    else:
        # Save the model to file
        joblib.dump(model, filename)
        print(f"Model saved to '{filename}'.")


# In[9]:


def load_model(filename):
    filename ='Models'+'/'+ filename
    # Check if the file exists
    if os.path.exists(filename):
        # Load the model from file
        model = joblib.load(filename)
        print(f"Model loaded from '{filename}'.")
        return model
    else:
        print(f"File '{filename}' does not exist.")
        return None


# # Accuracy & Confusion Matrix

# In[10]:


def Print_Accuracy(model,X_data,y_data,data_name):
    print("--------------------------------------------------")
    # Use the classifier to make predictions on the test data
    predictions = model.predict(X_data)
    print("MSE : %f " % (metrics.mean_squared_error(y_data, predictions)))
    accuracy = r2_score(y_data , predictions)
    print(f"{str(model)}: {data_name} accuracy: {accuracy:.2%}")
    print("--------------------------------------------------")
    return predictions

def Display_Confusion_Matrix(y_data, predictions_data,model_name):
    import matplotlib.pyplot as plt

    # Assuming you have the predicted values in 'y_pred' and actual values in 'y_actual'
    plt.scatter(y_data, predictions_data)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(model_name)
    plt.show()
    return

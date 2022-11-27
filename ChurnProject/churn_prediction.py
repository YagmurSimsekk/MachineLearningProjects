import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.info()
df.SeniorCitizen.value_counts()
df.Partner.value_counts()

# total_charges = pd.to_numeric(df.TotalCharges, errors="coerce")
# df[total_charges.isnull()][["customerID", "TotalCharges"]]

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors="coerce") #coerce is to skip non-numeric data such as spaces
df.TotalCharges = df.TotalCharges.fillna(0)

df.columns = df.columns.str.lower().str.replace(" ", "_")
string_columns = list(df.dtypes[df.dtypes == "object"].index)


for col in string_columns:
    df[col] = df[col].str.lower().str.replace(" ", "_")

df.churn = (df.churn == "yes").astype(int)

#df is the all dataset and it is divided to df_train_full and df_test
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)

#then the train data set is divided to train and validation
df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state = 11)

# Exploratory Data Analysis

df_train_full.isnull().sum()
df_train_full.churn.value_counts()

global_mean = round(df_train_full.churn.mean(), 3)

categorical = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
]
numerical = ["tenure", "monthlycharges", "totalcharges"]

df_train_full[categorical].nunique()



female_mean = df_train_full[df_train_full.gender == "female"].churn.mean() #mean = the percentage of churn females/all females
male_mean = df_train_full[df_train_full.gender == "male"].churn.mean()

partner_yes = df_train_full[df_train_full.partner == "yes"].churn.mean()
partner_no = df_train_full[df_train_full.partner == "no"].churn.mean()


df_group = df_train_full.groupby(by="gender").churn.agg(["mean"])
df_group["diff"] = df_group["mean"] - global_mean
df_group["risk"] = df_group["mean"] / global_mean

# risk is close to 1: this people has same risk as anyone else. Close to 1 is not risky. 
# risk 0.5 means that this group of people is two times less likely to churn compared to rest
# risk is over 1: this group has more churn inside than the rest. more likely to churn 

for col in categorical:
    df_group = df_train_full.groupby(by=col).churn.agg(["mean"])
    df_group["diff"] = df_group["mean"] - global_mean
    df_group["risk"] = df_group["mean"] / global_mean
    print(df_group)


def calculate_mi(series):
    return mutual_info_score(series, df_train_full.churn)

#higher mutual info = higher degree of dependence

df_mi = df_train_full[categorical].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name="MI")

#for numerical values correlation is applied instead of mutual info
df_train_full[numerical].corrwith(df_train_full.churn)

train_dict = df_train[categorical + numerical].to_dict(orient="records")
# records, veri seti içerisindeki her bir satır
# churn değişkeni hedef değişken olduğundan dolayı alınmadı.

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)
X_train = dv.transform(train_dict) #converts dictionaries to matrix
dv.get_feature_names_out()

y_train = df_train["churn"].values

# logistic model is like a linear but in linear we could predict the price of a car(numeric), here it is binary (churn =1, not churn = 0)
# sigmoid function maps any value between zero and one
model = LogisticRegression(solver="liblinear", random_state=1)
model.fit(X_train, y_train)
# model is trained

#check how good model works, use validation data set
val_dict = df_val[categorical + numerical].to_dict(orient="records")
y_val = df_val["churn"].values
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1] #takes only churn values(Second column)

churn = y_pred >= 0.5
(y_val == churn).mean() # 80 percent predicted correct = accuracy


model.intercept_[0]
model.coef_[0]

dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))


small_subset = ["contract", "tenure", "totalcharges"]
train_dict_small = df_train[small_subset].to_dict(orient="records")
dv_small = DictVectorizer(sparse=False)
dv_small.fit(train_dict_small)
X_small_train = dv_small.transform(train_dict_small)

dv_small.get_feature_names_out()
model_small = LogisticRegression(solver="liblinear", random_state=1)
model_small.fit(X_small_train, y_train) # amount of data is the same(rows), only columns are less. y_train = real churn values for all

#validation set for small
val_dict_small = df_val[small_subset].to_dict(orient="records")
y_val = df_val["churn"].values
X_val_small = dv_small.transform(val_dict_small)

y_pred_small = model_small.predict_proba(X_val_small)[:, 1] #takes only churn values(Second column)

churn_small = y_pred_small >= 0.5
(y_val == churn_small).mean() 


model_small.intercept_[0]

dict(zip(dv_small.get_feature_names_out(), model_small.coef_[0].round(3)))

customer = {
    "customerid": "8879-zkjof",
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "no",
    "tenure": 41,
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "dsl",
    "onlinesecurity": "yes",
    "onlinebackup": "no",
    "deviceprotection": "yes",
    "techsupport": "yes",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "one_year",
    "paperlessbilling": "yes",
    "paymentmethod": "bank_transfer_(automatic)",
    "monthlycharges": 79.85,
    "totalcharges": 3320.75,
}

X_test = dv.transform([customer]) #uses DictVectorizer: converts dict to matrix

model.predict_proba(X_test)[:, 1] #churn olma ihtimali

customer = {
    "gender": "female",
    "seniorcitizen": 1,
    "partner": "no",
    "dependents": "no",
    "phoneservice": "yes",
    "multiplelines": "yes",
    "internetservice": "fiber_optic",
    "onlinesecurity": "no",
    "onlinebackup": "no",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "yes",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 85.7,
    "totalcharges": 85.7,
}

X_test = dv.transform([customer])
model.predict_proba(X_test)[0, 1] #0.83 = most probably churn


def train(df, y):
    cat = df[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver="liblinear")
    model.fit(X, y)

    return dv, model


def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient="records")

    X = dv.transform(cat)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


kfold = KFold(n_splits=10, shuffle=True, random_state=1)
aucs = []

#in this way two variables iterates over df_train_full, it doesn't overlap
for train_idx, val_idx in kfold.split(df_train_full):
    df_train = df_train_full.iloc[train_idx]
    df_val = df_train_full.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    aucs.append(auc)

aucs

print("auc = %0.3f ± %0.3f" % (np.mean(aucs), np.std(aucs)))


def train(df, y, C):
    cat = df[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver="liblinear", C=C)
    model.fit(X, y)

    return dv, model


nfolds = 10
kfold = KFold(n_splits=nfolds, shuffle=True, random_state=1)

for C in [0.001, 0.01, 0.1, 0.5, 1, 10]:
    aucs = []
    for train_idx, val_idx in kfold.split(df_train_full):
        df_train = df_train_full.iloc[train_idx]
        df_val = df_train_full.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
    print("C=%s, auc = %0.3f ± %0.3f" % (C, np.mean(aucs), np.std(aucs)))

#when c parameter is small, the model is more regularized = weights are smaller = model will behave the same on a real data set
y_train = df_train_full.churn.values
y_test = df_test.churn.values

dv, model = train(df_train_full, y_train, C=0.5)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
print(auc)

customer = {
    "customerid": "8879-zkjof",
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "no",
    "tenure": 41,
    "phoneservice": "yes",
    "multiplelines": "no",
    "internetservice": "dsl",
    "onlinesecurity": "yes",
    "onlinebackup": "no",
    "deviceprotection": "yes",
    "techsupport": "yes",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "one_year",
    "paperlessbilling": "yes",
    "paymentmethod": "bank_transfer_(automatic)",
    "monthlycharges": 79.85,
    "totalcharges": 3320.75,
}

# df = pd.DataFrame([customer])
# y_pred = predict(df, dv, model)
# y_pred[0]


def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred[0]


predict_single(customer, dv, model)


with open("churn-model.bin", "wb") as f_out:
    pickle.dump((dv, model), f_out)

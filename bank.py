# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:52:05 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
bank= pd.read_csv('bank-full.csv', sep=";")
bank.drop(['poutcome'],axis=1,inplace=True)
bank.shape
bank.head()
bank.tail()
bank.columns
###bank.drop(bank[bank[:3:15]=='unknown'],axis=1, inplace=True)
##bank.drop(bank.iloc[:]=='unknown',axis=1)
bank.shape
bank.isnull().sum()
import seaborn as sns
bank.describe()


###factorizing
###bank['jobs']=pd.factorize(bank.job)[0]
#3bank.drop(['job'],axis=1,inplace=True)
##bank
####3bank['martials']=pd.factorize(bank.marital)[0]
#bank['educations']=pd.factorize(bank.education)[0]
##bank['defaults']=pd.factorize(bank.default)[0]
#bank['housings']=pd.factorize(bank.housing)[0]
##bank['loans']=pd.factorize(bank.loan)[0]
#bank['contacts']=pd.factorize(bank.contact)[0]
#bank['months']=pd.factorize(bank.month)[0]
#bank.drop(['education','default','housing','loan','contact','month'],axis=1,inplace=True)
#bank.drop(['marital'],axis=1,inplace=True)
#bank

import statsmodels.formula.api as sm
from scipy import stats
import scipy.stats as st

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
bank['jobs']=le.fit_transform(bank.job)
bank.jobs.unique()
bank['maritals']=le.fit_transform(bank.marital)
bank.maritals.unique()
bank['educations']=le.fit_transform(bank.education)
bank.educations.unique()
bank['defaults']=le.fit_transform(bank.default)
bank.defaults.unique()
bank['housings']=le.fit_transform(bank.housing)
bank.housings.unique()
bank['loans']=le.fit_transform(bank.loan)
bank.loans.unique()
bank['contacts']=le.fit_transform(bank.contact)
bank.contacts.unique()
bank['months']=le.fit_transform(bank.month)
bank.months.unique()
bank['yy']=le.fit_transform(bank.y)
bank.yy.unique()
bank.drop(['y','education','default','housing','loan','contact','month','marital'],axis=1,inplace=True)
bank.drop(['job'], axis=1, inplace=True)
bank
##bank=pd.get_dummies(bank,columns=['job'], prefix='job')
##bank
bank.columns
##job=pd.concat(bank,['job_admin', 'job_blue-collar', 'job_entrepreneur','job_housemaid', 'job_management', 'job_retired', 'job_self-employed','job_services', 'job_student', 'job_technician', 'job_unemployed','job_unknown'],axis=1,inplace=True)
########3from sklearn.preprocessing import LabelEncoder,OneHotEncoder
######3from sklearn.preprocessing import Normalizer
####33x=bank.iloc[:,:].values
####3x
##z=pd.DataFrame(x)
#labelencoder_x=LabelEncoder()
##x[:,1]=labelencoder_x.fit_transform(x[:,1])

#z=pd.DataFrame(x)
###3onehotencoder=OneHotEncoder(categorical_features=[1])
#####x= onehotencoder.fit_transform(x).toarray()
bank.columns
bank.corr()
banks=pd.DataFrame(bank)
banks.shape
import statsmodels.formula.api as sm
logit_model=sm.logit('yy~age+balance+day+duration+campaign+pdays+previous+jobs+maritals+educations+defaults+housings+loans+contacts+months',data=banks).fit()
logit_model.summary()
logit_model2=sm.logit('yy~age+balance+day+duration+campaign+pdays+previous+maritals+educations+defaults+housings+loans+contacts+months',data=banks).fit()
logit_model2.summary()
y_pred=logit_model2.predict(banks)
y_pred
banks['pred_prob']=y_pred
banks
banks["Att_val"] = 0
bankss.loc[y_pred>=0.5,"Att_val"] = 1
banks.Att_val
from sklearn.metrics import classification_report
classification_report(banks.Att_val,banks.yy)
confusion_matrix = pd.crosstab(banks['yy'],banks.Att_val)
confusion_matrix
accuracy=(39150+1112)/(39150+772+4177+1112)
accuracy
######0.89 accuracy

##roc curve
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(bank.yy, y_pred)
plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")


# area under ROC curve
roc_auc = metrics.auc(fpr, tpr)  
roc_auc #0.87


#####divide the data into train and test
banks.drop('Att_val', axis=1, inplace=True)
banks
from sklearn.model_selection import train_test_split
train,test= train_test_split(banks,test_size=0.3)
train.isnull().sum();test.isnull().sum()

###train data model
train_model=sm.logit('yy~age+balance+day+duration+campaign+pdays+previous+maritals+educations+defaults+housings+loans+contacts+months',data=banks).fit()
train_model.summary()
train_pred = train_model.predict(train.iloc[:,:])
print(train.iloc[:,:])
###new column for storing values
train['train_pred']=0
train.loc[train_pred>=0.5,"train_pred"] = 1
train_pred
confusion_matrix = pd.crosstab(train['yy'],train.train_pred)
confusion_matrix
accuracy_train=(27425+781)/(27425+523+2918+781)
accuracy_train
####0.891

#####test
test_pred=train_model.predict(test)
test_pred
test['test_pred']=0
test.loc[test_pred>=0.5,"test_pred"]=1
testmatrix = pd.crosstab(test['yy'],test.test_pred)
testmatrix
 accuracytest=(11725+331)/(11725+249+1259+331)
accuracytest
###0.88
from sklearn.linear_model import LassoCV
import pandas as pd

train_data=pd.read_csv('D:\sufe\A\data_train_changed.csv')
train_data=train_data.ix[0:,1:].drop(['REPORT_ID',"ID_CARD",'LOAN_DATE'],1)
train_data=train_data.dropna()
# print(train_data.info())
X=train_data.drop(['Y'],1).as_matrix()#7
y=train_data['Y'].as_matrix()#1
lassocv = LassoCV()
lassocv.fit(X,y)
print(train_data.columns.drop('Y'),lassocv.coef_)
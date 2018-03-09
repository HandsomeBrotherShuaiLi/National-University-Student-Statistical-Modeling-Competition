import pandas as pd
from keras import Sequential
from keras.layers.core import Dense,Activation,Dropout
from sklearn.model_selection import train_test_split
# from matplotlib.pyplot import plt
train_data=pd.read_csv('D:\sufe\A\contest_basic_train.tsv',sep='\t')
train_data=train_data.drop(['REPORT_ID',"ID_CARD",'LOAN_DATE'],1)
train_data=train_data.dropna()
# print(train_data.info())
X=train_data.drop(['Y'],1).as_matrix()#7
y=train_data['Y'].as_matrix()#1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model=Sequential()
model.add(Dense(14,input_shape=(7,)))
model.add(Activation('relu'))
model.add(Dense(1))
model.add((Dropout(0.3)))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(X_train,y_train,epochs=10000,batch_size=16)
t=model.predict(X_test)

rate=0

for i in range(len(t)):
    if t[i]==y_test[i]:
        rate+=1
    else:
        pass
rate=1.0*rate/len(t)

print(rate)


# test_data=pd.read_csv('D:\sufe\A\contest_basic_test.tsv',sep='\t')
# test_data=test_data.dropna()
# test_data=test_data.drop(['REPORT_ID',"ID_CARD",'LOAN_DATE'],1)








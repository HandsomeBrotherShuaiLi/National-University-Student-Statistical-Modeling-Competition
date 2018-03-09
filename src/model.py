import pandas as pd
# from keras import Sequential
# from keras.layers.core import Dense,Activation,Dropout
from sklearn.model_selection import train_test_split
from sklearn import linear_model,tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.ensemble import BaggingClassifier
import pydotplus

# # from matplotlib.pyplot import plt
# from sklearn.svm import SVC
def str2level(data):
    for j in data.columns:#列
        for i in data.index:#行
            if j=='AGENT':#agent
                if data.ix[i,j]=="weijinhui":
                    data.ix[i,j]=0
                elif data.ix[i,j]=="orgloan":
                    data.ix[i, j]=1
                elif data.ix[i,j]=="bestpay":
                    data.ix[i,j]=2
                elif data.ix[i,j]=="kuaiqian":
                    data.ix[i,j]=3
                elif data.ix[i,j]=="ifpp":
                    data.ix[i, j] = 4
                elif data.ix[i,j]=="fenqile":
                    data.ix[i, j] = 5
                elif data.ix[i,j]=="huifusdb":
                    data.ix[i,j]=6
                elif data.ix[i,j]=="rongzhijia":
                    data.ix[i,j]=7
                elif data.ix[i,j]=="chinapnr":
                    data.ix[i,j]=8
                elif data.ix[i,j]=="wechat":
                    data.ix[i,j]=9
                elif data.ix[i,j]=="APP":
                    data.ix[i,j]=10

            if j=='IS_LOCAL':
                if data.ix[i,j]=="本地籍":
                    data.ix[i, j] =1
                else:
                    data.ix[i,j]=0
            if j=='EDU_LEVEL':#edu_level
                if data.ix[i,j]=="其他":
                    data.ix[i,j]=0
                elif data.ix[i,j]=="初中":
                    data.ix[i,j]=1
                elif data.ix[i,j]=="高中":
                    data.ix[i,j]=2
                elif data.ix[i,j]=="专科及以下":
                    data.ix[i,j]=3
                elif data.ix[i,j]=="专科":
                    data.ix[i,j]=4
                elif data.ix[i,j]=="本科":
                    data.ix[i,j]=5
                elif data.ix[i,j]=="硕士研究生":
                    data.ix[i,j]=6
                elif data.ix[i,j]=="博士研究生":
                    data.ix[i,j]=7
                elif data.ix[i,j]=="硕士及以上":
                    data.ix[i,j]=8
                # else:
                #     data.ix[i,j]=9
            if j=='MARRY_STATUS':#marry_status
                if data.ix[i,j]=="其他":
                    data.ix[i, j]=0
                elif data.ix[i,j]=="丧偶":
                    data.ix[i, j]=1
                elif data.ix[i,j]=="离婚":
                    data.ix[i, j]=2
                elif data.ix[i,j]=="离异":
                    data.ix[i, j]=3
                elif data.ix[i,j]=="未婚":
                    data.ix[i, j]=4
                elif data.ix[i,j]=="已婚":
                    data.ix[i, j]=5
                else:
                    data.ix[i, j]=6
    return data


train_data=pd.read_csv('D:\sufe\A\WoEdata.csv')

train_data=train_data.ix[0:,1:].drop(['REPORT_ID',"ID_CARD",'LOAN_DATE'],1)
train_data=train_data.dropna()
# print(train_data.info())
# train_data=str2level(train_data)

X=train_data.drop(['Y'],1).as_matrix()#7

y=train_data['Y'].as_matrix()#1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf=BaggingClassifier(clf,n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False)
bagging_clf.fit(X_train, y_train)

treemodel=DecisionTreeClassifier()
#treemodel=BaggingClassifier(treemodel,n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False)
treemodel.fit(X_train,y_train)

randomtree=linear_model.RidgeClassifier()
randomtree=BaggingClassifier(randomtree,n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False)
randomtree.fit(X_train,y_train)

sgd=linear_model.SGDClassifier(tol=1e-3)
sgd=BaggingClassifier(sgd,n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False)
sgd.fit(X_train,y_train)


dot_data = tree.export_graphviz(treemodel, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("决策树.pdf")

# model=Sequential()
# model.add(Dense(2*(X_train.shape[1]),input_shape=((X_train.shape[1]),)))
# model.add(Activation('relu'))
# model.add(Dense(1))
# model.add((Dropout(0.3)))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.summary()
#
# model.fit(X_train,y_train,epochs=10000,batch_size=50 )
# svmmodel=SVC()
# svmmodel.fit(X_train,y_test)

t=bagging_clf.predict(X_test)
joblib.dump(bagging_clf,'clf.model')

z=treemodel.predict(X_test)
joblib.dump(treemodel,'treemodel.model')

w=randomtree.predict(X_test)
joblib.dump(randomtree,'randomtree.model')

s=sgd.predict(X_test)
joblib.dump(sgd,'sgd.model')

# m=model.predict(X_test)
# model.save('NNmodel.h5')

rate1=0
rate2=0
rate3=0
rate4=0
# rate5=0
for i in range(len(t)):

    if t[i]==y_test[i]:
        rate1+=1
    if z[i]==y_test[i]:
        rate2+=1
    if w[i]==y_test[i]:
        rate3+=1
    if s[i]==y_test[i]:
        rate4+=1
    # if m[i]==y_test[i]:
    #     rate5+=1

rate1=1.0*rate1/len(t)
rate2=1.0*rate2/len(t)
rate3=1.0*rate3/len(t)
rate4=1.0*rate4/len(t)
print('逻辑回归的准确率是',rate1,'\n','决策树的准确率是',rate2,'\n','Ridge分类决策准确率是',rate3,'\n','SGD分类器准确率',rate4,
       '\n')





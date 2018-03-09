#试着补全AGENT WORK_PROVINCE EDU_LEVEL SALARY
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
#先要进行文字等级换成数字等级
#对于IS_LOCAL 非本地籍是0 本地籍是1
#对于学历 其他0 初中1 高中2 专科及以下3 专科 4 本科5 硕士研究生6 博士研究生7 硕士及以上8
#对于婚姻 其他0 丧偶1 离婚2 离异3 未婚4 已婚5
#对于agent weijinhui 0 orgloan 1 bestpay 2 kuaiqian 3 ifpp 4 fenqile 5 huifusdb 6 rongzhijia 7 chinapnr 8 wechat 9 APP 10
def str2level(data):
    for j in range(3,len(data.columns)):#列
        for i in range(len(data.index)):#行
            if j==100:#agent
                if data.iat[i,j]=="weijinhui":
                    data.iat[i,j]=0
                elif data.iat[i,j]=="orgloan":
                    data.iat[i, j]=1
                elif data.iat[i,j]=="bestpay":
                    data.iat[i,j]=2
                elif data.iat[i,j]=="kuaiqian":
                    data.iat[i,j]=3
                elif data.iat[i,j]=="ifpp":
                    data.iat[i, j] = 4
                elif data.iat[i,j]=="fenqile":
                    data.iat[i, j] = 5
                elif data.iat[i,j]=="huifusdb":
                    data.iat[i,j]=6
                elif data.iat[i,j]=="rongzhijia":
                    data.iat[i,j]=7
                elif data.iat[i,j]=="chinapnr":
                    data.iat[i,j]=8
                elif data.iat[i,j]=="wechat":
                    data.iat[i,j]=9
                elif data.iat[i,j]=="APP":
                    data.iat[i,j]=10

            if j==3:
                if data.iat[i,j]=="本地籍":
                    data.iat[i, j] =1
                else:
                    data.iat[i,j]=0
            if j==4:#edu_level
                if data.iat[i,j]=="其他":
                    data.iat[i,j]=0
                elif data.iat[i,j]=="初中":
                    data.iat[i,j]=1
                elif data.iat[i,j]=="高中":
                    data.iat[i,j]=2
                elif data.iat[i,j]=="专科及以下":
                    data.iat[i,j]=3
                elif data.iat[i,j]=="专科":
                    data.iat[i,j]=4
                elif data.iat[i,j]=="本科":
                    data.iat[i,j]=5
                elif data.iat[i,j]=="硕士研究生":
                    data.iat[i,j]=6
                elif data.iat[i,j]=="博士研究生":
                    data.iat[i,j]=7
                elif data.iat[i,j]=="硕士及以上":
                    data.iat[i,j]=8
                # else:
                #     data.iat[i,j]=9
            if j==5:#marry_status
                if data.iat[i,j]=="其他":
                    data.iat[i, j]=0
                elif data.iat[i,j]=="丧偶":
                    data.iat[i, j]=1
                elif data.iat[i,j]=="离婚":
                    data.iat[i, j]=2
                elif data.iat[i,j]=="离异":
                    data.iat[i, j]=3
                elif data.iat[i,j]=="未婚":
                    data.iat[i, j]=4
                elif data.iat[i,j]=="已婚":
                    data.iat[i, j]=5
                else:
                    data.iat[i, j]=6
    return data

#先补全fund ，再补全edu_level，再补全salary


def set_missing_HASFUND(data):
    funddata=data[["HAS_FUND","IS_LOCAL","MARRY_STATUS"]]
    known_fund=funddata[funddata.HAS_FUND.notnull()].as_matrix()
    unknow_fund=funddata[funddata.HAS_FUND.isnull()].as_matrix()
    y=known_fund[:,0]
    X=known_fund[:,1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
    rfr.fit(X, y)
    predictfund = rfr.predict(unknow_fund[:, 1::])
    data.loc[(data.HAS_FUND.isnull()), 'HAS_FUND'] = predictfund
    return data


def set_missing_edu(data):
    edudata=data[["EDU_LEVEL","IS_LOCAL","MARRY_STATUS","HAS_FUND"]]
    known_edu=edudata[edudata.EDU_LEVEL.notnull()].as_matrix()
    unknown_edu=edudata[edudata.EDU_LEVEL.isnull()].as_matrix()
    y=known_edu[:,0]
    X=known_edu[:,1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
    rfr.fit(X, y)
    predictedu = rfr.predict(unknown_edu[:, 1::])
    data.loc[(data.EDU_LEVEL.isnull()), 'EDU_LEVEL'] = predictedu
    return data
def set_missing_province(data):
    province=data[["WORK_PROVINCE","IS_LOCAL","MARRY_STATUS","HAS_FUND","EDU_LEVEL"]]
    known_province=province[province.WORK_PROVINCE.notnull()].as_matrix()
    unknown_province=province[province.WORK_PROVINCE.isnull()].as_matrix()
    y = known_province[:, 0]
    X = known_province[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
    rfr.fit(X, y)
    predictpro = rfr.predict(unknown_province[:, 1::])
    data.loc[(data.WORK_PROVINCE.isnull()), 'WORK_PROVINCE'] = predictpro
    return data
def set_missing_salary(data):
    salary_data=data[["SALARY","IS_LOCAL","EDU_LEVEL","MARRY_STATUS","HAS_FUND","develpment_level"]]
    known_salary=salary_data[salary_data.SALARY.notnull()].as_matrix()
    unknow_salary=salary_data[salary_data.SALARY.isnull()].as_matrix()
    #工资
    y=known_salary[:,0]
    #其他特征值
    X=known_salary[:,1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
    rfr.fit(X,y)
    predictsalary=rfr.predict(unknow_salary[:,1::])
    data.loc[(data.SALARY.isnull()),'SALARY']=predictsalary
    return data
def set_missing_salary_y(data):
    salary_data = data[["SALARY", "IS_LOCAL", "EDU_LEVEL", "MARRY_STATUS", "HAS_FUND", "develpment_level","Y"]]
    known_salary = salary_data[salary_data.SALARY.notnull()].as_matrix()
    unknow_salary = salary_data[salary_data.SALARY.isnull()].as_matrix()
    # 工资
    y = known_salary[:, 0]
    # 其他特征值
    X = known_salary[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, n_jobs=-1)
    rfr.fit(X, y)
    predictsalary = rfr.predict(unknow_salary[:, 1::])
    data.loc[(data.SALARY.isnull()), 'SALARY'] = predictsalary
    return data
data_train=pd.read_excel("D:\sufe\dataset1.xls")
data_train=str2level(data_train)
data_train2=data_train
# print(data_train.info(),'\n',data_train2.info())
# print(data_train.IS_LOCAL.value_counts(),'\n',data_train.MARRY_STATUS.value_counts(),'\n',data_train.HAS_FUND.value_counts())
# data_train=set_missing_HASFUND(data_train)
# data_train=set_missing_edu(data_train)
# data_train=set_missing_province(data_train)
# data_train_withY=set_missing_salary_y(data_train2)
# data_train_withY.to_csv("D:\sufe\A\dataset_train_withY.csv")

data_train_withoutY=set_missing_salary(data_train)
data_train_withoutY.to_csv("D:\sufe\A\dataset_train_withoutY.csv")

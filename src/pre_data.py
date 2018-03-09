def DeleteDuplicatedElementFromList(list):
    resultList = []
    for item in list:
        if not item in resultList and str(item)!="nan":
            resultList.append(item)
    return resultList
import pandas as pd
#coding:utf-8
import matplotlib.pyplot as plt
import numpy
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False
import numpy as np
from pandas import Series,DataFrame
data_train=pd.read_csv("D:\sufe\A\contest_basic_train.tsv", sep='\t')

print(data_train.info())
print(data_train.describe())

d1=pd.DataFrame(columns=['特征','未逾期比例','逾期比例'])



fig=plt.figure("Y值分类统计图")
fig.set(alpha=0.2)
data_train.Y.value_counts().plot(kind='bar')#我用柱状图
# plt.title(u"Y=0代表未逾期客户，Y=1代表逾期客户")
plt.ylabel(u"人数")
plt.savefig("Y值分类统计图.png")
fig=plt.figure('户籍统计情况图')
data_train.IS_LOCAL.value_counts().plot(kind='bar')
# plt.title(u"户籍情况")
plt.ylabel(u"人数")
plt.savefig("户籍统计情况图.png")

#属性对于目标变量的关联性统计（IS_LOCAL AGENT WORK_PROVINCE,EDU_LEVEL,MARRY_STATUS,SALATY,HAS_FUND)


#IS_LOCAL

Y1=data_train.IS_LOCAL[data_train.Y==0].value_counts()
Y2=data_train.IS_LOCAL[data_train.Y==1].value_counts()
df=pd.DataFrame({u'未逾期客户':Y1,u'逾期客户':Y2})
df.plot(kind='bar',stacked=True)
# plt.title(u"户籍因素中是否是逾期客户分析")
plt.xlabel(u"是否本地户籍")
plt.ylabel(u"人数")
print("本地与非本地因素中客户逾期分析")
print("未逾期客户人数\n",Y1)
print("逾期客户人数\n",Y2)
print("本地户籍中未逾期比例是",1.0*Y1["本地籍"]/(Y1["本地籍"]+Y2["本地籍"]),"  逾期比例是："
      ,1.0*Y2["本地籍"]/(Y1["本地籍"]+Y2["本地籍"]))

print("非本地户籍中逾期比例是",1.0*Y1["非本地籍"]/(Y1["非本地籍"]+Y2["非本地籍"]),"  逾期比例是："
      ,1.0*Y2["非本地籍"]/(Y1["非本地籍"]+Y2["非本地籍"]))
d1.iat["本地籍",0]=1.0*Y1["本地籍"]/(Y1["本地籍"]+Y2["本地籍"])
d1.iat["本地籍",1]=1.0*Y2["本地籍"]/(Y1["本地籍"]+Y2["本地籍"])
d1.iat["非本地籍",0]=1.0*Y1["非本地籍"]/(Y1["非本地籍"]+Y2["非本地籍"])
d1.iat["非本地籍",1]=1.0*Y2["非本地籍"]/(Y1["非本地籍"]+Y2["非本地籍"])
plt.savefig('户籍对于逾期的影响.png')
#WORK_PRO工作省份
Y3=data_train.WORK_PROVINCE[data_train.Y==0].value_counts()
Y4=data_train.WORK_PROVINCE[data_train.Y==1].value_counts()
df=pd.DataFrame({u'未逾期客户':Y3,u'逾期客户':Y4})
df.plot(kind='bar',stacked=True)
# plt.title(u"工作省份中是否是逾期客户分析")
plt.xlabel(u"工作省份")
plt.ylabel(u"人数")
plt.savefig('工作省份对于逾期的影响.png')
# listA=data_train["WORK_PROVINCE"]
# b=DeleteDuplicatedElementFromList(listA)
#
# print(b)
# for i in b:
#     print(i,"中的未逾期比例",1.0*Y3[i]/(Y3[i]+Y4[i]),"  逾期比例是",1.0*Y4[i]/(Y3[i]+Y4[i]))

#EDU_LEVEL
Y3=data_train.EDU_LEVEL[data_train.Y==0].value_counts()
Y4=data_train.EDU_LEVEL[data_train.Y==1].value_counts()
df=pd.DataFrame({u'未逾期客户':Y3,u'逾期客户':Y4})
df.plot(kind='bar',stacked=True)
# plt.title(u"学历与逾期客户分析")
plt.xlabel(u"学历")
plt.ylabel(u"人数")
print("学历与是否是逾期客户的分析")
print("未逾期客户人数：\n",Y3)
print("逾期客户人数：\n",Y4)
d2=pd.DataFrame(columns=['未逾期比例','逾期比例'])
b=["专科","本科","高中","专科及以下","初中","其他","硕士研究生","博士研究生","硕士及以上"]
d2.index=b
for i in b:
    print(i,"中的未逾期比例",1.0*Y3[i]/(Y3[i]+Y4[i]),"  逾期比例是",1.0*Y4[i]/(Y3[i]+Y4[i]))
    d2.iat[i,0]=1.0*Y3[i]/(Y3[i]+Y4[i])
    d2.iat[i,1]=1.0*Y4[i]/(Y3[i]+Y4[i])
plt.savefig('学历的影响.png')
#Marry_STATUS
Y3=data_train.MARRY_STATUS[data_train.Y==0].value_counts()
Y4=data_train.MARRY_STATUS[data_train.Y==1].value_counts()
df=pd.DataFrame({u'未逾期客户':Y3,u'逾期客户':Y4})
df.plot(kind='bar',stacked=True)
# plt.title(u"婚姻情况与逾期客户分析")
plt.xlabel(u"婚姻情况")
plt.ylabel(u"人数")
d3=pd.DataFrame(columns=['未逾期比例','逾期比例'])
print("婚姻情况与逾期情况分析")
print("未逾期客户人数：\n",Y3)
print("逾期客户人数：\n",Y4)
b=["丧偶","其他","已婚","未婚","离婚","离异"]
d3.index=b
for i in b:
    print(i,"中的未逾期比例",1.0*Y3[i]/(Y3[i]+Y4[i]),"  逾期比例是",1.0*Y4[i]/(Y3[i]+Y4[i]))
    d3.iat[i,0]=1.0*Y3[i]/(Y3[i]+Y4[i])
    d3.iat[i,1]=1.0*Y4[i]/(Y3[i]+Y4[i])

plt.savefig('婚姻的影响.png')
#has_fund
Y3=data_train.HAS_FUND[data_train.Y==0].value_counts()
Y4=data_train.HAS_FUND[data_train.Y==1].value_counts()
df=pd.DataFrame({u'未逾期客户':Y3,u'逾期客户':Y4})
df.plot(kind='bar',stacked=True)
# plt.title(u"公积金与逾期客户分析")
plt.xlabel(u"公积金情况")
plt.ylabel(u"人数")
print("公积金（0-无公积金,1-有公积金）与逾期客户分析")
b=[0,1]
d4=pd.DataFrame(columns=['未逾期比例','逾期比例'])
d4.index=b
for i in b:
    print(i,"中的未逾期比例",1.0*Y3[i]/(Y3[i]+Y4[i]),"  逾期比例是",1.0*Y4[i]/(Y3[i]+Y4[i]))
    d4.iat[i,0]=1.0*Y3[i]/(Y3[i]+Y4[i])
    d4.iat[i,1]=1.0*Y4[i]/(Y3[i]+Y4[i])
plt.savefig('公积金的影响.png')
#salary
print("salary的基本情况")
print(data_train["SALARY"].describe())
Y3=data_train.SALARY[data_train.Y==0].value_counts()
Y4=data_train.SALARY[data_train.Y==1].value_counts()
df=pd.DataFrame({u'未逾期客户':Y3,u'逾期客户':Y4})
df.plot(kind='bar',stacked=True)
# plt.title(u"工资与逾期客户分析")
plt.xlabel(u"工资情况")
plt.ylabel(u"人数")
plt.savefig('工资的影响.png')
print("有工资记录下的逾期比例分析")
b=[1,2,3,4,5,6,7]
d5=pd.DataFrame(columns=['未逾期比例','逾期比例'])
d5.index=b
for i in b:
    print(i,"中的未逾期比例",1.0*Y3[i]/(Y3[i]+Y4[i]),"  逾期比例是",1.0*Y4[i]/(Y3[i]+Y4[i]))
    d5.iat[i,0]=1.0*Y3[i]/(Y3[i]+Y4[i])
    d5.iat[i,1]=1.0*Y4[i]/(Y3[i]+Y4[i])
# print("无工资记录情况下的比例分析")
# nanSalary=data_train.groupby(by=["SALARY","Y"])
# print(nanSalary.size())
# num0=0
# num1=0
# numn=0
# for i in range(len(data_train)):
#     if str(data_train['SALARY'])=="nan" and data_train['Y']==0:
#         num0=num0+1
#         numn=numn+1
#     elif str(data_train['SALARY'])=="nan" and data_train['Y']==1:
#         num1=num1+1
#         numn=numn+1
# print(1.0*num0/numn,1.0*num1/numn)
d1=d1.append(d2,ignore_index=False)
print(d1)
plt.show()

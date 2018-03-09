import numpy as np
import pandas as pd
import scipy.stats.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
def mono_bin(Y, X, n = 20):
    r = 0
    good=Y.sum()
    bad=Y.count()-good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X.rank(method='first'), n)})

        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe']=np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_values(by='min')).reset_index(drop=True)
    print("=" * 60)
    print(d4)
    cut = []
    cut.append(float('-inf'))
    for i in range(1, n + 1):
        qua = X.quantile(i / (n + 1))
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return iv,cut,woe

def replace_woe(series,cut,woe):
    list=[]
    i=0
    while i<len(series):
        value=series[i]
        j=len(cut)-2
        m=len(cut)-2
        while j>=0:
            if value>=cut[j]:
                j=-1
            else:
                j -=1
                m -= 1
        list.append(woe[m])
        i += 1
    return list

train_data=pd.read_csv("D:\sufe\A\dataset_train_withY.csv")
v2,cut2,woe2=mono_bin(train_data.Y,train_data.EDU_LEVEL)
v1,cut1,woe1=mono_bin(train_data.Y,train_data.IS_LOCAL)
v3,cut3,woe3=mono_bin(train_data.Y,train_data.MARRY_STATUS)
v4,cut4,woe4=mono_bin(train_data.Y,train_data.SALARY)
v5,cut5,woe5=mono_bin(train_data.Y,train_data.HAS_FUND)
v6,cut6,woe6=mono_bin(train_data.Y,train_data.develpment_level)
corr=train_data.corr()
xlabels=corr.index[1:]
ylabels=corr.index[1:]
fig=plt.figure("关联性图")
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True,cmap='rainbow')
#, cmap='rainbow', ax=ax1, annot_kws={'size': 8, 'weight': 'bold', 'color': 'blue'}
ax1.set_xticklabels(xlabels, rotation=0, fontsize=5)
ax1.set_yticklabels(ylabels, rotation=0, fontsize=5)
fig=plt.figure("Value Information 值图")
ax1 = fig.add_subplot(1, 1, 1)
ivlist=[v1,v2,v3,v4,v5,v6]
index =train_data.columns[4:].drop("Y") # x轴的标签
x = np.arange(len(index))+1
ax1=fig.add_subplot(1,1,1)
ax1.bar(x, ivlist, width=0.4)#生成柱状图
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=5)
ax1.set_ylabel('IV(Information Value)', fontsize=5)
#在柱状图上添加数字标签
for a, b in zip(x, ivlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=8)
# train_data['IS_LOCAL']=pd.Series(replace_woe(train_data['IS_LOCAL'],cut1,woe1))
train_data['EDU_LEVEL']=pd.Series(replace_woe(train_data['EDU_LEVEL'],cut2,woe2))
train_data['MARRY_STATUS']=pd.Series(replace_woe(train_data['MARRY_STATUS'],cut3,woe3))
# train_data['SALARY']=pd.Series(replace_woe(train_data['SALARY'],cut4,woe4))
train_data['HAS_FUND']=pd.Series(replace_woe(train_data['HAS_FUND'],cut5,woe5))
train_data['develpment_level']=pd.Series(replace_woe(train_data['develpment_level'],cut6,woe6))
# train_data.drop(["REPORT_ID","ID_CARD","LOAN_DATE"],axis=1)
train_data.to_csv("D:\sufe\A\WoEdata.csv",index=False)
plt.show()


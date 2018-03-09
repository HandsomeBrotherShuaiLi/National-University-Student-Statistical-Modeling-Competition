import pandas as pd
import numpy as np
import os

path='D:\sufe\A'
files=os.listdir(path)
train_data=pd.read_csv('D:\sufe\A\data_train_changed.csv')
data1=pd.read_csv('D:\sufe\A\contest_ext_crd_cd_ln.tsv',sep='\t')
data2=pd.read_csv('D:\sufe\A\contest_ext_crd_cd_ln_spl.tsv',sep='\t')
p=pd.merge(train_data,data1,on='REPORT_ID',how='left')
p=pd.merge(p,data2,on='REPORT_ID',how='left')
print(p.info())


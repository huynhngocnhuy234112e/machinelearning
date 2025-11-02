import pandas as pd
from numpy import nan as NA
data=pd.DataFrame([[1.,6.5,3],[1.,NA,NA],[NA,NA,NA],[NA,6.5,3.]])
print(data)
cleaned=data.dropna()
print(cleaned)
cleaned2=data.dropna(how='all')
print(cleaned)
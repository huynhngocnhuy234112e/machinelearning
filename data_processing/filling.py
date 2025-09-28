from numpy import nan as NA
import pandas as pd

from data_processing.filtering import cleaned

data=pd.DataFrame([[1.,6.5,3.],
                   [1.,NA,NA],
                   [NA,NA,NA],
                   [NA,6.5,3.]])
print(data)
print("-"*10)
cleaned=data.fillna(data.mean())
print(cleaned)
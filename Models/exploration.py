import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv('..//Data/train.csv')
print(df_train.columns)

print(df_train['SalePrice'].describe())

sns.distplot(df_train['SalePrice'])
plt.show()

print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
'''
Skewdness is the degree of distortion from the normal distribution.
Positive skewness is if the tail is to the right, negative on left.
    Low is between -0.5 and 0.5
    Medium is up to -1 and 1
    High is over that

Kurtosis tells you if there are a lot of outliers. It describes the tails i.e. the extremes.
Values:
    3 (Mesokurtic) is closes to normal distribution
    >3 (Leptokurtic) means more outliers. Longer and fatter tails, higher peak
    <3 (Platykurtic) means not many outliers. Shorter and thinner tails, lower peak
'''

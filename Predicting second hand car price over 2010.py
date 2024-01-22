import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pylab
from scipy.stats import shapiro
from sklearn.neighbors import LocalOutlierFactor
import missingno as msno;
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# Import data
data = pd.read_csv("car_price_prediction.csv",encoding='ISO-8859-1')
data.head()

#Select columns used
data = data.iloc[:,[1,5,7,8,9,11,12,13,17]]

#Change column name to lower-case and remove space
data.columns = data.columns.str.lower()
data.rename(columns={ 'gear box type' : 'gearbox'},inplace=True)
data.rename(columns={'prod. year':'year'}, inplace=True)

#Extract data before 2002
data = data[(data.year <=2010)]

# Checking if there is Missing values 
data.isnull()
data.isnull().values.any()
data.isnull().sum()

msno.matrix(data)

#Clean column
(data["engine volume"] =="2.0 Turbo").any()
split_str = data["engine volume"].str.split().str.get(0)
data["engine volume"] = np.array(split_str, dtype = "float64")

(data["engine volume"] == "2.0 Turbo").any()

#Checking average price based fuel type and gearbox
data.groupby('fuel type').price.mean()
data.groupby(["fuel type", "gearbox"])['price'].mean().unstack()
data.pivot_table('price' , index='gearbox',columns='fuel type')



dataInt = data.select_dtypes(include = ["int64", "float64"])
dataInt.head()

clf = LocalOutlierFactor(n_neighbors= 200, contamination=0.2)
clf.fit_predict(dataInt)

score = clf.negative_outlier_factor_
np.sort(score)[0:20]

esik_deger = np.sort(score)[100]
data2 = data[score > esik_deger]
plt.hist(data2["price"]);

len(data) - len(data2)


#Manage outliers
Q1 = data["price"].quantile(0.25)
Q3 = data["price"].quantile(0.75)
IQR = Q3 - Q1

alt_sinir = Q1 - 1.5*IQR
ust_sinir = Q3 + 1.5*IQR

data3 = data[~((data["price"] < alt_sinir) | (data["price"] > ust_sinir))]
plt.hist(data3["price"],);


data2.columns


#Change categorical to numeric for furture usages 
dummy_val = pd.get_dummies(data2[["leather interior","fuel type","gearbox","drive wheels","airbags"]])
dummy_val.head()

y = data2["price"]

x = data2.drop(["price","leather interior","fuel type","gearbox","drive wheels"], axis = 1).astype("float64")
x.head()

data3 = pd.concat([y, x, dummy_val[['leather interior_No', 'leather interior_Yes', 'fuel type_CNG',
       'fuel type_Diesel', 'fuel type_LPG', 'fuel type_Petrol',
       'fuel type_Plug-in Hybrid', 'gearbox_Automatic', 'gearbox_Manual',
       'gearbox_Tiptronic', 'gearbox_Variator', 'drive wheels_4x4',
       'drive wheels_Front', 'drive wheels_Rear']]], axis = 1)



from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

## Split train and test 
x = data3.drop("price", axis = 1)
x = x.astype(float)
y = data3["price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 21)

#Multi linear regression
lm = LinearRegression()
model = lm.fit(x_train, y_train)
model.score(x_train, y_train)

y_predict = model.predict(x_test)
r2_score(y_test, y_predict)

np.sqrt(mean_squared_error(y_test, model.predict(x_test)))

##GMB

gbm_model = GradientBoostingRegressor()
gbm_model.fit(x_train, y_train)

y_predict = gbm_model.predict(x_test)
r2_score(y_test, y_predict)

np.sqrt(mean_squared_error(y_test, gbm_model.predict(x_test)))









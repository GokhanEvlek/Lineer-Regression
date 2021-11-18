import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

veriler = pd.read_csv('tenis.csv')



from sklearn import preprocessing



veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)

ruzgar=veriler2.iloc[:,3:4].values
oynama=veriler2.iloc[:,-1:].values

le = preprocessing.LabelEncoder()
havadurumu=veriler.iloc[:,0:1].values
havadurumu[:,0] = le.fit_transform(veriler.iloc[:,0])#label encoder e göre eğitti ve dönüşüm yaptı
ohe = preprocessing.OneHotEncoder()#çok ihtimalli polinomal kategorik değişkeni 2 elemanlı arrayelere döndürdü
havadurumu = ohe.fit_transform(havadurumu).toarray()




oynama[:,-1] = le.fit_transform(veriler.iloc[:,-1])


ruzgar=veriler2.iloc[:,3:4].values

ruzgar[:,0]=le.fit_transform(veriler2.iloc[:,3])
print(ruzgar)


kalan=veriler.iloc[:,1:3].values





#dataframe oluşturma
sonuc3 = pd.DataFrame(data = kalan, index = range(14), columns = ['Sıcaklık',"Nem"])
sonuc = pd.DataFrame(data=havadurumu, index = range(14), columns = ['Bulutlu','Yağmurlu','Güneşli'])
ruzgarlılık=pd.DataFrame(data=ruzgar, index = range(14), columns = ['Rüzgar'])
sonuc2 = pd.DataFrame(data=oynama, index = range(14), columns = ['Oynanma Durumu'])



#dataframeleeri birleştirme
s=pd.concat([sonuc,sonuc3], axis=1)
s2=pd.concat([s,ruzgarlılık], axis=1)



from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s2,sonuc2,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_predict=regressor.predict(x_test)
print(y_predict)



import statsmodels.api as sm
#çıkan tablodaki P değerine baktık ve yüksek olandan 

#X = np.append(arr=np.ones((14,1)).astype(int),values=s2,axis=1)
X_l= s2.iloc[:,[0,1,2,3,4,5]].values
X_l= np.array(X_l,dtype=float)
model = sm.OLS(oynama,X_l).fit()

print(model.summary())

X_l= s2.iloc[:,[0,1,2,4,5]].values
X_l= np.array(X_l,dtype=float)
model = sm.OLS(oynama,X_l).fit()

print(model.summary())

X_l= s2.iloc[:,[0,1,2,3]].values
X_l= np.array(X_l,dtype=float)
model=sm.OLS(oynama,X_l).fit()
print(model.summary())

x_train, x_test,y_train,y_test = train_test_split(X_l,sonuc3,test_size=0.33, random_state=0)

regressor.fit(x_train,y_train)

y_predict=regressor.predict(x_test)
print(y_predict)
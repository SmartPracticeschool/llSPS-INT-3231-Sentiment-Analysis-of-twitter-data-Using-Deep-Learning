import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

dataset=pd.read_csv("twitter.csv",encoding="latin1")
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
ps=PorterStemmer()
data=[]
for i in range(0,1000):
    t1=dataset["SentimentText"][i]
    t1=re.sub('[^a-zA-Z]'," ",str(t1))
    t1=t1.lower()
    t1=t1.split()
    t1=[ps.stem(word) for word in t1 if not word in set(stopwords.words("english"))]
    t1=" ".join(t1)
    data.append(t1)
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=7000)
x=cv.fit_transform(data).toarray()  
with open('CountVectorizer','wb') as file:
    pickle.dump(cv,file)


y=dataset.iloc[:1000,1:2].values
  
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()

model.add(Dense(units= 2911 ,activation="relu" ,init="uniform"))

model.add(Dense(units= 5000 ,activation="relu" ,init="uniform"))
model.add(Dense(units= 3000 ,activation="relu" ,init="uniform"))
model.add(Dense(units= 1 ,activation="sigmoid" ,init="uniform"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=7,batch_size=32)

y_pred=model.predict(x_test)
y_pred
y_p=model.predict(cv.transform(["goodbye exams, HELLO ALCOHOL TONIGHT"]))
y_p
print(y_p>0.5) 
y_P=model.predict(cv.transform(["Very sad about Iran"])) 
y_P
print(y_P>0.5) 
model.save("twitter.h5")


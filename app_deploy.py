from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


df= pd.read_csv("spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
	
# Extract Feature With CountVectorizer
#Convert a collection of text documents to a matrix of token counts
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data

print(cv.get_feature_names())
print(X.toarray())
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


data = [" "]
vect = cv.transform(data).toarray()
my_prediction = clf.predict(vect)

if my_prediction == 1:
    print('The msg is Spam')
else:
    print('The msg is Ham')    
    
    
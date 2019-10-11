#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:14:36 2019

@author: shrinathpatel
"""
#from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle

    #df = pd.read_csv("SMSSpamCollection", sep='\t',names=["label", "message"])
    df= pd.read_csv("/Users/shrinathpatel/Downloads/NLP-Deployment-Heroku-master/spam.csv", encoding="latin-1")


    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    X=df['message']
    y=df['class']
    
    cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
    
    pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer())])
    
    
    pickle.dump(cv, open('tranform.pkl', 'wb'))
   
    
	#from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
	#Naive Bayes Classifier
	#from sklearn.naive_bayes import MultinomialNB

	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
    filename = 'nlp_model.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    
	#lternative Usage of Saved Model
	#joblib.dump(clf, 'NB_spam_model.pkl')
	#NB_spam_model = open('NB_spam_model.pkl','rb')
	#clf = joblib.load(NB_spam_model)
        
        
    
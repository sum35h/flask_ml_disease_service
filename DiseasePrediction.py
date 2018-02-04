import pickle
import numpy
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from lxml import html
import requests
import ctypes

words_file = "Symptoms.pkl"
diseases_file="Diseases.pkl"


 
    diseases_file_handler = open(diseases_file, "r")
    diseases = pickle.load(open(diseases_file, "rb"))
   
    diseases_file_handler.close()

    words_file_handler = open(words_file, "rb")
    word_data = pickle.load(words_file_handler)
   
    words_file_handler.close()

    

def predict():
    t = open("input.txt","r")

    lines = t.readlines()
    t.close()
    features_test=lines
   # print ("features_test: ",features_test)




    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
    features_train_transformed = vectorizer.fit_transform(word_data).toarray()
    features_test_transformed  = vectorizer.transform(features_test).toarray()




   
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()

    model.fit(features_train_transformed,diseases)
    pre = model.predict(features_test_transformed)
   
   

    print ("Disease is : ",pre[0])

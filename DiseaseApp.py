import pickle
import numpy
from flask import Flask,request, jsonify
from flask_restful import Resource,Api
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
app=Flask(__name__)
api=Api(app)

@app.route('/home', methods=['GET'])
def home():
     return ("Disease Predictor")

@app.route('/predict', methods=['GET'])
def perdict():
    #gets festures from user request payload

        #user_data =request.get_json(force=True)
        #input=user_data['data']
        input=request.args.get('data',default = '*', type = str)
        print(input)
        print("in")

        words_file = "Symptoms.pkl"
        diseases_file="Diseases.pkl"



        diseases_file_handler = open(diseases_file, "r")
        diseases = pickle.load(open(diseases_file, "rb"))

        diseases_file_handler.close()

        words_file_handler = open(words_file, "rb")
        word_data = pickle.load(words_file_handler)

        words_file_handler.close()

        features_test=[]
        features_test.append(input)
        print (features_test)
       # print(diseases)
       # print ("features_test: ",features_test)




        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
        features_train_transformed = vectorizer.fit_transform(word_data).toarray()
        features_test_transformed  = vectorizer.transform(features_test).toarray()

        model = LogisticRegression()

        model.fit(features_train_transformed,diseases)
        pre = model.predict(features_test_transformed).tolist()

        return jsonify({'Prediction': pre})
if __name__ == '__main__':
    app.run(port=5000)

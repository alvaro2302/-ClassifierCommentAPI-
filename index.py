#http://flask.palletsprojects.com/en/1.1.x/quickstart/#quickstart
from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle
import string
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

classifier_filepath = os.path.join("tree_v1.pkl")
classifier_file = open(classifier_filepath, "rb")
classifier = pickle.load(open(classifier_filepath, "rb"))
classifier_file.close()

def esPositivoONegativo(value):
    if(value == 0):
        return "el comentario es negativo"
    else:
        return "el comentario es positivo"


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json(force=True)

    data = pd.DataFrame([data])

    cleanWords=[]
    for sent in data['Review']:
        stop = set(stopwords.words('english') + list(string.punctuation))
        cleanWords.append(" ".join([str(i) for i in word_tokenize(sent.lower()) if i not in stop]))
    cv = CountVectorizer()
    cv_matrix = cv.fit_transform(cleanWords)
    oneHotCoding = np.sign(cv_matrix.toarray())
    with open('vocabularioProblema.txt','r')  as f:
        lines = f.readlines()

    content = [x.strip() for x in lines] 
    vocabulary = cv.get_feature_names()
    new_data = pd.DataFrame(oneHotCoding,columns=vocabulary)

    print(new_data.shape)
    x = new_data[vocabulary]
    for n in content:
        if (n not in x):
            x[str(n)]=0



    respuesta = classifier.predict(x)
    prediccion = esPositivoONegativo(respuesta[0])
    return prediccion

        
if __name__ == '__main__':
    app.run(port=8080, debug=True)
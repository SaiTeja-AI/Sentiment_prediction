from flask import Flask, render_template, request
from joblib import dump, load
import pandas as pd
import numpy as np

app = Flask(__name__)

lr_clf = open("lr_clf.joblib", "rb")
ml_model = load(lr_clf)
tfidf_vec = open("tfidf_vec.joblib", "rb")
vectorizer = load(tfidf_vec)            

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("I was here 1")
    if request.method == 'POST':
        print(request.form.get('review_text'))
        try:
            review_text = request.form['review_text']
            print(review_text)
            pred_args = [review_text]
            pred_args_arr = vectorizer.transform(pred_args)
            model_prediction = ml_model.predict(pred_args_arr.toarray())
            print(model_prediction)
            if model_prediction == 'pos':
                model_prediction_message = "Positive"
            else:
                model_prediction_message = "Negative"
            print(model_prediction_message)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction_message)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
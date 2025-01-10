from flask import Flask, render_template, redirect, url_for
import pandas as pd
import joblib
from forms import InputForm
import sklearn
import feature_engine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

model = joblib.load('model.joblib')

@app.route('/')
@app.route('/home')

def home():
    return render_template("home.html", title="Home")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = InputForm()
    if form.validate_on_submit():
        X_new = pd.DataFrame(dict(
            airline=[form.airline.data],
            date_of_journey=[form.date_of_journey.data.strftime("%Y-%m-%d")],
            source=[form.source.data],
            destination=[form.destination.data],
            dep_time=[form.dep_time.data.strftime("%H:%M:%S")],
            arrival_time=[form.arrival_time.data.strftime("%H:%M:%S")],
            duration=[form.duration.data],
            total_stops=[form.total_stops.data],
            additional_info=[form.additional_info.data],

        ))
        prediction = model.predict(X_new)[0]
        message = f"The predicted proce is  is {prediction:,.0f} INR!"
    else:
        message = "Please provide valid input details!"
    return render_template("predict.html", title="Predict", form = form, output = message)


if __name__ == '__main__':
    app.run(debug=True)
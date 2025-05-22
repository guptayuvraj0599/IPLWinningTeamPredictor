from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize app
app = Flask(__name__)

# Load the trained model
with open('pipe.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    import pandas as pd

    # Get form data
    batting_team = request.form['batting_team']
    bowling_team = request.form['bowling_team']
    city = request.form['city']
    target = int(request.form['target'])
    score = int(request.form['score'])
    overs = float(request.form['overs'])
    wickets = int(request.form['wickets'])

    # Feature engineering
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

    # Create a DataFrame instead of plain list
    input_df = pd.DataFrame([{
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'city': city,
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets': wickets_left,
        'total_runs_x': target,
        'crr': crr,
        'rrr': rrr
    }])

    # Predict
    prediction = model.predict_proba(input_df)[0]
    win_prob = round(prediction[1] * 100, 2)

    return render_template('index.html', prediction_text=f'Winning Probability: {win_prob}%')


if __name__ == "__main__":
    app.run(debug=True)

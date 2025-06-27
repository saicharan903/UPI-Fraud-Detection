from flask import Flask, request, render_template
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load model and model columns
model = pickle.load(open('logistic_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        input_dict = request.form.to_dict()

        # Convert types: float/int/bool
        for key in input_dict:
            val = input_dict[key]
            if val.lower() in ['true', 'false']:
                input_dict[key] = val.lower() == 'true'
            elif val.replace('.', '', 1).isdigit():
                input_dict[key] = float(val)
            else:
                # You can choose to log or clean other input types here
                input_dict[key] = val

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Align columns with model columns (missing = 0)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return render_template('index.html', pred=prediction, prob=round(probability, 4))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

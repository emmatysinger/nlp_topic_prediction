from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import src.explainability as explainability
import os

app = Flask(__name__)

# Load the model and transform pipeline
clf = joblib.load('model/model.joblib')
transform_pipeline = joblib.load('model/transform_pipeline.joblib')

latest_sample = None
latest_prediction = None
latest_explanation = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_sample():
    global latest_sample, latest_prediction, latest_explanation
    try:
        feature_names = transform_pipeline.named_steps['tfidf'].get_feature_names_out()
        data = request.form['text']
        latest_sample = data
        transformed_sample = transform_pipeline.transform([latest_sample])
        latest_prediction = clf.predict(transformed_sample)[0]
        latest_explanation = explainability.explain_prediction(transformed_sample, clf, feature_names)
        return redirect(url_for('result'))
    except Exception as e:
        print(f"Error during sample submission: {e}")
        return render_template('index.html', error=str(e))

@app.route('/result')
def result():
    if latest_sample is None:
        return redirect(url_for('index'))

    return render_template('result.html', sample=latest_sample, prediction=latest_prediction, explanation=latest_explanation)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port)

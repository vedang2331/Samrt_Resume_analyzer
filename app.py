from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained NLP model and CountVectorizer
model = pickle.load(open('model/nlp_model.pkl', 'rb'))
cv = pickle.load(open('model/cv.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        resume_text = request.form['resume']
        data = [resume_text]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

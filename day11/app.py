from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('maas.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    isim = request.form.get('isim')
    tecrube = float(request.form.get('tecrube'))
    yazili = float(request.form.get('yazili'))
    mulakat = float(request.form.get('mulakat'))
    
    tahmin = model.predict([[tecrube, yazili, mulakat]])
    
    # Assuming the model returns a 1D array, we access the first element.
    # If the model returns 2D (e.g. [[value]]), use tahmin[0][0]
    tahmin_text = f"Sayın {isim}, tahmin edilen maasiniz: ${tahmin[0]:.2f}"
    
    return render_template('index.html', tahmin=tahmin_text, isim=isim)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model from the .pkl file

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = 0 #model.predict(data)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import CryptoPricePredictor
import os

app = Flask(__name__)
CORS(app)

predictor = CryptoPricePredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        target_date = data.get('date')
        
        if not target_date:
            return jsonify({'error': 'Date is required'}), 400
            
        predictions = predictor.predict_for_date(target_date)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_for_date
import os

app = Flask(__name__)
CORS(app)

# predictor = CryptoPricePredictor()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("INCOMING DATA :- ", data)
        target_date = data.get('date')
        
        if not target_date:
            return jsonify({'error': 'Date is required'}), 400
            
        predictions = predict_for_date(target_date)
        print("OUTGOING DATA :- ", predictions)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)

from flask import Flask, request, jsonify
from src.pipelines.predict_pipeline import PredictPipeline,CustomData
app=Flask(__name__)
obj=CustomData()
obj.load_data()
@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    obj=CustomData()
    response = jsonify({
        'locations': obj.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
@app.route('/predict_home_price', methods=['GET', 'POST'])
def predict_home_price():
    pred_obj=PredictPipeline()
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    
    response = jsonify({
        'estimated_price': pred_obj.predict_price(location,total_sqft,bhk)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    app.run(debug=True)

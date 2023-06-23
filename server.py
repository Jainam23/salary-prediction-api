import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

#Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/api', methods=['POST'])
def predict():
    
    #Get data as {"years": number}
    data = request.get_json(force=True)
   
    #Use model to make and save prediction
    prediction = model.predict([[np.array(data['years'])]])
    output = prediction[0]

    return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
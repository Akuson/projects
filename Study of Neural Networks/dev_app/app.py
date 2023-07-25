from flask import Flask, render_template, request, jsonify
from build import build_model,predict_digit

app = Flask(__name__)

model = build_model()

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    pixel_data = data.get('pixelData')  # Assuming pixelData is an array of brightness values

    global model
    result = f"{predict_digit(model,pixel_data)}"
    
    return jsonify({'output': result})

if __name__ == '__main__':
    app.run(debug=True)

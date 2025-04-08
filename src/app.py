import io
import base64
from PIL import Image
from flask import Flask, render_template, request, url_for

from network import image_classify

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No file passed'
    
    file = request.files['image']

    if file:
        # Read the image
        img_data = file.read()
        img = Image.open(io.BytesIO(img_data))
        
        # Get the prediction
        prediction = image_classify(img)
        
        # Create a data URL for the image to display in the result page
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_data_url = f"data:image/jpeg;base64,{img_str}"
        
        # Render the result template
        return render_template('result.html', 
                              prediction=prediction,
                              image_path=image_data_url)

if __name__ == '__main__':
    app.run(debug=True)
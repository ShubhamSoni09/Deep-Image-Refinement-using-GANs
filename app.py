from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from srgan import process_with_srgan
from esrgan import process_with_esrgan
from PIL import Image
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("POST request received")
        print("Files in request:", request.files)
        print("Form data:", request.form)
        
        if 'image' not in request.files:
            print("No image file in request")
            return 'No file part'
        
        file = request.files['image']
        print("File received:", file.filename if file else "None")
        
        if file.filename == '':
            print("Empty filename")
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            try:
                print("Processing file:", file.filename)
                filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                print("File saved to:", filepath)

                # Process with GANs
                gan1_path = os.path.join(app.config['OUTPUT_FOLDER'], 'gan1_' + filename)
                gan2_path = os.path.join(app.config['OUTPUT_FOLDER'], 'gan2_' + filename)

                print("Processing with SRGAN...")
                process_with_srgan(filepath, gan1_path)
                print("Processing with ESRGAN...")
                process_with_esrgan(filepath, gan2_path)
                print("Processing complete!")

                return render_template('index.html',
                                       original=url_for('static', filename='uploads/' + filename),
                                       gan1=url_for('static', filename='outputs/' + 'gan1_' + filename),
                                       gan2=url_for('static', filename='outputs/' + 'gan2_' + filename))
            except Exception as e:
                print(f"Processing error: {e}")
                return f'Processing error: {str(e)}', 500
        else:
            print("File not allowed or invalid")
            return 'Invalid file type'
    
    print("GET request received")
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

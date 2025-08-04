from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from srgan import process_with_srgan
from esrgan import process_with_esrgan
from PIL import Image
import uuid
import gc
import psutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_image_if_needed(image_path, max_size=512):
    """Resize image if it's too large to prevent memory issues"""
    with Image.open(image_path) as img:
        width, height = img.size
        if width > max_size or height > max_size:
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            img.save(image_path)
            print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        return image_path

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

                # Resize image if too large to prevent memory issues
                filepath = resize_image_if_needed(filepath, max_size=256)  # Reduced to 256px

                # Process with GANs
                gan1_path = os.path.join(app.config['OUTPUT_FOLDER'], 'gan1_' + filename)
                gan2_path = os.path.join(app.config['OUTPUT_FOLDER'], 'gan2_' + filename)

                print("Processing with SRGAN...")
                process_with_srgan(filepath, gan1_path)
                gc.collect()  # Force garbage collection
                
                print("Processing with ESRGAN...")
                process_with_esrgan(filepath, gan2_path)
                gc.collect()  # Force garbage collection
                
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

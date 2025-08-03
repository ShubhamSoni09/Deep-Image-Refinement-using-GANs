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
        if 'image' not in request.files:
            return 'No file part'
        file = request.files['image']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process with GANs
            gan1_path = os.path.join(app.config['OUTPUT_FOLDER'], 'gan1_' + filename)
            gan2_path = os.path.join(app.config['OUTPUT_FOLDER'], 'gan2_' + filename)

            process_with_srgan(filepath, gan1_path)
            process_with_esrgan(filepath, gan2_path)

            return render_template('index.html',
                                   original=url_for('static', filename='uploads/' + filename),
                                   gan1=url_for('static', filename='outputs/' + 'gan1_' + filename),
                                   gan2=url_for('static', filename='outputs/' + 'gan2_' + filename))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

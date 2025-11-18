import os
from flask import Flask, request, render_template, jsonify
from deepface import DeepFace
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT = {"png","jpg","jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error':'no file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error':'no selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error':'file type not allowed'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Run DeepFace analyze for age and gender
        results = DeepFace.analyze(img_path=filepath, actions=['age','gender'])
        # delete uploaded file after processing to avoid persistent storage
        try:
            os.remove(filepath)
        except Exception:
            pass
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        # attempt to clean up file on error too
        try:
            os.remove(filepath)
        except Exception:
            pass
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

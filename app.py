from flask import Flask, request, redirect, url_for, send_file, render_template, flash
import pandas as pd
import os
import cv2
import numpy as np
import face_recognition
from PIL import Image
from werkzeug.utils import secure_filename
import zipfile
import shutil
import requests
import signal

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
PROCESSED_DATA_FOLDER = 'ProcessedData'
INVALID_FOLDER = os.path.join(PROCESSED_DATA_FOLDER, 'invalid')
VALID_FOLDER = os.path.join(PROCESSED_DATA_FOLDER, 'valid')
VALID_IMAGES_FOLDER = os.path.join(VALID_FOLDER, 'images')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['PROCESSED_DATA_FOLDER'] = PROCESSED_DATA_FOLDER
app.config['INVALID_FOLDER'] = INVALID_FOLDER
app.config['VALID_FOLDER'] = VALID_FOLDER
app.config['VALID_IMAGES_FOLDER'] = VALID_IMAGES_FOLDER

API_URL = 'https://api.picsart.io/tools/1.0/enhance/face'
API_KEY = "YOUR_API_KEY"

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return filename.lower().endswith('.zip')

def handle_sigterm(*args):
    zip_path = os.path.join(app.config['PROCESSED_FOLDER'], 'ProcessedData.zip')
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("ProcessedData.zip has been deleted upon shutdown.")
    os._exit(0)

def process_zip_file(zip_path, id_column, gender_column, nationality_column, age_column, birthdate_column):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(app.config['UPLOAD_FOLDER'])
        print("Extracted ZIP file.")

    base_path = os.path.join(app.config['UPLOAD_FOLDER'])
    excel_file = None
    images_folder = None

    for root, dirs, files in os.walk(base_path):
        for name in files:
            if name.endswith('.xlsx'):
                excel_file = os.path.join(root, name)
                print(f"Found Excel file: {excel_file}")
        for name in dirs:
            potential_images_folder = os.path.join(root, name)
            if any(fname.lower().endswith(('.png', '.jpg', '.jpeg')) for fname in os.listdir(potential_images_folder)):
                images_folder = potential_images_folder
                print(f"Found images folder: {images_folder}")

    if not excel_file or not images_folder:
        flash('لم يتم العثور على ملف Excel أو مجلد الصور.')
        print("Excel file or images folder not found.")
        return

    valid_df = pd.read_excel(excel_file)
    invalid_images_folder = os.path.join(app.config['INVALID_FOLDER'], 'images')
    if not os.path.exists(invalid_images_folder):
        os.makedirs(invalid_images_folder)
        print("Created invalid images folder.")

    duplicates = check_duplicates(valid_df, images_folder, invalid_images_folder)
    for duplicate_id, duplicate_list in duplicates.items():
        print(f"Duplicate set: {duplicate_id}, {', '.join(duplicate_list)}")

    issues_df, valid_df, images_to_move = process_excel(excel_file, images_folder, duplicates, id_column, gender_column, nationality_column, age_column, birthdate_column)
    
    invalid_data_file_path = os.path.join(app.config['INVALID_FOLDER'], 'invalid_data.xlsx')
    issues_df.to_excel(invalid_data_file_path, index=False)
    print("Invalid data file saved.")

    valid_file_path = os.path.join(app.config['VALID_FOLDER'], 'valid_data.xlsx')
    valid_df.to_excel(valid_file_path, index=False)
    print("Valid data file saved.")

    if not os.path.exists(app.config['VALID_IMAGES_FOLDER']):
        os.makedirs(app.config['VALID_IMAGES_FOLDER'])

    for index, row in valid_df.iterrows():
        img_id = row[id_column]
        img_path = os.path.join(images_folder, f"{img_id}.jpg")
        if os.path.exists(img_path):
            enhanced_path = os.path.join(app.config['VALID_IMAGES_FOLDER'], f"{img_id}.jpg")
            enhance_face_image(API_URL, API_KEY, img_path, enhanced_path)
            correct_orientation(enhanced_path, enhanced_path)
            print(f"Enhanced, corrected, and saved image {img_id}.")
        else:
            print(f"Image {img_path} not found in extracted files.")

    resize_images(app.config['VALID_IMAGES_FOLDER'], app.config['VALID_IMAGES_FOLDER'])

def process_excel(file_path, images_folder, duplicates, id_column, gender_column, nationality_column, age_column, birthdate_column):
    df = pd.read_excel(file_path)
    df[id_column] = df[id_column].apply(lambda x: str(int(x)) if not pd.isnull(x) else '')
    df['ID_Length_Valid'] = df[id_column].apply(lambda x: len(x) == 10)
    df['ID_Valid_For_Saudi'] = df.apply(lambda row: row[id_column].startswith('1') if row[nationality_column] == 'Saudi' else True, axis=1)
    
    issues = []
    valid_rows = []
    images_to_move = {}

    for index, row in df.iterrows():
        issue_list = []
        empty_columns = [col for col in df.columns if pd.isnull(row[col])]
        if not row['ID_Length_Valid']:
            issue_list.append('ID Length Invalid')
        if not row['ID_Valid_For_Saudi']:
            issue_list.append('Invalid Saudi ID')
        if empty_columns:
            issue_list.append(f'Missing Cell Values in: {", ".join(empty_columns)}')
        if row[id_column] in duplicates:
            duplicate_ids = ', '.join(duplicates[row[id_column]])
            issue_list.append(f'Duplicate Image with IDs: {duplicate_ids}')
        if issue_list:
            issues.append({
                id_column: row[id_column], 
                'Name': row.get('Name', 'N/A'), 
                gender_column: row[gender_column], 
                age_column: row.get(age_column, 'N/A'), 
                birthdate_column: row.get(birthdate_column, 'N/A'), 
                nationality_column: row[nationality_column], 
                'Issues': ', '.join(issue_list)
            })
        if not os.path.exists(os.path.join(images_folder, f"{row[id_column]}.jpg")):
            issue_list.append('Image file not found')
        if issue_list:
            images_to_move[row[id_column]] = ', '.join(issue_list)
        else:
            valid_rows.append(index)

    issues_df = pd.DataFrame(issues)
    valid_df = df.loc[valid_rows]  
    return issues_df, valid_df, images_to_move

def correct_orientation(image_path, output_path):
    image = cv2.imread(image_path)
    angles = [0, 90, 180, 270]
    landmarks = None

    for angle in angles:
        rotated_image = image if angle == 0 else cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        landmarks = detect_landmarks(rotated_image)
        
        if landmarks is not None:
            if angle != 0:
                image = rotated_image
            break

    if landmarks is not None:
        cv2.imwrite(output_path, image)
        print(f"Image saved at {output_path}.")
    else:
        print("No landmarks detected; image not saved.")

def detect_landmarks(image):
    # استخدم face_recognition لتحديد المعالم
    face_landmarks_list = face_recognition.face_landmarks(image)
    
    if face_landmarks_list:
        # إعادة المعالم كقائمة نقاط
        landmarks = np.array([[(point[0], point[1]) for feature in face_landmarks_list[0].values() for point in feature]])
        return landmarks
    return None

def check_duplicates(df, images_folder, invalid_folder):
    invalid_images_folder = os.path.join(invalid_folder, 'images')
    if not os.path.exists(invalid_images_folder):
        os.makedirs(invalid_images_folder)

    duplicates = {}
    for index, row in df.iterrows():
        id_1 = str(row['ID'])
        image_path_1 = os.path.join(images_folder, f"{id_1}.jpg")
        
        if not os.path.exists(image_path_1):
            continue

        image_1 = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
        if image_1 is None:
            continue
        image_1 = cv2.resize(image_1, (500, 500))

        for index_2, row_2 in df.loc[index+1:].iterrows():
            id_2 = str(row_2['ID'])
            image_path_2 = os.path.join(images_folder, f"{id_2}.jpg")

            if not os.path.exists(image_path_2):
                continue

            image_2 = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)
            if image_2 is None:
                continue
            image_2 = cv2.resize(image_2, (500, 500))
            difference = cv2.absdiff(image_1, image_2)
            similarity = 1 - (np.sum(difference) / (500 * 500 * 255))

            if similarity > 0.90:
                duplicates.setdefault(id_1, set()).add(id_2)
                duplicates.setdefault(id_2, set()).add(id_1)

    for id_1, duplicated_ids in duplicates.items():
        for id_2 in duplicated_ids:
            original_path = os.path.join(images_folder, f"{id_2}.jpg")
            issue_path = os.path.join(invalid_images_folder, f"{id_2}.jpg")
            if os.path.exists(original_path):
                shutil.move(original_path, issue_path)
                print(f"Moved {id_2} to invalid folder.")

    return duplicates

def resize_images(input_folder, output_folder, size=(640, 780)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)
            img = Image.open(file_path)
            img_resized = img.resize(size, Image.LANCZOS)
            img_resized.save(output_file_path)

def enhance_face_image(api_url, api_key, image_path, output_path):
    headers = {
        'X-Picsart-API-Key': api_key,
        'accept': 'application/json',
    }

    files = {
        'image': open(image_path, 'rb'),
        'format': (None, 'JPG'),
    }

    response = requests.post(api_url, headers=headers, files=files)

    if response.status_code == 200:
        response_data = response.json()
        image_url = response_data['data']['url']
        print(f"Enhanced image available at: {image_url}")
        download_image(image_url, output_path)
        print(f"Image {output_path} enhanced and saved.")
    else:
        print(f"Failed to enhance image {image_path}. Status code: {response.status_code}")
        print(f"Response: {response.text}")

def download_image(image_url, save_path):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(save_path, 'wb') as output_file:
            output_file.write(response.content)
        print(f"Image successfully downloaded and saved to {save_path}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    print("Handling a new request.")
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('لم يتم تحديد الملفات بشكل صحيح.')
            print("File part missing in request.")
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            flash('لم يتم تحديد الملفات أو نوع الملفات غير صحيح.')
            print("No file selected or invalid file type.")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            id_column = request.form.get('customerID')
            gender_column = request.form.get('customerGender')
            nationality_column = request.form.get('customerNationality')
            age_column = request.form.get('customerAge')
            birthdate_column = request.form.get('customerBirthdate')

            if not id_column or not gender_column or not nationality_column:
                flash('يجب ملء جميع الحقول المطلوبة.')
                return redirect(request.url)

            filename = secure_filename(file.filename)
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(zip_path)
            print(f"File {filename} saved successfully.")

            process_zip_file(zip_path, id_column, gender_column, nationality_column, age_column, birthdate_column)

            flash('تم معالجة الملفات بنجاح.')
            return redirect(url_for('download_processed'))

    return render_template('feu1.html')

@app.route('/download_processed', methods=['GET'])
def download_processed():
    zip_path = os.path.join(app.config['PROCESSED_FOLDER'], 'ProcessedData.zip')
    if not os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(app.config['PROCESSED_DATA_FOLDER']):
                for file in files:
                    zipf.write(os.path.join(root, file),
                               os.path.relpath(os.path.join(root, file),
                               os.path.join(app.config['PROCESSED_DATA_FOLDER'], '..')))
    return send_file(zip_path, as_attachment=True, download_name='ProcessedData.zip')

def delete_uploads_folder():
    folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print("Uploads folder has been deleted.")
    os.makedirs(folder)

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    try:
        for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, PROCESSED_DATA_FOLDER, INVALID_FOLDER, VALID_FOLDER, VALID_IMAGES_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        app.run(debug=True)
    finally:
        delete_uploads_folder()

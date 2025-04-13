import certifi
import gridfs
from pymongo import MongoClient
from bson.binary import Binary
import uuid
from flask import Flask, render_template, request, redirect, url_for, Response, flash
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
import os
import cv2
from datetime import datetime
from detection.face_matching import detect_faces, align_face, extract_features, match_face
from utils.configuration import load_yaml
import numpy as np
from flask import jsonify


# Connect to MongoDB Atlas
MONGO_URI = "mongodb+srv://rishipramodc:1891989@cluster0.787pn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())

# Load configuration
config_file_path = load_yaml("configs/database.yaml")

# MongoDB Atlas Setup
mongo_db = client["mydatabase"]
fs = gridfs.GridFS(mongo_db)
students_collection = mongo_db["students"]

# Load teacher password hash from config
TEACHER_PASSWORD_HASH = config_file_path["teacher"]["password_hash"]

# Flask app setup
app = Flask(__name__, template_folder="template")
app.secret_key = "123456"
UPLOAD_FOLDER = "examples"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

video = None
current_filename = None

def upload_database(filename, file_content):
    if fs.find_one({"filename": filename}):
        return True, f"Image {filename} already exists in the database"
    try:
        fs.put(file_content, filename=filename)
    except Exception as e:
        return True, f"Error uploading image: {str(e)}"
    return False, None

def match_with_database(img, database):
    global match
    faces = detect_faces(img)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.imwrite("static/recognized/recognized.png", img)
    for face in faces:
        try:
            aligned_face = align_face(img, face)
            embedding = extract_features(aligned_face)[0]["embedding"]
            match = match_face(embedding, database)
            return f"Match found: {match}" if match else "No match found"
        except Exception:
            return "No face detected"
    return "No face detected"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/add_info")
def add_info():
    return render_template("add_info.html")

@app.route("/teacher_login", methods=["GET", "POST"])
def teacher_login():
    if request.method == "POST":
        password = request.form.get("password")
        if check_password_hash(TEACHER_PASSWORD_HASH, password):
            return redirect(url_for("attendance"))
        flash("Incorrect password")
    return render_template("teacher_login.html")

@app.route("/attendance")
def attendance():
    students = list(students_collection.find())
    return render_template("attendance.html", students=students)

@app.route("/upload", methods=["POST"])
def upload():
    global current_filename
    if "file" not in request.files:
        flash("No file uploaded")
        return redirect(url_for("home"))
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("home"))
    if file and allowed_file(file.filename):
        filename = f"{uuid.uuid4()}.jpg"
        current_filename = filename
        file_content = file.read()
        error, message = upload_database(current_filename, file_content)
        if error:
            flash(message)
            return redirect(url_for("home"))
        flash("Image uploaded successfully!")
        return redirect(url_for("home"))
    flash("File upload failed")
    return redirect(url_for("home"))

def allowed_file(filename):
    return filename.rsplit(".", 1)[-1].lower() in {"png", "jpg", "jpeg", "gif"}

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f'<h1>File uploaded successfully</h1><img src="{url_for("static", filename="images/" + filename, v=timestamp)}" alt="Uploaded image">'

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/capture", methods=["POST"])
def capture():
    global current_filename, video
    if video is None:
        flash("Video stream not initialized")
        return redirect(url_for("home"))
    ret, frame = video.read()
    if not ret:
        flash("Failed to capture image")
        return redirect(url_for("home"))
    filename = f"{uuid.uuid4()}.jpg"
    current_filename = filename
    ret, buffer = cv2.imencode('.png', frame)
    if not ret:
        flash("Failed to encode captured image")
        return redirect(url_for("home"))
    file_content = buffer.tobytes()
    error, message = upload_database(current_filename, file_content)
    if error:
        flash(message)
        return redirect(url_for("home"))
    flash("Image captured and uploaded successfully!")
    return redirect(url_for("home"))

@app.route("/submit_info", methods=["POST"])
def submit_info():
    global current_filename
    name = request.form.get("name")
    email = request.form.get("email")
    userType = request.form.get("userType")
    classes = request.form.getlist("classes")
    password = request.form.get("password")
    file_grid = fs.find_one({"filename": current_filename})
    if file_grid is None:
        return "Uploaded file not found", 400
    file_bytes = file_grid.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = detect_faces(data)
    if faces is None or len(faces) == 0:
        return "No face detected", 400
    aligned_face = align_face(data, faces[0])
    embedding = extract_features(aligned_face)[0]["embedding"]
    student_doc = {
        "filename": current_filename,
        "name": name,
        "email": email,
        "userType": userType,
        "classes": {class_: 0 for class_ in classes},
        "password": generate_password_hash(password),
        "embeddings": embedding,
    }
    students_collection.insert_one(student_doc)
    return redirect(url_for("success", filename=current_filename))

@app.route("/recognize", methods=["GET", "POST"])
def recognize():
    if video is None:
        return jsonify({"error": "Video stream not initialized"}), 500

    ret, frame = video.read()
    if not ret:
        return jsonify({"error": "Failed to capture image"}), 500

    database = {}
    for student in students_collection.find():
        database[student.get("name")] = student.get("embeddings")

    detection = match_with_database(frame, database)

    if match:
        return jsonify({"redirect_url": url_for("success", filename=match + ".jpg")})
    else:
        return jsonify({"error": "No match found"}), 404

@app.route("/success/<filename>")
def success(filename):
    return render_template("success.html", filename=filename)

@app.route("/select_class")
def select_class():
    return render_template("select_class.html")


def gen_frames():
    global video
    video = cv2.VideoCapture(0)
    while True:
        success, frame = video.read()
        if not success:
            break
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

if __name__ == "__main__":
    app.run(debug=True)

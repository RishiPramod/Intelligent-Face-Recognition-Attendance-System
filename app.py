from flask import Flask, render_template, request, redirect, url_for, Response, flash
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
import os
import cv2
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db, storage
from detection.face_matching import detect_faces, align_face, extract_features, match_face
from utils.configuration import load_yaml

# Load configuration
config_file_path = load_yaml("configs/database.yaml")

# Firebase Configuration
firebase_config_path = os.path.join(os.getcwd(), "configs/serviceAccountKey.json")

# Check if service account file exists
if not os.path.exists(firebase_config_path):
    raise FileNotFoundError(f"Firebase service account file not found: {firebase_config_path}")

cred = credentials.Certificate(firebase_config_path)

# Initialize Firebase
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": config_file_path["firebase"]["databaseURL"],
        "storageBucket": config_file_path["firebase"]["storageBucket"],
    },
)

# Load teacher password hash from config
TEACHER_PASSWORD_HASH = config_file_path["teacher"]["password_hash"]

# Flask app setup
app = Flask(__name__, template_folder="template")
app.secret_key = "123456"

# Specify the directory to save uploaded images
UPLOAD_FOLDER = "examples"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global video capture object
video = None


def upload_database(filename):
    """
    Uploads a file to Firebase Storage if it does not already exist.
    """
    if storage.bucket().get_blob(filename):
        return True, f"<h1>{filename} already exists in the database</h1>"

    if not filename[:-4].isdigit():
        return True, f"<h1>Please ensure the filename {filename} is a number</h1>"

    local_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_filename(local_path)
    
    return False, None


def match_with_database(img, database):
    """
    Detects faces in an image and matches them against a database.
    """
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
    ref = db.reference("Students")
    students = {i: db.reference(f"Students/{i}").get() for i in range(1, len(ref.get()))}
    return render_template("attendance.html", students=students)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        ref = db.reference("Students")
        studentId = len(ref.get()) if ref.get() else 1
        filename = f"{studentId}.png"

        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        val, err = upload_database(filename)
        if val:
            return err

        return redirect(url_for("add_info"))

    return "File upload failed", 400


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
    global filename
    if video is None:
        return "Video stream not initialized", 500

    ret, frame = video.read()
    if not ret:
        return "Failed to capture image", 500

    ref = db.reference("Students")
    studentId = len(ref.get()) if ref.get() else 1
    filename = f"{studentId}.png"

    cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], filename), frame)

    val, err = upload_database(filename)
    if val:
        return err

    return redirect(url_for("add_info"))


@app.route("/submit_info", methods=["POST"])
def submit_info():
    name, email, userType, classes, password = (
        request.form.get("name"),
        request.form.get("email"),
        request.form.get("userType"),
        request.form.getlist("classes"),
        request.form.get("password"),
    )

    studentId, _ = os.path.splitext(filename)
    fileName = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    data = cv2.imread(fileName)
    faces = detect_faces(data)

    if not faces:
        return "No face detected", 400

    aligned_face = align_face(data, faces[0])
    embedding = extract_features(aligned_face)[0]["embedding"]

    ref = db.reference("Students")
    ref.child(studentId).set(
        {
            "name": name,
            "email": email,
            "userType": userType,
            "classes": {class_: 0 for class_ in classes},
            "password": password,
            "embeddings": embedding,
        }
    )

    return redirect(url_for("success", filename=filename))


@app.route("/recognize", methods=["GET", "POST"])
def recognize():
    if video is None:
        return "Video stream not initialized", 500

    ret, frame = video.read()
    if not ret:
        return "Failed to capture image", 500

    ref = db.reference("Students")
    database = {db.reference(f"Students/{i}").get()["name"]: db.reference(f"Students/{i}").get()["embeddings"] for i in range(1, len(ref.get()))}

    detection = match_with_database(frame, database)

    return redirect(url_for("select_class"))


def gen_frames():
    global video
    video = cv2.VideoCapture(0)
    while True:
        success, frame = video.read()
        if not success:
            break
        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


if __name__ == "__main__":
    app.run(debug=True)

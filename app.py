from flask import Flask, render_template, request, redirect, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Patient Model
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    blood_pressure = db.Column(db.Integer)
    albumin = db.Column(db.Integer)
    serum_creatinine = db.Column(db.Float)
    hemoglobin = db.Column(db.Float)
    blood_urea = db.Column(db.Float)
    sodium = db.Column(db.Float)
    potassium = db.Column(db.Float)
    specific_gravity = db.Column(db.Float)
    model_used = db.Column(db.String(50))
    prediction = db.Column(db.String(50))

with app.app_context():
    db.create_all()

# Load models
with open("logistic_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

@app.route("/")
def home():
    return redirect("/login") if "user" not in session else redirect("/index")

@app.route("/index")
def index():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/login")

    features = [float(request.form[k]) for k in [
        "age", "blood_pressure", "albumin", "serum_creatinine",
        "hemoglobin", "blood_urea", "sodium", "potassium", "specific_gravity"
    ]]
    model_choice = request.form["model"]

    model = logistic_model if model_choice == "logistic" else svm_model
    prediction_raw = model.predict([features])[0]

    prediction = "CKD Detected" if prediction_raw.lower() in ["yes", "ckd detected", "ckd"] else "No CKD"

    patient = Patient(
        age=features[0], blood_pressure=features[1], albumin=features[2],
        serum_creatinine=features[3], hemoglobin=features[4],
        blood_urea=features[5], sodium=features[6], potassium=features[7],
        specific_gravity=features[8], model_used=model_choice, prediction=prediction
    )
    db.session.add(patient)
    db.session.commit()

    return render_template("result.html", prediction=prediction)

@app.route("/patients")
def patients():
    if "user" not in session:
        return redirect("/login")
    all_patients = Patient.query.all()
    return render_template("patient_records.html", patients=all_patients)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session["user"] = username
            return redirect("/index")
        else:
            flash("Invalid username or password!", "danger")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if User.query.filter_by(username=username).first():
            flash("Username already taken. Try another one.", "warning")
            return redirect("/signup")

        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Signup successful. Please login.", "success")
        return redirect("/login")
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "info")
    return redirect("/login")

if __name__ == "__main__":
    app.run(debug=True)

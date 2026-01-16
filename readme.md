# Face Recognition Based Attendance System

This is a **Flask-based web application** for marking attendance using **face recognition**.  
User registration is handled offline via a CLI tool that captures multiple images per person for high-quality embeddings.

---

## Features

- Live webcam preview and capture in browser
- Face recognition using **DeepFace (Facenet512)** and **MTCNN**
- Attendance confirmation before marking
- Attendance saved in **CSV** with timestamp
- Attendance summary page:
  - Filter by last day, week, month, year
  - Custom date-time range
  - Collapsible view to see per-user timestamps

---

## Project Structure

```

face-attendance/
│
├─ app.py                  # Flask app
├─ utils/
│   └─ recognizer.py       # Recognition logic
├─ embeddings.npy          # Face embeddings (generated via CLI registration)
├─ templates/
│   ├─ index.html
│   ├─ capture.html
│   ├─ result.html
│   └─ attendance_summary.html
├─ static/
│   └─ style.css
├─ attendance.csv          # Generated attendance records
├─ requirements.txt
└─ README.md

````

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/sachinkashi21/FaceAttendance.git
cd faceRec
````

2. **Create and activate virtual environment (recommended)**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / Mac
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

*Optional dependencies*

```bash
pip install tensorflow pillow
```

4. **Register users (CLI)**

```bash
python src/register_user.py
```

* This will capture multiple images per person
* Embeddings are stored in `embeddings.npy`

5. **Run the Flask app**

```bash
python app.py
```

6. **Open in browser**

```
http://127.0.0.1:5000
```

* Go to **Mark Attendance** to capture and recognize faces
* Go to **View Attendance** to see summary and filters

---

## Notes

* Registration must be done offline before using the web app
* Attendance CSV grows with time; you can replace with a database for production
* Make sure your webcam is accessible and allowed by the browser

---

## Dependencies

* Flask
* OpenCV (`opencv-python`)
* NumPy
* DeepFace
* scikit-learn
* MTCNN

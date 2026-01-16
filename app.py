from flask import Flask, render_template, redirect, request
import cv2
import csv
import base64
import numpy as np
from datetime import datetime

from utils.recognizer import recognize_and_draw
from datetime import datetime, timedelta
from collections import defaultdict


app = Flask(__name__)

# -------------------------
# Helpers
# -------------------------
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def frame_to_base64(frame):
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()

def read_attendance():
    records = []

    try:
        with open("attendance.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                name, ts = row
                timestamp = datetime.fromisoformat(ts)
                records.append((name, timestamp))
    except FileNotFoundError:
        pass

    return records

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recognize", methods=["GET"])
def recognize_page():
    return render_template("capture.html")

@app.route("/recognize", methods=["POST"])
def recognize_face():
    try:
        image_data = request.form["image"]
        image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        annotated, name = recognize_and_draw(frame)

        _, buf = cv2.imencode(".jpg", annotated)
        result_image = base64.b64encode(buf).decode()

        return render_template(
            "result.html",
            image=result_image,
            name=name
        )

    except Exception:
        return render_template(
            "result.html",
            error="Face not detected. Try again."
        )

@app.route("/confirm", methods=["POST"])
def confirm():
    name = request.form.get("name")

    if not name or name == "Unknown":
        return redirect("/recognize")

    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, datetime.now()])

    return redirect("/recognize")

from collections import defaultdict

@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    records = read_attendance()  # [(name, datetime), ...]
    filtered = records

    if request.method == "POST":
        filter_type = request.form.get("filter")
        now = datetime.now()

        if filter_type == "custom":
            start = request.form.get("start")
            end = request.form.get("end")
            if start and end:
                start_dt = datetime.fromisoformat(start)
                end_dt = datetime.fromisoformat(end)
                filtered = [r for r in records if start_dt <= r[1] <= end_dt]
        elif filter_type == "day":
            filtered = [r for r in records if r[1] >= now - timedelta(days=1)]
        elif filter_type == "week":
            filtered = [r for r in records if r[1] >= now - timedelta(weeks=1)]
        elif filter_type == "month":
            filtered = [r for r in records if r[1] >= now - timedelta(days=30)]
        elif filter_type == "year":
            filtered = [r for r in records if r[1] >= now - timedelta(days=365)]

    # Group by name
    summary = defaultdict(list)
    for name, ts in filtered:
        summary[name].append(ts)

    # Sort by name
    summary = dict(sorted(summary.items()))

    return render_template("attendance.html", summary=summary)


if __name__ == "__main__":
    app.run(debug=True)

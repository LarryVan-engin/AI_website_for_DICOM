import os
import pydicom
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Để hiển thị thông báo flash
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"dcm"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("Vui lòng chọn một file để tải lên!", "error")
            return redirect(url_for("upload_file"))

        file = request.files["file"]

        if file.filename == "":
            flash("Không có file nào được chọn!", "error")
            return redirect(url_for("upload_file"))

        if not allowed_file(file.filename):
            flash("File không hợp lệ! Vui lòng chọn file DICOM (*.dcm)", "error")
            return redirect(url_for("upload_file"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        dicom_data = pydicom.dcmread(filepath)
        patient_name = dicom_data.PatientName if "PatientName" in dicom_data else "Không xác định"

        flash(f"File tải lên thành công! Tên bệnh nhân: {patient_name}", "success")
        return redirect(url_for("upload_file"))

    return render_template("ai_segment_dicom.html")

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

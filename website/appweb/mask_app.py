import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from mask_automation import read_dicom, create_mask, save_mask

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
MASK_FOLDER = "masks"
ALLOWED_EXTENSIONS = {"dcm"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MASK_FOLDER"] = MASK_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    mask_url = None
    if request.method == "POST":
        if "file" not in request.files:
            return "Vui lòng chọn file!"

        file = request.files["file"]
        if file.filename == "":
            return "Không có file nào được chọn!"

        if not allowed_file(file.filename):
            return "File không hợp lệ! Vui lòng chọn file DICOM (*.dcm)"

        filename = secure_filename(file.filename)
        dicom_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(dicom_path)

        # Xử lý file DICOM
        dicom_image = read_dicom(dicom_path)
        mask = create_mask(dicom_image, threshold=128)

        # Lưu mask thành file
        mask_filename = f"{os.path.splitext(filename)[0]}_mask.png"
        mask_path = os.path.join(app.config["MASK_FOLDER"], mask_filename)
        save_mask(mask_path, mask)

        mask_url = f"/masks/{mask_filename}"

    return render_template("upload.html", mask_url=mask_url)

if __name__ == "__main__":
    app.run(debug=True)

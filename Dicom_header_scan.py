import pydicom
from pydicom.errors import InvalidDicomError
import os
import tkinter as tk
from tkinter import messagebox

def read_dicom_header(file_path):
    try:
        # Đọc file DICOM
        dicom_file = pydicom.dcmread(file_path)
        
        # Các trường thông tin cần thiết
        fields = {
            "Patient Name": (0x0010, 0x0010),
            "Patient ID": (0x0010, 0x0020),
            "Study Date": (0x0008, 0x0020),
            "Modality": (0x0008, 0x0060),
            "Study Description": (0x0008, 0x1030),
            "Series Description": (0x0008, 0x103E),
            "Institution Name": (0x0008, 0x0080),
            "Manufacturer": (0x0008, 0x0070)
        }
        
        # Thu thập thông tin từ các trường
        info = {}
        for field_name, tag in fields.items():
            try:
                value = dicom_file[tag].value
                # Chuyển đổi giá trị thành chuỗi, xử lý trường hợp None hoặc đặc biệt
                if value is None:
                    info[field_name] = "N/A"
                elif isinstance(value, pydicom.multival.MultiValue):
                    info[field_name] = ", ".join(str(v) for v in value)
                else:
                    info[field_name] = str(value)
            except KeyError:
                info[field_name] = "N/A"
        
        return info
    
    except InvalidDicomError:
        messagebox.showerror("Lỗi", f"File {file_path} không phải là file DICOM hợp lệ.")
        return None
    except FileNotFoundError:
        messagebox.showerror("Lỗi", f"Không tìm thấy file: {file_path}")
        return None
    except Exception as e:
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {str(e)}")
        return None

def display_dicom_info(info):
    # Tạo cửa sổ Tkinter
    window = tk.Tk()
    window.title("DICOM Header Information")
    window.geometry("400x300")
    
    # Tiêu đề
    tk.Label(window, text="DICOM Header Information", font=("Arial", 14, "bold")).pack(pady=10)
    
    # Hiển thị thông tin
    for field, value in info.items():
        frame = tk.Frame(window)
        frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(frame, text=f"{field}:", font=("Arial", 10), width=20, anchor="w").pack(side="left")
        tk.Label(frame, text=value, font=("Arial", 10), wraplength=200, anchor="w").pack(side="left")
    
    # Nút đóng cửa sổ
    tk.Button(window, text="Đóng", command=window.destroy).pack(pady=20)
    
    window.mainloop()

def main():
    # Đường dẫn đến file DICOM (thay đổi theo file của bạn)
    dicom_file_path = "train\CT_dicom\dicom_dir\ID_0001_AGE_0069_CONTRAST_1_CT.dcm"
    
    if os.path.exists(dicom_file_path):
        info = read_dicom_header(dicom_file_path)
        if info:
            display_dicom_info(info)
    else:
        messagebox.showerror("Lỗi", f"File {dicom_file_path} không tồn tại. Vui lòng cung cấp đường dẫn hợp lệ.")

if __name__ == "__main__":
    main()
from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return render(request, 'pages/home.html')

def ai_create_mask(request):
    context={}
    return render(request, 'pages/ai_create_mask.html',context)

def ai_segment_dicom(request):
    context={}
    return render(request, 'pages/ai_segment_dicom.html',context)

def ai_analyze_header(request):
    context={}
    return render(request, 'pages/ai_analyze_header.html',context)

def upload_file(request):
    if request.method == "POST":
        if "file" not in request.FILES:
            return HttpResponse("Vui lòng chọn file!")  # Kiểm tra nếu không có file được gửi lên

        file = request.FILES["file"]
        # Xử lý file DICOM tại đây...

        return HttpResponse("File đã được xử lý thành công!")

    return render(request, "ai_create_mask.html")
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='home'),
    path('ai_create_mask/', views.ai_create_mask, name='ai_create_mask'),
    path('ai_segment_dicom/', views.ai_segment_dicom, name='ai_segment_dicom'),
    path('ai_analyze_header/', views.ai_analyze_header, name= 'ai_analyze_header'),
    path("upload/", views.upload_file, name="upload_file"),
]

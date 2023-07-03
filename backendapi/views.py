from django.shortcuts import render
from django.shortcuts import render
from rest_framework.decorators import api_view
# Create your views here.
from rest_framework import request, status, viewsets
from .models import PatientsList, DoctorList,AssignDoctortoPatient,DICOMSeriesload, PatientReport, UserAuthority
from .Serializers import PatientsListSerializer, DoctorListSerializer,AssignDoctortoPatientSerializer,FolderSerializer,DICOMSeriesloadSerializer,PatientReportSerializer,CustomTokenObtainPairSerializer,UserAuthnSerializer,ImageSerializer
from rest_framework import status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from django.contrib.auth.models import User 
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import  AllowAny, IsAuthenticated, IsAdminUser, IsAuthenticatedOrReadOnly
from rest_framework.authtoken.models import Token
from django.http import JsonResponse
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser
from rest_framework import generics

from pydicom import dcmread
from rest_framework.parsers import MultiPartParser, FormParser
import pydicom
from django.http import HttpResponse
from django.http import HttpResponseBadRequest, HttpResponseNotAllowed
import os   
import logging
import vtk
from django.core.files.storage import FileSystemStorage

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import User
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import get_user_model, login, logout
from rest_framework.authentication import SessionAuthentication
from rest_framework.views import APIView
from rest_framework.response import Response

from rest_framework import permissions, status

from django.core.exceptions import ValidationError
from django.contrib.auth import get_user_model

from rest_framework import generics, permissions
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.permissions import AllowAny

from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin

from pydicom import dcmread
from rest_framework.parsers import MultiPartParser, FormParser
import pydicom
from django.http import HttpResponse
from django.http import HttpResponseBadRequest, HttpResponseNotAllowed
import os   
import logging
import vtk
from django.core.files.storage import FileSystemStorage
from rest_framework.test import APIClient
import os
from keras import backend as Kb
import pydicom
import numpy as np
import keras
import tensorflow as tf
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from tensorflow.keras.models import load_model
import json


#This module provides a portable way of using operating system dependent functionality.
#If you just want to read or write a file see open()
import os
import cv2
from PIL import Image
import imageio
import json



from skimage import io
#import image

#The Extensible Markup Language (XML) is a markup language much like HTML.
#It is a portable and it is useful for handling small to medium amounts of data without using any SQL database
#XML is a tree like hierarchical data format.
#The tree is a hierarchical structure of elements starting with root followed by other elements. 
#Each element is created by using Element() function of this module
import xml.etree.ElementTree as ET

#https://github.com/SimpleITK/TUTORIAL/blob/master/05_basic_registration.ipynb
import SimpleITK as sitk
import statistics

import numpy as np
#import pydicom
import PIL 
from PIL import Image
import pickle
from matplotlib import pyplot as plt
from skimage import draw
import random
from shutil import copyfile
import csv
import pandas as pd



class viewsets_PatientsList(viewsets.ModelViewSet):
    queryset = PatientsList.objects.all()
    serializer_class = PatientsListSerializer
    filter_backend = [filters.SearchFilter]
    search_fields = ['Patient_IDs','Patient_Name']
    #authentication_classes = [TokenAuthentication]
    # permission_classes = [IsAuthenticated]

class viewsets_DICOMSeriesload(viewsets.ModelViewSet):
    queryset = DICOMSeriesload.objects.all()
    serializer_class = DICOMSeriesloadSerializer
    
    
class viewsets_DoctorsList(viewsets.ModelViewSet):
    queryset = DoctorList.objects.all()
    serializer_class = DoctorListSerializer
    #filter_backend = [filters.SearchFilter]
    #search_fields = ['Doctor_Ids','Doctor_Name']
    #authentication_classes = [TokenAuthentication]
    # permission_classes = [IsAuthenticated]

class viewsets_PatientReport(viewsets.ModelViewSet):
    queryset = PatientReport.objects.all()
    serializer_class = PatientReportSerializer

class viewsets_AssignDoctor(viewsets.ModelViewSet):
    queryset = AssignDoctortoPatient.objects.all()
    serializer_class = AssignDoctortoPatientSerializer
    #authentication_classes = [TokenAuthentication]
    # permission_classes = [IsAuthenticated]

class viewsets_UserAuthn(viewsets.ModelViewSet):
    queryset = UserAuthority.objects.all()
    serializer_class = UserAuthnSerializer

# Find doctor
@api_view(['GET'])
def find_doctor(request):
    doctors= DoctorList.objects.filter(
        Doctor_Ids = request.data['Doctors_IDs'],
        Doctor_Name  = request.data['Doctors_Name'],
    )
    serializer = DoctorListSerializer(doctors, many= True)
    return Response(serializer.data)


# Find Patient
@api_view(['GET'])
def find_patient(request):
    patients=PatientsList.objects.filter(
        Patient_IDs = request.data['Patient_IDs'],
        Patient_Name  = request.data['Patient_Name'],
    )
    serializer =PatientsListSerializer(patients, many= True)
    return Response(serializer.data)


#9 create new assign(Problem)
@api_view(['POST'])
def new_assign(request):

    patients_ids_For_Doctors =PatientsList.objects.get(
        Patient_IDs = request.data['Patient_IDs'],
        Patient_Name = request.data['Patient_Name'],
    ),
    
    Doctors_ids_For_Patients=DoctorList.objects.get(
        Doctor_Ids = request.data['Doctors_IDs'],
        Doctor_Name = request.data['Doctors_Name'],
    )
    '''
    doctor = DoctorsList()
    doctor.Doctor_Ids = request.data['Doctor_Ids']
    doctor.Doctor_Name = request.data['Doctor_Name']
    doctor.Doctor_Phone = request.data['Doctor_Phone']
    doctor.Doctor_Email = request.data['Doctor_Name']
    doctor.Doctor_Address = request.data['Doctor_Name']
    doctor.Doctor_Position = request.data['Doctor_Name']
    doctor.save()
'''
    assign = AssignDoctortoPatient()
    assign.patients_ids_For_Doctors = patients_ids_For_Doctors
    assign.Doctors_ids_For_Patients = Doctors_ids_For_Patients
    assign.save()
    serializer =AssignDoctortoPatientSerializer(assign, many= True)
    
    return Response(serializer.data,status=status.HTTP_201_CREATED)

       

@api_view(['GET'])
def series_urls(request):
    series_urls = PatientsList.objects.values_list('Series_URL', flat=True)
    data = {
        'seriesUrls': list(series_urls)
    }
    return JsonResponse(data)



            
#################################################################################
def get_relative_path(file_path):
    media_root = settings.MEDIA_ROOT
    relative_path = os.path.relpath(file_path, media_root)
    return relative_path.replace('\\', '/')  # Replace backslashes with forward slashes          
          

def get_dicom_image_urls(folder_url):
    
    #folder_url_edit=get_relative_path(folder_url)
    #print(folder_url_edit)

    
    
    #dicom_files = [f for f in os.listdir(folder_url) if os.path.isfile(os.path.join(folder_url, f)) and f.endswith('.dcm')]
    dicom_files = [os.path.join(folder_url, file_name) for file_name in os.listdir(folder_url) if file_name.endswith('.dcm')]
    image_urls = []
    for file in dicom_files:
        try:
            ds = pydicom.dcmread(os.path.join(folder_url, file))
            print(ds.file_meta.TransferSyntaxUID)
            print(ds.file_meta.TransferSyntaxUID.is_little_endian)
            print(ds.file_meta.TransferSyntaxUID.is_implicit_VR)
            print(ds.pixel_array[250]) 
            print(ds.pixel_array.shape) 
            print(ds.SpecificCharacterSet)
            #image_url = ds.file_meta.TransferSyntaxUID.is_little_endian and 'http://127.0.0.1:8000/media/1.dcm'
            image_url ="http://127.0.0.1:8000/media/"+get_relative_path(file)
            
            #image_url = request.build_absolute_uri(file)

            image_urls.append(image_url)
            #print(pydicom.dcmread(image_urls[1]).pixel_array[250]) 
            print(image_urls) 
         
            
        except Exception as e:
            # Handle the exception here, e.g. by logging it
            pass
    return image_urls

def get_dicom_image_VTK_File(folder_url, VTK_File=None):
    vtk_urls = []
    folder_name_edit=get_relative_path(folder_url)
    print("llllllllll")
    print(folder_name_edit)
    storage = FileSystemStorage(location=settings.MEDIA_ROOT)
    
    print(storage)
    file_name = f"output_{folder_name_edit}.vti"
    print(file_name)
    file_path = storage.path(file_name)
    print(file_path)
    file_url = storage.url(file_name)
    print(file_url)
    if VTK_File and storage.exists(VTK_File.name):
        # If VTK_File is provided and the file already exists, return the file path
        print("kkkkkkkkkkk")
        print(file_path)
        return file_path
    else:
        # Read DICOM images from directory
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(folder_url)
        reader.Update()

        # Convert images to VTK polydata
        #geometryFilter = vtk.vtkImageDataGeometryFilter()
        #geometryFilter.SetInputConnection(reader.GetOutputPort())
        #geometryFilter.Update()
        vtkPolyData = reader.GetOutput()

        # Save VTK file
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(file_path)
        writer.SetInputData(vtkPolyData)
        writer.Write()
        vtk_url= "http://127.0.0.1:8000"+file_url
        print (vtk_url)
        vtk_urls.append(vtk_url)
            #print(pydicom.dcmread(image_urls[1]).pixel_array[250]) 
        print(vtk_urls) 
         

        # Save file path in model field
       # if VTK_File:
         #   VTK_File.name = file_name
         #   VTK_File.save()
       # else:
            # If VTK_File is not provided, create a new object and save it to the database
           # model_instance = DICOMSeriesload(VTK_File=file_name)
           # model_instance.save()

        return vtk_urls

class DicomFolderView(APIView):
    def post(self, request):
        patient_id = request.data.get('patient_id')
        if not patient_id:
            return Response({'error': 'patient_id parameter is missing'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            patient = PatientsList.objects.get(Patient_IDs=patient_id)
        except PatientsList.DoesNotExist:
            return Response({'error': 'patient with given ID not found'}, status=status.HTTP_404_NOT_FOUND)
        folder_url = patient.Series_URL
        image_urls = get_dicom_image_urls(folder_url)
        vtk_urls = get_dicom_image_VTK_File(folder_url)
        
        # Get the DICOMSeriesload object for the patient
        series_load, created = DICOMSeriesload.objects.get_or_create(Patient_id_Series_Load=patient )
        series_load.Series_array = image_urls
        series_load.VTK_File = vtk_urls
        series_load.save()
            # Get the Patient and Series_Load objects
        #patient = PatientsList.objects.get(Patient_IDs=1215)
            #series_load = DICOMSeriesload.objects.get(id=1)
        #PatientsList_instance = DICOMSeriesload(Patient_id_Series_Load=patient,  Series_array=image_urls , VTK_File= vtk_urls)
        #PatientsList_instance.save()
        
        return Response({'image_urls': image_urls, 'vtk_file': vtk_urls}, status=status.HTTP_200_OK) 
    
################################################################################################################   
class PatientReportView(APIView):
    def post(self, request):
        patient_id = request.data.get('patient_id')
        if not patient_id:
            return Response({'error': 'patient_id parameter is missing'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            patient = PatientsList.objects.get(Patient_IDs=patient_id)
        except PatientsList.DoesNotExist:
            return Response({'error': 'patient with given ID not found'}, status=status.HTTP_404_NOT_FOUND)
       
        DoctorName = patient.Doctor_Name
        TechnacianName = patient.Technacian_Name 
        TechnacianID = patient.Technacian_ID
        PatientName = patient.Patient_Name
        PatientPhone = patient.Patient_Phone
        
       
        #image_urls = get_dicom_image_urls(folder_url)
        #vtk_urls = get_dicom_image_VTK_File(folder_url)
        
        # Get the DICOMSeriesload object for the patient
        Report_load, created = PatientReport.objects.get_or_create(reports_patientlist =patient )
        Report_load.Doctor_Name =DoctorName
        Report_load.Technacian_Name =TechnacianName
        Report_load.Technacian_ID =TechnacianID 
        Report_load.Patient_Name =PatientName
        Report_load.Patient_Phone =PatientPhone
        Report_load.save()
            # Get the Patient and Series_Load objects
        #patient = PatientsList.objects.get(Patient_IDs=1215)
            #series_load = DICOMSeriesload.objects.get(id=1)
        #PatientsList_instance = DICOMSeriesload(Patient_id_Series_Load=patient,  Series_array=image_urls , VTK_File= vtk_urls)
        #PatientsList_instance.save()
        
        return Response({'Doctor_Name': DoctorName, 'Technacian_Name': TechnacianName}, status=status.HTTP_200_OK) 
    
##############################################################################
class IsAdminUser(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.is_superuser

class IsDoctorUser(permissions.BasePermission):
    def has_permission(self, request, view):
        return hasattr(request.user, 'doctor')

class PatientListView(generics.ListAPIView):
    serializer_class = PatientsListSerializer
    permission_classes = [permissions.IsAuthenticated & IsDoctorUser]

    def get_queryset(self):
       Technacian_Name = self.request.user.doctor
       return PatientsList.objects.filter(doctor=Technacian_Name)

class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer
    
###############################################################################

      
class UserViewSettest(APIView):
    def post(self, request):
        doctor_id = request.data.get('doctor_id')
        if not doctor_id:
            return Response({'error': 'doctor_id parameter is missing'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            doctor = DoctorList.objects.get(Doctors_IDs=doctor_id)
        except DoctorList.DoesNotExist:
            return Response({'error': 'doctor_id given ID not found'}, status=status.HTTP_404_NOT_FOUND)
       
        #DoctorNameAuth = doctor.Doctors_Name
        
        
       
        #image_urls = get_dicom_image_urls(folder_url)
        #vtk_urls = get_dicom_image_VTK_File(folder_url)
        
        # Get the DICOMSeriesload object for the patient
        User_load, created = UserAuthority.objects.get_or_create(Doctors_ids_Auth =doctor )
        #User_load.Doctor_Name_Auth = DoctorNameAuth
        
       
        User_load.save()
        
        from rest_framework_simplejwt.tokens import RefreshToken
        refresh = RefreshToken.for_user(User_load)
        token = str(refresh.access_token)
            # Get the Patient and Series_Load objects
        #patient = PatientsList.objects.get(Patient_IDs=1215)
            #series_load = DICOMSeriesload.objects.get(id=1)
        #PatientsList_instance = DICOMSeriesload(Patient_id_Series_Load=patient,  Series_array=image_urls , VTK_File= vtk_urls)
        #PatientsList_instance.save()
        
        return Response({ 'token': token}, status=status.HTTP_200_OK) 
            
def set_password(self, raw_password):
        self.password = make_password(raw_password)

#def check_password(self, raw_password):
 #   return check_password(raw_password, self.password)  

import hashlib

def check_password(raw_password, hashed_password):
    """
    Check if the raw password matches the hashed password.
    """
    salt = hashed_password[:32]  # Get the salt from the hashed password
    encoded_raw_password = raw_password.encode('utf-8')
    encoded_salt = salt.encode('utf-8')
    hashed_raw_password = hashlib.pbkdf2_hmac('sha256', encoded_raw_password, encoded_salt, 100000)
    encoded_hashed_raw_password = hashed_raw_password.hex()
    return encoded_hashed_raw_password == hashed_password[32:]


from django.contrib.auth import get_user_model 


    
def check_password_plain_text(password, user):
    
   
    if password == user.password:
        
        return True
    else:
        return False
        
class UserViewSet(viewsets.ModelViewSet):
    queryset = UserAuthority.objects.all()
    serializer_class = UserAuthnSerializer
    permission_classes = [permissions.AllowAny]

    @action(detail=False, methods=['post'])
    def signup(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        #if serializer.is_valid():
            # Hash the plaintext password before saving the user
            #password = make_password(serializer.validated_data['password'])
        # If the user is an employee, require the DoctorList field to be submitted
        if user.is_employee:
            doctor_list_id = request.data.get('doctors_ids_auth')
            doctor_list = DoctorList.objects.get(id=doctor_list_id)
            user.doctors_ids_auth = doctor_list
            user.save()

        # Generate a JWT token for the user
        refresh = RefreshToken.for_user(user)
        token = str(refresh.access_token)

        # Return the JWT token along with the user data
        return Response({
            'token': token,
            'user': self.serializer_class(user).data
        })

    @action(detail=False, methods=['post'])
    def signin(self, request):
        email = request.data.get('email')
        password = request.data.get('password')

        # Check if email and password are provided in the request data
        if not email or not password:
            return Response({'error': 'Please provide both email and password.'}, status=status.HTTP_400_BAD_REQUEST)
        print(email)
        #print(password)
        # Retrieve the user object with the provided email address
        try:
            user = UserAuthority.objects.get(email=email)
           # print(user.password)
        except UserAuthority.DoesNotExist:
            return Response({'error': 'Invalid email or passwordjjjj.'}, status=status.HTTP_401_UNAUTHORIZED)
        print(user.password)
        print(password)
        
        # Check if the provided password matches the user's password
        if  password == user.password:
            print("password")
            #return  Response(1)
        else:
            return  Response({'error': 'Invalid email or password.'}, status=status.HTTP_401_UNAUTHORIZED)
         # Check if the provided password matches the user's password
        
         # Generate a JWT token for the user
        refresh = RefreshToken.for_user(user)
        token = str(refresh.access_token)

        # Return the JWT token along with the user data
        return Response({
            'token': token,
            'user': self.serializer_class(user).data
        })
        
        
####################################################################3
#filters       
class viewsets_PatientsListfilter(viewsets.ModelViewSet):
    serializer_class = PatientsListSerializer

    def get_queryset(self):
        technician_id = self.kwargs.get('technician_id')
        if technician_id:
            queryset = PatientsList.objects.filter(Technacian_ID=technician_id)
        else:
            queryset = PatientsList.objects.all()
        return queryset      
    
class viewsets_PatientsListfilterid(viewsets.ModelViewSet):
    serializer_class = PatientsListSerializer

    def get_queryset(self):
        patient_id = self.kwargs.get('patient_id')
        if patient_id:
            queryset = PatientsList.objects.filter(Patient_IDs=patient_id)
        else:
            queryset = PatientsList.objects.all()
        return queryset      
    
class viewsets_userAuthfilter(viewsets.ModelViewSet):
    serializer_class = UserAuthnSerializer

    def get_queryset(self):
        doctor_id_user = self.kwargs.get('doctor_id_user')
        if doctor_id_user:
            queryset = UserAuthority.objects.filter( Doctor_id_user= doctor_id_user)
        else:
            queryset = UserAuthority.objects.all()
        return queryset      



class viewsets_Dicomseriesfilter(viewsets.ModelViewSet):
    serializer_class = DICOMSeriesloadSerializer

    def get_queryset(self):
        patient_IDs = self.kwargs.get('patient_IDs')
        if patient_IDs:
            queryset = DICOMSeriesload.objects.filter(Patient_id_Series_Load__Patient_IDs=patient_IDs)
        else:
            queryset = DICOMSeriesload.objects.all()
        return queryset      
#########################################################################################################
#Models:
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
from django.shortcuts import get_object_or_404
import ast
@csrf_exempt

  # Example string representation of a list

def handle_image_id(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        #data = json.loads(request.body)
        patient_id = data.get('patient_id')
            
        image_id = data.get('id')
        pixils= data.get('pixel_data')
       # datal = data.get('pixelData')
        print(pixils)
        string = image_id 
        id = int(string.split(":")[1].strip())
        print(id)
        print(f"Received image ID: {id}")
        patient = PatientsList.objects.get(Patient_IDs=patient_id)
        image = DICOMSeriesload.objects.get( id=3)
        image_url = image.Series_array 
        image_urls = ast.literal_eval( image_url)
        print(image_urls[id] )


        # Return a JSON response indicating success
        return JsonResponse({'message': f'Image ID {image_id} and Patient ID {patient_id} received and processed successfully.'})
    elif request.method == 'GET':
        # Handle GET requests, if needed
        # You can return a different response or perform any required logic
        return JsonResponse({'message': 'GET request handled successfully.'})
    else:
        # Return a JSON response indicating invalid request method
        return JsonResponse({'error': 'Invalid request method.'}, status=400)
    
import json

def save_json_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file)


####################################################model###################################################

def dice_coeff(ytrue, ypred, smooth=100):
    ytrue_flat=ytrue
    ypred_flat=ypred
    ytrue_flat = Kb.flatten((tf.cast(ytrue, tf.float32)))
    ypred_flat = Kb.flatten((tf.cast(ypred, tf.float32))) 
    intersection = kb.sum(ytrue_flat * ypred_flat)
    total_area = kb.sum(ytrue_flat) + kb.sum(ypred_flat)
    return (2*intersection )/(total_area )

def dice_coeff_loss(ytrue,ypred, smooth=100):
    return 1-dice_coeff(ytrue, ypred, smooth)

def iou_coeff(ytrue, ypred, smooth=100):
    intersection = kb.sum(ytrue * ypred)
    union = kb.sum(ytrue + ypred) - intersection
    return (intersection + smooth)/(union + smooth)

model = load_model("K:/Studing/GP/DICOM Viewer/GP/GP_Back/savedModels/unet_carotid_seg_85.hdf5",custom_objects={'dice_coeff_loss': dice_coeff_loss, 'iou_coeff': iou_coeff, 'dice_coeff': dice_coeff})
model_classification = load_model("K:/Studing/GP/DICOM Viewer/GP/GP_Back/savedModels/carotid_classification.hdf5")



def image_processing(img):
    #dimensions
    img = np.expand_dims(img, axis=1)
    img = np.moveaxis(img,1,2)
    img = img[np.newaxis, :, :, :]
    pred = model.predict(img)
    
    #normalize
    max_input = np.max(img)
    min_input = np.min(img)
    img= (img- min_input)/(max_input-min_input)
    return img

def get_predicted_mask(img):
    pred = model.predict(img)
    return pred

def get_predicted_contour(pred):
    cs = plt.contour(np.squeeze(pred)>0.5, colors = 'r')
    
    wall_contour = cs.collections[0].get_paths()[0]
    wall_cs = wall_contour.vertices.tolist()
 
    try:
        lumen_contour = cs.collections[0].get_paths()[1]
        lumen_cs = lumen_contour.vertices.tolist()
        
        return wall_cs, lumen_cs
    
    except:
        return wall_cs


def crop_left(img):
    img = np.array(img)
    print(f'imag after np array {np.shape(img)}')
    img=np.reshape(img,(432,432))
    print(f'imag after reshape {img}')
    img = img[120:320, 216:400]
    img =np.pad( img , pad_width=((28, 28), (36, 36)), mode='constant')
    return img
def crop_right(img):
    img = np.array(img)
    img=np.reshape(img,(432,432))
    img= img[120:320,32:216]
    img =np.pad( img , pad_width=((28, 28), (36, 36)), mode='constant')
    return img 


def classification_processing(img):
    cropped_left=crop_left(img)
    processed_left = image_processing(cropped_left)

    ##ocessed_left = processed_left[np.newaxis, :, :,:]
    
    pred_left = get_predicted_mask(processed_left)

    img_thres_left = pred_left
    img_thres_left[ pred_left < 0.5 ] = 0
    img_thres_left[ pred_left > 0.5 ] = 1
   
    new_img_left=np.multiply(processed_left,img_thres_left)



    cropped_right=crop_right(img)
    processed_right = image_processing(cropped_right)

    ##ocessed_right = processed_right[np.newaxis, :, :,:]
    
    pred_right = get_predicted_mask(processed_right)

    img_thres_right = pred_right
    img_thres_right[ pred_right < 0.5 ] = 0
    img_thres_right[ pred_right > 0.5 ] = 1
   
    new_img_right=np.multiply(processed_right,img_thres_right)


    return new_img_left, new_img_right


def perform_classification(img):
    left, right = classification_processing(img)

    print('left', left.shape)
    left_pred = model_classification.predict(left)
    if left_pred<0.5:
        print('left pred',0)
        left_result='Normal'
    else:
        print('left pred',1)
        left_result='Abnormal'



    print('right', right.shape)
    right_pred=model_classification.predict(right)
    if right_pred<0.5:
        print('right pred',0)
        right_result='Normal'
    else:
        print('right pred',1)
        right_result='abNormal'
    return(left_pred,left_result, right_pred,right_result)

def head_fun(data):
    cropped_left=crop_left(data)
    processed_img = image_processing(cropped_left)
    pred_left = get_predicted_mask(processed_img)   

    pred_left = np.squeeze(pred_left)
    pred_left = pred_left[28:-28, 36:-36]
    pred_left_reconstructed = np.zeros((432, 432))
    pred_left_reconstructed[120:320, 216:400] = pred_left
    cropped_right=crop_right(data)
    processed_img_right = image_processing(cropped_right)
    pred_right = get_predicted_mask(processed_img_right)   
    pred_right = np.squeeze(pred_right)
    pred_right = pred_right[28:-28, 36:-36]
    pred_right_reconstructed = np.zeros((432, 432))                            
    pred_right_reconstructed[120:320, 32:216] = pred_right  
    print(f'shape of ritgh{(pred_right_reconstructed)} ')
    print(f'shape of left{np.shape(pred_left_reconstructed)} ')

    
    json_list=[]
    try:
         lumen_cs, wall_cs = get_predicted_contour(pred_left_reconstructed)
         print(f'lumen left  {np.shape(lumen_cs)}')
         print(f'wall left {np.shape(wall_cs)}')
         
    
    except:
         print("salma")
         lumen_cs=[]
         wall_cs = get_predicted_contour(pred_left_reconstructed)
         print(f'wall except left {np.shape(wall_cs)}')
       
    json_list=lumen_cs+wall_cs
    print(f'after left {np.shape(json_list)}')
    contour_lumen_right=[]
    contour_wall_right=[]
    try:
         lumen_cs, wall_cs = get_predicted_contour(pred_right_reconstructed)
         print(f'after lumen right {np.shape(lumen_cs)}')
         print(f'after wall right {np.shape(wall_cs)}')
        
    
    
    except:
         print("salma")
    json_list+=lumen_cs+wall_cs
    print(f'after except right {np.shape(json_list)}')
    
    json_data=[]
    for i in range(len(json_list)-1):
         x1, y1 = json_list[i]
         x2, y2 = json_list[i+1]
         json_data.append({
             "x": x1,
             "y": y1,
             "highlight": True,
             "active": True,
             "lines": [x2,y2]})
    print(f"json_list {json_data}")     
         
    with open('data_final_yarab.json', 'w') as file:
         json.dump(json_data, file) 
    return json_data 


import numpy as np
@api_view(['POST'])
def save_image(request):
    serializer = ImageSerializer(data=request.data)
    if serializer.is_valid():
        image_id = serializer.validated_data['image_id']
        pixel_data = serializer.validated_data['pixel_data']
        number = int( image_id .split(':')[1])
        number=int(number)
        # Save the data to the database or perform other actions as needed
        # Return a JSON response
        data = {
            
            'pixel_data': pixel_data
        }
      #  filename = 'image_data.json'
# Specify the desired file path on your computer
       # filepath = os.path.join('C:/Users/seif/Downloads/Front-with-Back-main (2)', filename)
       # save_json_file(data, filepath)
        left_pred,left_result, right_pred,right_result=perform_classification(pixel_data)

        print(image_id)
        print(type(pixel_data))
        print(np.unique(pixel_data))
        data_file=head_fun(pixel_data)
        print(data_file)
        response_data = {
            'status': 'success',
            'data_file': data_file,  # Include the data_file in the response
            'id': image_id,
            'left_pred':left_pred,
            'left_result':left_result,
            'right_pred':right_pred,
            'right_result':right_result,


        }
        
        return Response(response_data)
       # return Response({'status': 'success'})
    else:
        return Response(serializer.errors, status=400)
    


  
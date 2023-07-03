from django.db import models

# Create your models here.
from django.db import models
#from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token
from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from pydicom import dcmread
import gc
import os
import time
import vtk
import pydicom
from pydicom.sequence import Sequence
from pydicom import dcmread
from rest_framework.request import Request

from django.urls import reverse
from rest_framework.response import Response
from django.db import models
from django.contrib.auth.base_user import BaseUserManager
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin

from django.contrib.auth.base_user import BaseUserManager, AbstractBaseUser
from django.contrib.auth.models import PermissionsMixin, Group, Permission
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.validators import UnicodeUsernameValidator
from django.utils import timezone
from datetime import datetime, timedelta
import jwt
from django.conf import settings
from django.contrib.auth.hashers import make_password, check_password

# Try importing numpy
try:
    import numpy as np
    have_numpy = True
except ImportError:
    np = None  # NOQA
    have_numpy = False

# Create your models here.
class PatientsList(models.Model):
    NONE = 'None'
    MRI = 'MRI'
    CT = 'CT'
    ULTRASOUND = 'ULTRASOUND'
    XRAY = 'XRAY'
    Modalities = [
        (NONE, 'NONE'),
        (MRI, 'MRI'),
        (CT, 'CT'),
        (ULTRASOUND, 'ULTRASOUND'),
        (XRAY, 'XRAY'),
    ]
   
    Patient_IDs=models.CharField(max_length=150,unique=True)
    Patient_Name =models.CharField(max_length=150) 
    Patient_Gender =models.CharField(max_length=10) 
    Patient_Phone =models.CharField(max_length=20,unique=True)
    Patient_Email =models.CharField(max_length=100,unique=True)
    Patient_Address =models.CharField(max_length=200)
    Patient_Framingham_Score_target =models.IntegerField()
    # Patient_BirthDay =models.DateField(auto_now_add=True)
    Patient_National_ID =models.CharField(max_length=20,unique=True)
    Patient_Nationality =models.CharField(max_length=50)
    Patient_Weight =models.IntegerField()
    Patient_Height =models.IntegerField()
    Patient_BMI =models.IntegerField()
    Patient_Modality = models.CharField(max_length=15,default='none')
    Patient_Smoking =models.CharField(max_length=10)
    Patient_Diabetes =models.CharField(max_length=10)
    Patient_Cholestrol =models.IntegerField()
    Patient_Triglycrides =models.IntegerField()
    Patient_LDL =models.IntegerField()
    Patient_HDL =models.IntegerField()
    Patient_Systolic_Pressure =models.IntegerField()
    Patient_Calcium_Score =models.IntegerField()
    Patient_diastolic_Pressure =models.IntegerField()
    Series_URL = models.CharField(max_length=400,unique=True)
    DoctorPatient_Name =models.CharField(max_length=200)
    Technacian_Name =models.CharField(max_length=200)
    Technacian_ID =models.CharField(max_length=200)
    
    
    #VTK_File = models.FileField(max_length=300,unique=True)
    #Series_array = models.CharField(max_length=400,unique=True)
    
   
    #def list_files(self):
        #files=[]
        #pixels=[]
        #patientsseriesurl = PatientsList.objects.filter() 
        
       # for path in  patientsseriesurl:
            #path= path.Series_URL
            #"""List all files in the directory, recursively. """
            #for item in os.listdir(path):
                #item = os.path.join(path, item)
                #ds = dcmread(item)
                #pixel_item = ds.pixel_array
        #print("item :", item)
                #if os.path.isdir(item):
                   #print("Hello Sandra")
           # list_files()
           # print("list files :", listFiles)
                #else:
                   #files.append(item)
                  # pixels.append(pixel_item)
        
        #return "done"
    
   
    #def convert_to_vtk(self):
        
        
        # Check if VTK file already exists
        #if self.VTK_File:
           # return os.path.basename(self.VTK_File.name)
        
        #else:
            # Read DICOM images from directory
           # reader = vtk.vtkDICOMImageReader()
          #  reader.SetDirectoryName(self.Series_URL)
           # reader.Update()

            # Convert images to VTK polydata
           # geometryFilter = vtk.vtkImageDataGeometryFilter()
           # geometryFilter.SetInputConnection(reader.GetOutputPort())
           # geometryFilter.Update()
           # vtkPolyData = geometryFilter.GetOutput()

            # Save VTK file
            #file_name = f"output_{self.Patient_IDs}.vti"
            #writer = vtk.vtkXMLImageDataWriter()
            #writer.SetFileName(file_name)
            #writer.SetInputData(vtkPolyData)
            #writer.Write()

            # Save file path in model field
           # self.VTK_File.save(file_name, open(file_name, 'rb'))
        
        
           # return file_name


class DoctorList(models.Model):
    Doctors_IDs= models.CharField(max_length=250,unique=True)
    Doctors_Name =models.CharField(max_length=250) 
    Doctors_Email =models.CharField(max_length=250,unique=True)
    Doctors_Address =models.CharField(max_length=250)
    Doctors_Phone =models.CharField(max_length=50,unique=True)
    Doctors_Position =models.CharField(max_length=200)

    
class AssignDoctortoPatient(models.Model):     
    patients_ids_For_Doctors = models.ForeignKey(PatientsList, on_delete=models.CASCADE)
    Doctors_ids_For_Patients = models.ForeignKey(DoctorList, on_delete=models.CASCADE)
    



#@receiver(post_save, sender=settings.AUTH_USER_MODEL)
#def TokenCreate(sender, instance, created, **kwargs):
   # if created:
      #  Token.objects.create(user=instance)
      

    
    
    
class DICOMSeriesload(models.Model):
    Patient_id_Series_Load = models.ForeignKey(PatientsList, on_delete=models.CASCADE)
    Series_array = models.CharField(max_length=400,unique=True)
    VTK_File = models.CharField(max_length=300,unique=True)
    Series_state= models.CharField(max_length=50)
    #patient_name = models.CharField(max_length=100)
    #study_date = models.DateField()
    #series_data = models.BinaryField()  
    
    
class PatientReport(models.Model):  
   
    reports_patientlist = models.ForeignKey(PatientsList, on_delete=models.CASCADE)
    Doctor_Name =models.CharField(max_length=200)
    Technacian_Name =models.CharField(max_length=200)
    Technacian_ID =models.CharField(max_length=200)
    Patient_Name =models.CharField(max_length=150)
    Patient_Phone =models.CharField(max_length=20)
    #reports_doctorlist = models.ForeignKey(DoctorList, on_delete=models.CASCADE)
    Description_patient_case = models.CharField(max_length=500)
    
    
######################################################################################################33

class UserAuthority(models.Model):
    email = models.EmailField(_('email'),
        max_length=150,
        blank=True,
        null=True,
        unique=True,
        help_text=_('Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.'),
        validators=[UnicodeUsernameValidator()],
        error_messages={
            'unique': _("A user with that username already exists."),
        },)
    password = models.CharField(max_length=50,unique=True)
    is_employee = models.BooleanField(default=False)
    is_Manager = models.BooleanField(default=False)
    is_Receptionist = models.BooleanField(default=False)
    Doctor_id_user = models.CharField(max_length=50)
    Doctors_ids_Auth= models.CharField(max_length=50)
    Doctor_Name_Auth =models.CharField(max_length=200)
    Gender=models.CharField(max_length=50)
##################################################################################################


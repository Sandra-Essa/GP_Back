from rest_framework import serializers, status
from .models import PatientsList,DoctorList,AssignDoctortoPatient,DICOMSeriesload,PatientReport,UserAuthority

from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from rest_framework import serializers
from django.contrib.auth import get_user_model, authenticate
from django.core.exceptions import ValidationError
from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer


#class UserSerializer(serializers.ModelSerializer):
   # class Meta:
       # model = User
       # fields = ('id', 'username', 'password')
       # extra_kwargs = {'password': {'write_only': True, 'required': True}}


class PatientsListSerializer(serializers.ModelSerializer):
    class Meta:
        model=PatientsList
        fields=('id','Patient_IDs','Patient_Name','Patient_Gender','Patient_Framingham_Score_target','Patient_Phone','Patient_Email','Patient_Address','Patient_National_ID','Patient_Nationality','Patient_Weight','Patient_Height','Patient_BMI','Patient_Modality','Patient_Smoking','Patient_Diabetes','Patient_Cholestrol','Patient_Triglycrides','Patient_LDL','Patient_HDL','Patient_Systolic_Pressure','Patient_Calcium_Score','Patient_diastolic_Pressure','Series_URL','DoctorPatient_Name','Technacian_Name','Technacian_ID')
        

 
  

class DICOMSeriesloadSerializer(serializers.ModelSerializer):
    Patient_IDs = serializers.ReadOnlyField(source='Patient_id_Series_Load.Patient_IDs')
    class Meta:
        model=DICOMSeriesload
        fields=('id','Patient_IDs','Series_array','VTK_File','Series_state')       

    
  
  
class DoctorListSerializer(serializers.ModelSerializer):
    class Meta:
        model=DoctorList
        fields=('id','Doctors_IDs','Doctors_Name','Doctors_Phone','Doctors_Email','Doctors_Address','Doctors_Position')


class AssignDoctortoPatientSerializer(serializers.ModelSerializer):
    class Meta:
        model=AssignDoctortoPatient
        fields='__all__'

#class PatientsSeriesURLSSerializer(serializers.ModelSerializer):
   # class Meta:
        #model= AssignDoctor
        #fields='__all__'   
        
        
class PatientReportSerializer(serializers.ModelSerializer):
   
    Patient_IDs = serializers.ReadOnlyField(source='reports_patientlist.Patient_IDs')
    #Patient_Name = serializers.ReadOnlyField(source='reports_patientlist.Patient_Name')
    #Patient_Phone = serializers.ReadOnlyField(source='reports_patientlist.Patient_Phone')
    #Doctor_Name = serializers.ReadOnlyField(source='reports_patientlist.Doctor_Name')
    #Technacian_Name = serializers.ReadOnlyField(source='reports_patientlist.Technacian_Name')
    #Technacian_ID= serializers.ReadOnlyField(source='reports_patientlist.Technacian_ID')
    class Meta:
        model=PatientReport
        fields=('id','Patient_IDs','Patient_Name','Patient_Phone','Doctor_Name','Technacian_Name','Technacian_ID','Description_patient_case') 
    
    
    
class FolderSerializer(serializers.Serializer):
    folder_url = serializers.CharField()
    #patient_id = serializers.CharField()
    
    
#####################################################
class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        data = super().validate(attrs)
        user = self.user
        if user.is_superuser:
            # If the user is an admin, return the data as is
            return data
        elif hasattr(user, 'Doctors_Name'):
            # If the user is a doctor, add the doctor information to the data
            serializer = DoctorListSerializer(user.Doctors_Name)
            data['Doctors_Name'] = serializer.data
            return data
        else:
            raise serializers.ValidationError('Invalid credentials')
        ###################################################################
        
class UserAuthnSerializer(serializers.ModelSerializer):
    #Doctors_IDs = serializers.ReadOnlyField(source='Doctors_ids_Auth.Doctors_IDs')
    
    class Meta:
        model= UserAuthority
        fields=('id','email','password','Gender','is_employee','is_Manager','is_Receptionist','Doctor_id_user','Doctor_Name_Auth','Gender')


    
class ImageSerializer(serializers.Serializer):
    image_id = serializers.CharField()
    pixel_data = serializers.ListField(child=serializers.IntegerField())       


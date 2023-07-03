# Generated by Django 4.1.7 on 2023-06-22 18:41

import django.contrib.auth.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DoctorList',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Doctors_IDs', models.CharField(max_length=250, unique=True)),
                ('Doctors_Name', models.CharField(max_length=250)),
                ('Doctors_Email', models.CharField(max_length=250, unique=True)),
                ('Doctors_Address', models.CharField(max_length=250)),
                ('Doctors_Phone', models.CharField(max_length=50, unique=True)),
                ('Doctors_Position', models.CharField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='PatientsList',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Patient_IDs', models.CharField(max_length=150, unique=True)),
                ('Patient_Name', models.CharField(max_length=150)),
                ('Patient_Gender', models.CharField(max_length=10)),
                ('Patient_Phone', models.CharField(max_length=20, unique=True)),
                ('Patient_Email', models.CharField(max_length=100, unique=True)),
                ('Patient_Address', models.CharField(max_length=200)),
                ('Patient_Framingham_Score_target', models.IntegerField()),
                ('Patient_National_ID', models.CharField(max_length=20, unique=True)),
                ('Patient_Nationality', models.CharField(max_length=50)),
                ('Patient_Weight', models.IntegerField()),
                ('Patient_Height', models.IntegerField()),
                ('Patient_BMI', models.IntegerField()),
                ('Patient_Modality', models.CharField(default='none', max_length=15)),
                ('Patient_Smoking', models.CharField(max_length=10)),
                ('Patient_Diabetes', models.CharField(max_length=10)),
                ('Patient_Cholestrol', models.IntegerField()),
                ('Patient_Triglycrides', models.IntegerField()),
                ('Patient_LDL', models.IntegerField()),
                ('Patient_HDL', models.IntegerField()),
                ('Patient_Systolic_Pressure', models.IntegerField()),
                ('Patient_Calcium_Score', models.IntegerField()),
                ('Patient_diastolic_Pressure', models.IntegerField()),
                ('Series_URL', models.CharField(max_length=400, unique=True)),
                ('DoctorPatient_Name', models.CharField(max_length=200)),
                ('Technacian_Name', models.CharField(max_length=200)),
                ('Technacian_ID', models.CharField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='UserAuthority',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(blank=True, error_messages={'unique': 'A user with that username already exists.'}, help_text='Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.', max_length=150, null=True, unique=True, validators=[django.contrib.auth.validators.UnicodeUsernameValidator()], verbose_name='email')),
                ('password', models.CharField(max_length=50, unique=True)),
                ('is_employee', models.BooleanField(default=False)),
                ('is_Manager', models.BooleanField(default=False)),
                ('Doctor_id_user', models.CharField(max_length=50)),
                ('Doctors_ids_Auth', models.CharField(max_length=50)),
                ('Doctor_Name_Auth', models.CharField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='PatientReport',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Doctor_Name', models.CharField(max_length=200)),
                ('Technacian_Name', models.CharField(max_length=200)),
                ('Technacian_ID', models.CharField(max_length=200)),
                ('Patient_Name', models.CharField(max_length=150)),
                ('Patient_Phone', models.CharField(max_length=20)),
                ('Description_patient_case', models.CharField(max_length=500)),
                ('reports_patientlist', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='backendapi.patientslist')),
            ],
        ),
        migrations.CreateModel(
            name='DICOMSeriesload',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Series_array', models.CharField(max_length=400, unique=True)),
                ('VTK_File', models.CharField(max_length=300, unique=True)),
                ('Patient_id_Series_Load', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='backendapi.patientslist')),
            ],
        ),
        migrations.CreateModel(
            name='AssignDoctortoPatient',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Doctors_ids_For_Patients', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='backendapi.doctorlist')),
                ('patients_ids_For_Doctors', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='backendapi.patientslist')),
            ],
        ),
    ]

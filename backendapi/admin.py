from django.contrib import admin
from .models import PatientsList,DoctorList,AssignDoctortoPatient

admin.site.register(PatientsList)
admin.site.register(DoctorList)
admin.site.register(AssignDoctortoPatient)

#admin.site.register(Post)
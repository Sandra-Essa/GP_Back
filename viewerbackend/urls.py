from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from backendapi import views
from backendapi.views import DicomFolderView,PatientReportView,viewsets_PatientsListfilterid,PatientListView,viewsets_Dicomseriesfilter, UserViewSettest,UserViewSet,UserViewSettest,viewsets_PatientsListfilter,viewsets_userAuthfilter
from rest_framework.authtoken.views import obtain_auth_token
from rest_framework import routers
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView






router = routers.DefaultRouter()
router.register('PatientList', views.viewsets_PatientsList)
router.register('DoctorsList', views.viewsets_DoctorsList)
router.register('AssignDoctor', views.viewsets_AssignDoctor)
router.register('DICOMSeriesload', views.viewsets_DICOMSeriesload)
router.register('PatientReport', views.viewsets_PatientReport)
router.register('UserAuthn', views.viewsets_UserAuthn)

#router.register(r'users', UserViewSet)
#router.register('AssignDoctor',  views.viewsets_AssignDoctor)
#router.register('User', UserViewSet)
#router.register('Patients', PatientsViewSet)
#router.register('PatientsSeries', PatientsSeriesViewSet)
#router.register('DoctorsList', DoctorsListViewSet,basename='DoctorsList')
#router.register('DoctorsListFilter', DoctorsListFilterViewSet, basename='DoctorsListFilter')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('backendapi/', include(router.urls)),
    # find doctor 
    path('backendapiv/finddoctor', views.find_doctor),
    # find doctor 
    path('backendapi/findpatient', views.find_patient),
    # assign doctor to specific patient
    path('backendapi/new_assign', views.new_assign),
    path('backendapi/series_urls', views.series_urls),
    path('tokenrequest/', obtain_auth_token),
   
    
    path('backendapi/dicom-series-test', DicomFolderView.as_view()),
    path('backendapi/patient-report-test', PatientReportView.as_view()),
    
    
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
   
   
    path('api/signin/', UserViewSet.as_view({'post': 'signin'}), name='signin'),
    path('api/signup/', UserViewSet.as_view({'post': 'signup'}), name='signup'),
    path('UserViewSettest/', UserViewSettest.as_view(), ),
    path('patientslistfilter/<int:technician_id>/', viewsets_PatientsListfilter.as_view({'get': 'list'})),
    
    path('PatientsListfilterid/<int:patient_id>/', viewsets_PatientsListfilterid.as_view({'get': 'list'})),
    
    path('dicomseriesfilter/<int:patient_IDs>/', viewsets_Dicomseriesfilter.as_view({'get': 'list'})),
    path('userAuthfilter/<int:doctor_id_user>/', viewsets_userAuthfilter.as_view({'get': 'list'})),
    path('backendapi/saveimage/', views.save_image),
    path('backendapi/imageid/',views.handle_image_id, name='handle_image_ids'),
   
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
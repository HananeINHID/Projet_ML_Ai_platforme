from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('regLog_details/', views.regLog_details, name='regLog_details'),
    path('regLog_atlier/', views.regLog_atelier, name='regLog_atelier'),
    path('regLog_tester/', views.regLog_tester, name='regLog_tester'),
    path('regLog_prediction', views.regLog_prediction, name='regLog_prediction'),
]

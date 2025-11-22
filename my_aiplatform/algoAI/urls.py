from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('regLog_details/', views.regLog_details, name='regLog_details'),
    path('regLog_atlier/', views.regLog_atelier, name='regLog_atelier'),
    path('regLog_tester/', views.regLog_tester, name='regLog_tester'),
    path('regLog_prediction', views.regLog_prediction, name='regLog_prediction'),
    path('ran_forest_details/', views.ran_forest_details, name='ran_forest_details'),
    path('ran_forest_atelier/', views.ran_forest_atelier, name='ran_forest_atelier'),
    path('ran_forest_tester/', views.ran_forest_tester, name='ran_forest_tester'),
    path('ran_forest_reg_details/', views.ran_forest_reg_details, name='ran_forest_reg_details'),
    path('ran_forest_reg_atelier/', views.ran_forest_reg_atelier, name='ran_forest_reg_atelier'),
    path('rf_prediction/', views.rf_prediction, name='rf_prediction'),
    path('ran_forest_reg_tester/', views.ran_forest_reg_tester, name='ran_forest_reg_tester'),
    path('rf_student_prediction/', views.rf_student_prediction, name='rf_student_prediction'),
    path('XGBoost_details/', views.XGBoost_details, name='XGBoost_details'),
    path('XGBoost_atelier/', views.XGBoost_atelier, name='XGBoost_atelier'),


]

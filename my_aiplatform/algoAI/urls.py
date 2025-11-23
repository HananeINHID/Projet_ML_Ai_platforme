from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('regLog_details/', views.regLog_details, name='regLog_details'),
    path('regLog_atelier/', views.regLog_atelier, name='regLog_atelier'),
    path('regLog_tester/', views.regLog_tester, name='regLog_tester'),
    path('regLog_prediction/', views.regLog_prediction, name='regLog_prediction'),
    path('tree_r_details/', views.tree_r_details, name='tree_r_details'),
    path('tree_r_atelier/', views.tree_r_atelier, name='tree_r_atelier'),
    path('tree_r_tester/', views.tree_r_tester, name='tree_r_tester'),
    path('tree_r_prediction/', views.tree_r_prediction, name='tree_r_prediction'),
    path('svm_c_details/', views.svm_c_details, name='svm_c_details'),
    path('svm_c_atelier/', views.svm_c_atelier, name='svm_c_atelier'),
    path('svm_c_tester/', views.svm_c_tester, name='svm_c_tester'),
    path('svm_c_prediction/', views.svm_c_prediction, name='svm_c_prediction'),
    path('xgboost_r_details/', views.xgboost_r_details, name='xgboost_r_details'),
    path('xgboost_r_atelier/', views.xgboost_r_atelier, name='xgboost_r_atelier'),
    path('xgboost_r_tester/', views.xgboost_r_tester, name='xgboost_r_tester'),
    path('xgboost_r_prediction/', views.xgboost_r_prediction, name='xgboost_r_prediction'),
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
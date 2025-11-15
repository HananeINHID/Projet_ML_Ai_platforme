from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('regLog_details/', views.regLog_details, name='regLog_details'),
    path('regLog_aetlier/', views.regLog_atelier, name='regLog_atelier'),
    path('regLog_tester/', views.regLog_tester, name='regLog_tester'),
    path('regLog_prediction/', views.regLog_prediction, name='regLog_prediction'),
    path('tree_c_details/', views.tree_c_details, name='tree_c_details'),
    path('tree_c_atelier/', views.tree_c_atelier, name='tree_c_atelier'),
    path('tree_c_tester/', views.tree_c_tester, name='tree_c_tester'),
    path('tree_c_prediction/', views.tree_c_prediction, name='tree_c_prediction'),
]

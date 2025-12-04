from django.urls import path
from django.views.generic import RedirectView
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    
    # Regression Logistique
    path('logistic_regression/regLog_details/', views.regLog_details, name='regLog_details'),
    path('logistic_regression/regLog_atelier/', views.regLog_atelier, name='regLog_atelier'),
    path('logistic_regression/regLog_tester/', views.regLog_tester, name='regLog_tester'),
    path('logistic_regression/regLog_prediction/', views.regLog_prediction, name='regLog_prediction'),
    
    # Linear Regression
    path('linear_regression/linreg_details/', views.linreg_details, name='linreg_details'),
    path('linear_regression/linreg_atelier/', views.linreg_atelier, name='linreg_atelier'),
    path('linear_regression/linreg_tester/', views.linreg_tester, name='linreg_tester'),
    path('linear_regression/linreg_prediction/', views.linreg_prediction, name='linreg_prediction'),

    # Arbre de Décision (Regression)
    path('decision_tree/tree_r_details/', views.tree_r_details, name='tree_r_details'),
    path('decision_tree/tree_r_atelier/', views.tree_r_atelier, name='tree_r_atelier'),
    path('decision_tree/tree_r_tester/', views.tree_r_tester, name='tree_r_tester'),
    path('decision_tree/tree_r_prediction/', views.tree_r_prediction, name='tree_r_prediction'),

    # Arbre de Décision (Classification)
    path('decision_tree/tree_c_details/', views.tree_c_details, name='tree_c_details'),
    path('decision_tree/tree_c_atelier/', views.tree_c_atelier, name='tree_c_atelier'),
    path('decision_tree/tree_c_tester/', views.tree_c_tester, name='tree_c_tester'),
    path('decision_tree/tree_c_prediction/', views.tree_c_prediction, name='tree_c_prediction'),
    
    # SVM (Classification)
    path('svm/svm_c_details/', views.svm_c_details, name='svm_c_details'),
    path('svm/svm_c_atelier/', views.svm_c_atelier, name='svm_c_atelier'),
    path('svm/svm_c_tester/', views.svm_c_tester, name='svm_c_tester'),
    path('svm/svm_c_prediction/', views.svm_c_prediction, name='svm_c_prediction'),

    # SVM (Régression)
    path('svm/svm_r_details/', views.svm_r_details, name='svm_r_details'),
    path('svm/svm_r_atelier/', views.svm_r_atelier, name='svm_r_atelier'),
    path('svm/svm_r_tester/', views.svm_r_tester, name='svm_r_tester'),
    path('svm/svm_r_prediction/', views.svm_r_prediction, name='svm_r_prediction'),
    
    # XGBoost (Regression)
    path('xgboost/xgboost_r_details/', views.xgboost_r_details, name='xgboost_r_details'),
    path('xgboost/xgboost_r_atelier/', views.xgboost_r_atelier, name='xgboost_r_atelier'),
    path('xgboost/xgboost_r_tester/', views.xgboost_r_tester, name='xgboost_r_tester'),
    path('xgboost/xgboost_r_prediction/', views.xgboost_r_prediction, name='xgboost_r_prediction'),
    
    # XGBoost (Classification)
    path('xgboost/XGBoost_details/', views.XGBoost_details, name='XGBoost_details'),
    path('xgboost/XGBoost_atelier/', views.XGBoost_atelier, name='XGBoost_atelier'),
    path('xgboost/XGboost_tester/', views.XGboost_tester, name='XGboost_tester'),
    path('xgboost/XGboost_prediction/', views.XGboost_prediction, name='XGboost_prediction'),
    
    # Random Forest (Classification)
    path('random_forest/ran_forest_details/', views.ran_forest_details, name='ran_forest_details'),
    path('random_forest/ran_forest_atelier/', views.ran_forest_atelier, name='ran_forest_atelier'),
    path('random_forest/ran_forest_tester/', views.ran_forest_tester, name='ran_forest_tester'),
    path('random_forest/rf_prediction/', views.rf_prediction, name='rf_prediction'),
    
    # Random Forest (Regression)
    path('random_forest/ran_forest_reg_details/', views.ran_forest_reg_details, name='ran_forest_reg_details'),
    path('random_forest/ran_forest_reg_atelier/', views.ran_forest_reg_atelier, name='ran_forest_reg_atelier'),
    path('random_forest/ran_forest_reg_tester/', views.ran_forest_reg_tester, name='ran_forest_reg_tester'),
    path('random_forest/rf_student_prediction/', views.rf_student_prediction, name='rf_student_prediction'),

    # URLs de compatibilité (redirections depuis les anciens chemins sans préfixes)
    # Regression Logistique
    path('regLog_details/', RedirectView.as_view(url='/logistic_regression/regLog_details/', permanent=False)),
    path('regLog_atelier/', RedirectView.as_view(url='/logistic_regression/regLog_atelier/', permanent=False)),
    path('regLog_tester/', RedirectView.as_view(url='/logistic_regression/regLog_tester/', permanent=False)),
    path('regLog_prediction/', RedirectView.as_view(url='/logistic_regression/regLog_prediction/', permanent=False)),
    
    # Linear Regression
    path('linreg_details/', RedirectView.as_view(url='/linear_regression/linreg_details/', permanent=False)),
    path('linreg_atelier/', RedirectView.as_view(url='/linear_regression/linreg_atelier/', permanent=False)),
    path('linreg_tester/', RedirectView.as_view(url='/linear_regression/linreg_tester/', permanent=False)),
    path('linreg_prediction/', RedirectView.as_view(url='/linear_regression/linreg_prediction/', permanent=False)),
    
    # Arbre de Décision (Regression)
    path('tree_r_details/', RedirectView.as_view(url='/decision_tree/tree_r_details/', permanent=False)),
    path('tree_r_atelier/', RedirectView.as_view(url='/decision_tree/tree_r_atelier/', permanent=False)),
    path('tree_r_tester/', RedirectView.as_view(url='/decision_tree/tree_r_tester/', permanent=False)),
    path('tree_r_prediction/', RedirectView.as_view(url='/decision_tree/tree_r_prediction/', permanent=False)),
    
    # Arbre de Décision (Classification)
    path('tree_c_details/', RedirectView.as_view(url='/decision_tree/tree_c_details/', permanent=False)),
    path('tree_c_atelier/', RedirectView.as_view(url='/decision_tree/tree_c_atelier/', permanent=False)),
    path('tree_c_tester/', RedirectView.as_view(url='/decision_tree/tree_c_tester/', permanent=False)),
    path('tree_c_prediction/', RedirectView.as_view(url='/decision_tree/tree_c_prediction/', permanent=False)),
    
    # SVM (Classification)
    path('svm_c_details/', RedirectView.as_view(url='/svm/svm_c_details/', permanent=False)),
    path('svm_c_atelier/', RedirectView.as_view(url='/svm/svm_c_atelier/', permanent=False)),
    path('svm_c_tester/', RedirectView.as_view(url='/svm/svm_c_tester/', permanent=False)),
    path('svm_c_prediction/', RedirectView.as_view(url='/svm/svm_c_prediction/', permanent=False)),
    
    # SVM (Régression)
    path('svm_r_details/', RedirectView.as_view(url='/svm/svm_r_details/', permanent=False)),
    path('svm_r_atelier/', RedirectView.as_view(url='/svm/svm_r_atelier/', permanent=False)),
    path('svm_r_tester/', RedirectView.as_view(url='/svm/svm_r_tester/', permanent=False)),
    path('svm_r_prediction/', RedirectView.as_view(url='/svm/svm_r_prediction/', permanent=False)),
    
    # XGBoost (Regression)
    path('xgboost_r_details/', RedirectView.as_view(url='/xgboost/xgboost_r_details/', permanent=False)),
    path('xgboost_r_atelier/', RedirectView.as_view(url='/xgboost/xgboost_r_atelier/', permanent=False)),
    path('xgboost_r_tester/', RedirectView.as_view(url='/xgboost/xgboost_r_tester/', permanent=False)),
    path('xgboost_r_prediction/', RedirectView.as_view(url='/xgboost/xgboost_r_prediction/', permanent=False)),
    
    # XGBoost (Classification)
    path('XGBoost_details/', RedirectView.as_view(url='/xgboost/XGBoost_details/', permanent=False)),
    path('XGBoost_atelier/', RedirectView.as_view(url='/xgboost/XGBoost_atelier/', permanent=False)),
    path('XGboost_tester/', RedirectView.as_view(url='/xgboost/XGboost_tester/', permanent=False)),
    path('XGboost_prediction/', RedirectView.as_view(url='/xgboost/XGboost_prediction/', permanent=False)),
    
    # Random Forest (Classification)
    path('ran_forest_details/', RedirectView.as_view(url='/random_forest/ran_forest_details/', permanent=False)),
    path('ran_forest_atelier/', RedirectView.as_view(url='/random_forest/ran_forest_atelier/', permanent=False)),
    path('ran_forest_tester/', RedirectView.as_view(url='/random_forest/ran_forest_tester/', permanent=False)),
    path('rf_prediction/', RedirectView.as_view(url='/random_forest/rf_prediction/', permanent=False)),
    
    # Random Forest (Regression)
    path('ran_forest_reg_details/', RedirectView.as_view(url='/random_forest/ran_forest_reg_details/', permanent=False)),
    path('ran_forest_reg_atelier/', RedirectView.as_view(url='/random_forest/ran_forest_reg_atelier/', permanent=False)),
    path('ran_forest_reg_tester/', RedirectView.as_view(url='/random_forest/ran_forest_reg_tester/', permanent=False)),
    path('rf_student_prediction/', RedirectView.as_view(url='/random_forest/rf_student_prediction/', permanent=False)),
]

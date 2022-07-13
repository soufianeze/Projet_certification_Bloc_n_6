## Librairies à importer
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,confusion_matrix,roc_auc_score,classification_report
from sklearn.metrics import roc_curve
from sklearn import metrics
from matplotlib.pyplot import figure
#from collections import Counter
from imblearn.over_sampling import SMOTENC 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

st.title ('Patients Survival Prediction')


## Charger le dataset
df = pd.read_csv("dataset.csv")

#col to keep 
col = ['age', 'elective_surgery', 'ethnicity', 'gender', 'icu_admit_source',
       'icu_type', 'pre_icu_los_days', 'weight', 'apache_2_diagnosis',
       'apache_post_operative', 'arf_apache', 'gcs_eyes_apache',
       'gcs_motor_apache', 'gcs_verbal_apache', 'heart_rate_apache',
       'intubated_apache', 'map_apache', 'resprate_apache', 'temp_apache',
       'ventilated_apache', 'd1_diasbp_min', 'd1_heartrate_max', 'd1_mbp_min',
       'd1_resprate_max', 'd1_resprate_min', 'd1_spo2_min', 'd1_sysbp_min',
       'd1_temp_max', 'd1_temp_min', 'h1_diasbp_noninvasive_max',
       'h1_diasbp_noninvasive_min', 'h1_heartrate_max', 'h1_heartrate_min',
       'h1_mbp_noninvasive_max', 'h1_mbp_noninvasive_min', 'h1_resprate_max',
       'h1_resprate_min', 'h1_spo2_max', 'h1_spo2_min', 'h1_sysbp_max',
       'h1_sysbp_min', 'd1_glucose_max', 'd1_glucose_min', 'd1_potassium_max',
       'd1_potassium_min', 'apache_4a_hospital_death_prob',
       'apache_4a_icu_death_prob', 'aids', 'cirrhosis', 'diabetes_mellitus',
       'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma',
       'solid_tumor_with_metastasis', 'apache_3j_bodysystem',
       'apache_2_bodysystem', 'hospital_death']
df = df[col]

# Enelver les NaN

df = df.dropna()

# Enlever les outliers

df = df[df.pre_icu_los_days>=0]                                                                                                                                                  
df = df[df.apache_4a_hospital_death_prob>=0]                                               
df = df[df.apache_4a_icu_death_prob>=0]

## Création des datasets target et features

target_name = 'hospital_death'

y = df.loc[:,target_name]
X = df.loc[:,[c for c in df.columns if c!=target_name]] 


# Seperate the dataset into categorical features and numeric features

numeric_features_with_target = numeric_features = ['age', 'elective_surgery','pre_icu_los_days', 'weight','apache_post_operative', 'arf_apache', 'gcs_eyes_apache', 'gcs_motor_apache', 
                    'gcs_verbal_apache', 'heart_rate_apache', 'intubated_apache', 'map_apache', 'resprate_apache', 'temp_apache', 
                    'ventilated_apache', 'd1_diasbp_min', 'd1_heartrate_max', 'd1_mbp_min', 'd1_resprate_max', 'd1_resprate_min', 
                    'd1_spo2_min', 'd1_sysbp_min', 'd1_temp_max', 'd1_temp_min', 'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min', 
                    'h1_heartrate_max', 'h1_heartrate_min', 'h1_mbp_noninvasive_max', 'h1_mbp_noninvasive_min', 'h1_resprate_max', 'h1_resprate_min', 
                    'h1_spo2_max', 'h1_spo2_min', 'h1_sysbp_max', 'h1_sysbp_min', 'd1_glucose_max', 'd1_glucose_min', 'd1_potassium_max', 'd1_potassium_min', 
                    'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 
                    'leukemia', 'lymphoma', 'solid_tumor_with_metastasis','apache_2_diagnosis']

categorical_features = ['icu_type','ethnicity','apache_3j_bodysystem','gender','icu_admit_source','apache_2_bodysystem']

# Find the index of the categorical features

#[col.index(i) for i in categorical_features]

#Utilisation de SMOTE pour équilibrer le dataset

sm = SMOTENC(categorical_features=[2,3,4,5,55,56],random_state=42)
X_res, y_res = sm.fit_resample(X, y)
y = y_res
X = X_res

#Split les datasets

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=.20,random_state=0)

## Create a Pipeline for numeric features
numeric_features = ['age', 'elective_surgery','pre_icu_los_days', 'weight','apache_post_operative', 'arf_apache', 'gcs_eyes_apache', 'gcs_motor_apache', 
                    'gcs_verbal_apache', 'heart_rate_apache', 'intubated_apache', 'map_apache', 'resprate_apache', 'temp_apache', 
                    'ventilated_apache', 'd1_diasbp_min', 'd1_heartrate_max', 'd1_mbp_min', 'd1_resprate_max', 'd1_resprate_min', 
                    'd1_spo2_min', 'd1_sysbp_min', 'd1_temp_max', 'd1_temp_min', 'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min', 
                    'h1_heartrate_max', 'h1_heartrate_min', 'h1_mbp_noninvasive_max', 'h1_mbp_noninvasive_min', 'h1_resprate_max', 'h1_resprate_min', 
                    'h1_spo2_max', 'h1_spo2_min', 'h1_sysbp_max', 'h1_sysbp_min', 'd1_glucose_max', 'd1_glucose_min', 'd1_potassium_max', 'd1_potassium_min', 
                    'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 
                    'leukemia', 'lymphoma', 'solid_tumor_with_metastasis','apache_2_diagnosis',]

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create a Pipeline for categorical features
categorical_features = ['icu_type','ethnicity','apache_3j_bodysystem','gender','icu_admit_source','apache_2_bodysystem']
categorical_transformer = Pipeline(
    steps=[
        ('encoder', OneHotEncoder())
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocessing on train set
xtrain = preprocessor.fit_transform(xtrain)

# Preprocessing on test set
xtest = preprocessor.transform(xtest)

# Ajouter des sidebar


# Basic information function


def patient_info ():
    #seperate the patient's information into basic information, h1 information and d1 information
    st.sidebar.header('''Patient's information''')

    Age = st.sidebar.slider('Age',float(0), float(130), float(df.age.mean())) #1
    Surgery = st.selectbox('Was the patient admitted to the hospital for an elective surgical operation (0=Yes,1=No) ?',
                            df['elective_surgery'].sort_values().unique()) #2
    Ethnicity = st.selectbox("Select the ethnicity of the patient", df["ethnicity"].sort_values().unique()) #3
    Gender = st.selectbox("Sex", df["gender"].sort_values().unique()) #4
    ICU_admit_source = st.selectbox('Location of the patient before being admitted to the unit',df["icu_admit_source"].sort_values().unique()) #5
    ICU_type = st.selectbox("Select the ICU type", df["icu_type"].sort_values().unique()) #6
    Pre_ICU_days = st.sidebar.slider('pre icu time', float(0),float(df['pre_icu_los_days'].max()),float(df['pre_icu_los_days'].mean())) #7
    Weight = st.sidebar.slider('Weight(kg)',float(0), float(500), float(df.weight.mean())) #8
    Apache_2_diagnosis = st.selectbox("Select the Apache_2_diagnosis", df["apache_2_diagnosis"].sort_values().unique()) #9
    Apache_post_operative = st.selectbox('Select the APACHE operative status (1 for post-operative, 0 for non-operative)',
                                         df["apache_post_operative"].sort_values().unique()) #10
    Renal_failure = st.selectbox('Did this patient have acute renal failure during the first 24 hours (1 for post-operative, 0 for non-operative)',
                                 df["arf_apache"].sort_values().unique()) #11
    GCS_eyes_apache = st.selectbox ("Select the patient's eye opening component",df["gcs_eyes_apache"].sort_values().unique()) #12
    GCS_motor_apche = st.selectbox("Select the patient's motor component",df["gcs_motor_apache"].sort_values().unique()) #13
    GCS_verbal_apche = st.selectbox("Select the patient's verbal component",df["gcs_verbal_apache"].sort_values().unique()) #14
    Heart_rate = st.sidebar.slider('Heart beat', int(0), int(300), int(df.heart_rate_apache.mean())) #15
    Intubated_apache = st.selectbox("Was the patient intubated ? (o for Yes, 1 for No)",df["intubated_apache"].sort_values().unique()) #16
    Map_apache = st.sidebar.slider("The mean arterial pressure measured during the first 24 hours", int(0), int(300), int(df.map_apache.mean())) #17
    Resprate_apache = st.sidebar.slider("The respiratory rate during the first 24 hours", float(0), float(500),float(df.resprate_apache.mean())) #18
    Temp_apache = st.sidebar.slider("The temperature during the first 24 hours", float(0), float(50),float(df.temp_apache.mean())) #19
    Ventilated_apache = st.selectbox("Was the patient ventilated ? (0 for Yes, 1 for No)", df["ventilated_apache"].sort_values().unique()) #20
    D1_diasbp_min = st.sidebar.slider("The patient's lowest diastolic blood pressure during the first 24 hours", 
                                        int(0),int(200),int(df.d1_diasbp_min.mean()))#21
    D1_heartrate_max = st.sidebar.slider("The patient's highest heart rate during the first 24 hours", int(0),int(400),int(df.d1_heartrate_max.mean())) #22
    D1_mbp_min = st.sidebar.slider("The patient's lowest mean blood pressure during the first 24 hours",int(0),int(400),int(df.d1_mbp_min.mean())) #23
    D1_resprate_max = st.sidebar.slider("The patient's highest respiratory rate during the first 24 hours",int(0),int(400),int(df.d1_resprate_max.mean())) #24
    D1_resprate_min = st.sidebar.slider("The patient's lowest respiratory rate during the first 24 hours",int(0),int(400),int(df.d1_resprate_min.mean())) #25
    D1_spo2_min = st.sidebar.slider("The patient's lowest peripheral oxygen saturation during the first 24 hours",int(0),int(400),int(df.d1_spo2_min.mean())) #26
    D1_sysbp_min = st.sidebar.slider("The patient's lowest systolic blood pressure during the first 24 hours",int(0),int(400),int(df.d1_sysbp_min.mean())) #27
    D1_temp_max = st.sidebar.slider("The patient's highest temparature during the first 24 hours",int(0),int(50),int(df.d1_temp_max.mean())) #28
    D1_temp_min = st.sidebar.slider("The patient's lowest temparature during the first 24 hours",int(0),int(50),int(df.d1_temp_min.mean())) #29
    H1_diasbp_noninvasive_max = st.sidebar.slider("The patient's highest diastolic blood pressure during the first hour",int(0),int(400),
                                                    int(df.h1_diasbp_noninvasive_max.mean())) #30
    H1_heartrate_max = st.sidebar.slider("The patient's highest heart rate during the first hour",int(0),int(400),int(df.h1_heartrate_max.mean()))#31
    H1_diasbp_noninvasive_min = st.sidebar.slider("The patient's lowest diastolic blood pressure during the first hour",int(0),int(400),
                                                 int(df.h1_diasbp_noninvasive_min.mean()))#32
    H1_heartrate_min = st.sidebar.slider("The patient's lowest heart rate during the first hour",int(0),int(400),int(df.h1_heartrate_max.mean()))#33
    H1_mbp_noninvasive_max =st.sidebar.slider("The patient's highest mean blood pressure during the first hour",
                                            int(0),int(400),int(df.h1_mbp_noninvasive_max.mean()))#34
    H1_mbp_noninvasive_min =st.sidebar.slider("The patient's lowest mean blood pressure during the first hour",
                                            int(0),int(400),int(df.h1_mbp_noninvasive_min.mean()))#35
    H1_resprate_max = st.sidebar.slider("The patient's highest heart rate during the first hour",int(0),int(400),int(df.h1_resprate_max.mean()))#36
    H1_resprate_min = st.sidebar.slider("The patient's lowest heart rate during the first hour",int(0),int(400),int(df.h1_resprate_min.mean()))#37
    H1_spo2_max = st.sidebar.slider("The patient's highest peripheral oxygen saturation during the first hour",int(0),int(400),int(df.h1_spo2_max.mean()))#38
    H1_spo2_min = st.sidebar.slider("The patient's lowest peripheral oxygen saturation during the first hour",int(0),int(400),int(df.h1_spo2_min.mean()))#39
    H1_sysbp_max = st.sidebar.slider("The patient's highest systolic blood pressure during the first hour",int(0),int(400),int(df.h1_sysbp_max.mean())) #40
    H1_sysbp_min = st.sidebar.slider("The patient's highest systolic blood pressure during the first hour",int(0),int(400),int(df.h1_sysbp_min.mean()))#41
    D1_glucose_max = st.sidebar.slider("The highest glucose concentration of the patient in their serum or plasma during the first 24 hours",
                                        int(0),int(400),int(df.d1_glucose_max.mean()))#42
    D1_glucose_min = st.sidebar.slider("The lowest glucose concentration of the patient in their serum or plasma during the first 24 hours",
                                        int(0),int(400),int(df.d1_glucose_min.mean()))#43
    D1_potassium_max = st.sidebar.slider("The highest potassium concentration for the patient in their serum or plasma during the first 24 hours",
                                        float(0),float(400),float(df.d1_glucose_max.mean()))#44
    D1_potassium_min = st.sidebar.slider("The lowest potassium concentration for the patient in their serum or plasma during the first 24 hours",
                                        float(0),float(400),float(df.d1_glucose_min.mean()))#45
    Apache_4a_hospital_death_prob  = st.sidebar.slider("The APACHE IVa probabilistic prediction of in-hospital mortality for the patient",
                                        float(0),float(400),float(df.apache_4a_hospital_death_prob.mean()))#46
    Apache_4a_icu_death_prob = st.sidebar.slider("The APACHE IVa probabilistic prediction of in ICU mortality for the patient",                           
                                                 float(0),float(400),float(df.apache_4a_icu_death_prob.mean()))#47
    Aids = st.selectbox("Aids (0 for Yes, 1 for No)",df["aids"].sort_values().unique())#48
    Cirrhosis = st.selectbox("Cirrhosis (0 for Yes, 1 for No)",df["cirrhosis"].sort_values().unique())#49
    Diabetes = st.selectbox("Diabetes (0 for Yes, 1 for No)",df["diabetes_mellitus"].sort_values().unique())#50
    Hepatic_failure = st.selectbox("Hepatic failure (0 for Yes, 1 for No)",df["hepatic_failure"].sort_values().unique())#51
    Immunosuppression = st.selectbox("Immunosuppression (0 for Yes, 1 for No)",df["immunosuppression"].sort_values().unique())#52
    Leukemia = st.selectbox("Leukemia (0 for Yes, 1 for No)",df["leukemia"].sort_values().unique())#53
    Lymphoma = st.selectbox("Lymphoma (0 for Yes, 1 for No)",df["lymphoma"].sort_values().unique())#54
    Solid_tumor_with_metastasis = st.selectbox("Solid_tumor_with_metastasis (0 for Yes, 1 for No)",df["solid_tumor_with_metastasis"].sort_values().unique())#55
    Apache_3j_bodysystem = st.selectbox("Apache_3j_bodysystem",df["apache_3j_bodysystem"].sort_values().unique())#56
    Apache_2_bodysystem = st.selectbox("Apache__2_bodysystem",df["apache_2_bodysystem"].sort_values().unique())#57


    data = {    
       'age' : Age,#1
       'elective_surgery' :Surgery,#2
       'ethnicity' : Ethnicity,#3
       'gender' : Gender,#4
       'icu_admit_source' : ICU_admit_source,#5
       'icu_type' : ICU_type,#6
       'pre_icu_los_days' : Pre_ICU_days,#7
       'weight' : Weight,
       'apache_2_diagnosis' : Apache_2_diagnosis,#9
       'apache_post_operative' :Apache_post_operative,
       'arf_apache' : Renal_failure,#11
       'gcs_eyes_apache' : GCS_eyes_apache,
       'gcs_motor_apache' : GCS_motor_apche,#13
       'gcs_verbal_apache' : GCS_verbal_apche,
       'heart_rate_apache':Heart_rate,#15
       'intubated_apache':Intubated_apache,
       'map_apache':Map_apache,#17
       'resprate_apache':Resprate_apache,
       'temp_apache':Temp_apache,#19
       'ventilated_apache':Ventilated_apache,
       'd1_diasbp_min':D1_diasbp_min,#21
       'd1_heartrate_max':D1_heartrate_max,
       'd1_mbp_min':D1_mbp_min,#23
       'd1_resprate_max':D1_resprate_min,
       'd1_resprate_min':D1_resprate_min,#25
       'd1_spo2_min':D1_spo2_min,
       'd1_sysbp_min':D1_sysbp_min,#27
       'd1_temp_max':D1_temp_max,
       'd1_temp_min':D1_temp_min,#29
       'h1_diasbp_noninvasive_max':H1_diasbp_noninvasive_max,
       'h1_heartrate_max':H1_heartrate_max,#31
       'h1_heartrate_min':H1_heartrate_min,
       'h1_diasbp_noninvasive_min':H1_diasbp_noninvasive_min,
       'h1_diasbp_heartrate_min':H1_heartrate_min,#33
       'h1_resprate_max' :H1_resprate_max,
       'h1_resprate_min' :H1_resprate_min,#35
       'h1_mbp_noninvasive_max':H1_mbp_noninvasive_max,
       'h1_mbp_noninvasive_min':H1_mbp_noninvasive_min,
       'h1_spo2_max':H1_spo2_max,
       'h1_spo2_min':H1_spo2_min,#37
       'h1_sysbp_max':H1_sysbp_max,
       'h1_sysbp_min':H1_sysbp_min,#39
       'd1_glucose_max':D1_glucose_max,
       'd1_glucose_min':D1_glucose_min,#41
       'd1_potassium_max':D1_potassium_max,
       'd1_potassium_min':D1_potassium_min,#43
       'apache_4a_hospital_death_prob' : Apache_4a_hospital_death_prob,
       'apache_4a_icu_death_prob':Apache_4a_icu_death_prob,#45
       'aids':Aids,
       'cirrhosis':Cirrhosis,
       'diabetes_mellitus':Diabetes,#47
       'hepatic_failure':Hepatic_failure,
       'immunosuppression':Immunosuppression,#49
       'leukemia':Leukemia,
       'lymphoma':Lymphoma,#51
       'solid_tumor_with_metastasis':Solid_tumor_with_metastasis,
       'apache_3j_bodysystem':Apache_3j_bodysystem,#53
       'apache_2_bodysystem':Apache_2_bodysystem#54
        }
    features = pd.DataFrame(data, index=[0])
    return features
    
    
data = patient_info()


st.write('---')
#Utilisation de LogisticRegression

model=LogisticRegression(C = 0.00026366508987303583,solver='liblinear'  )
model=model.fit(xtrain,ytrain)#

data = preprocessor.transform(data)

    #faire le prediction 
prediction = model.predict(data)

st.subheader('Will this patient survive according this model ?)')
if st.button('Prediction'):
    prediction = model.predict(data)
    if prediction == 0:
        st.success(f'The patient will not survive according to this model.')

    else:
        st.success(f'The patient will not survive according to this model.')




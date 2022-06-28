import streamlit as st
import pandas as pd
import numpy as np

# Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

st.write("""

# Final Project submission
#### Name = Syed Muhammad Talha Bin Ashraf
#### Matricola Number = VR479128


# Problem Description

It is important to know if a patient will be readmitted in some hospital. The reason is that you can change the treatment, in order to avoid a readmission.
 
In this database, you have 3 different outputs:
 
 - No readmission;
 - A readmission in less than 30 days (this situation is not good, because maybe your treatment was not appropriate);
 - A readmission in more than 30 days (this one is not so good as well the last one, however, the reason can be the state of the patient.

# Import Libraries
 - pandas 
 - Numpy 

#### Visualization Libraries
 - Seaborn 
 - Matplotlib.pyplot 
""")

st.write("""
# Read Data
""")
# ### 2. Read Data

st.write("""
#### File Name
     
 -  diabetic_data.csv
    
""")

df = pd.read_csv('diabetic_data.csv')

st.write("""
## Data Frame Head
""")

# df.head()
st.write(df.head())
st.write('5 rows × 50 columns')


st.write("""
# Data Analysis, Visualization and Cleaning

### Shape of the data ?
""")

st.write('The shape of the Dataset is :', df.shape, 'with', df.shape[0], 'records and', df.shape[1], 'columns')

st.write(""" #### Columns """)
st.write(df.columns)

st.write(""" ### Number of columns in the data?""")


st.write('There are total', len(df.columns), 'columns in the dataset')

# From the 50 columns 49 columns such as encounter_id, patient_nbr etc are the independent variables and the column name <b>"readmitted"</b> is the dependent variable and the label of the data. 

st.write(""" # Statistics of the Data ? """)

# In[16]:


df.describe(include = 'all').T

st.write(""" ### How many Null Values in Data? """)

st.write(""" The data contains some null values, but null values are filled with "?". so we will look for '?' in each column for null values. """)


for i in df.columns:
  st.write(i, df[df[i] == '?'].shape[0])



st.write("""

 We can see that there are many null values in the columns like "medical_specialty" , "race" and "payer_code". So we will have to fill these null values or drop the rows or columns with null values.  

 We start analyzing columns sequentially and will drill down the data to look for insights. We will look for Number of Patients in the data. AS we know we can check from the "patient_nbr" column that how many unique patients in the data.
""")

st.write('There are', len(df['patient_nbr'].unique()), 'unique "patients" in the data.')

st.write('There are', len(df['encounter_id'].unique()), 'unique "encounters" in the data.')

st.write("""  - Everytime the patient visits the hospital, it is called as "encounter". 
 - So we have multiple encounters per patient. 
 
 So we will take the problem as simple classification problem and didnt deal it with like a "Time Series" problem as we dont have much encounters per patient in the data.
  """)

st.write(""" ### Encounter per patient?  """)


st.write(""" If we divide total patient with total encounter, we can get the average encounters per patient.  """)


st.write(len(df['encounter_id'].unique())/len(df['patient_nbr'].unique()))

st.write(""" - So we have "1.4 encounters" per patient and majority of the patients will have only 1 encounter in the data.  """)


st.write(""" ###  Lets check this with the statistics.  """)


df_encounters_check = df.groupby(['patient_nbr']).agg(encounters = ('encounter_id', 'count')).reset_index().sort_values(['encounters'], ascending = False)

df_encounters_check[df_encounters_check['encounters']==1]


st.write("""  From the 71518 patients, 54745 patients have only 1 encounter in the data.
 Remaining patients have more than 1 encounter in the data. 
 So we concluded that we will only take data as simple data, not a Time Series Data.   """)

st.write(""" ### Lets analyze the label column ?  """)


st.write("""  First of All, Lets check the Distribution of Label column.   """)


df['readmitted'].value_counts()

fig = plt.figure(figsize=(10, 4))
ax = sns.barplot(x=df['readmitted'].value_counts().index,   y=df['readmitted'].value_counts())
plt.xlabel('labels', size = 12)
plt.ylabel('# of Readmitted', size = 12)
plt.title('Class Distribution \n', size = 12)
st.pyplot(fig)



st.write("""
 - As Approximately 50% of the data belongs to the  "NO" class, and other classes have less labels.
 - It will create class imbalance problem. So we will take this problem as 2 class problem.
 - We will only try to predict if the patinet will readmitted or Not, We will skip the part of less than 30 days or greater than 30 days.
""")

st.write(""" ### Created another column and take it as 2 class problem, Label the <30 and >30 as YES and Other N0 as No. """)

st.write(df['readmitted'].unique())


def check_label(text):
    if text == '>30' or text =='<30':
        return 'Yes'
    else:
        return 'No'
    
df['readmitted_2'] =df['readmitted'].apply(check_label)

fig = plt.figure(figsize=(10, 4))
ax = sns.countplot(x='readmitted_2',   data= df)
plt.xlabel('Readmitted', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.title('Distribution of Readmission Class  \n\n', size = 12)
st.pyplot(fig)


st.write(""" PLOT """)



st.write(""" # Race Column """)

st.write(""" According to Documentaiton the values for race can be: 
 
 - Caucasian 
 - Asian
 - African American  
 - Hispanic
 - other """)

fig = plt.figure(figsize=(10, 4))
ax = sns.barplot(x=df['race'].value_counts().index,   y=df['race'].value_counts())
plt.xlabel('Race', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.title('Distribution of Race of Patients \n', size = 12)
st.pyplot(fig)


st.write("""  The majority of the people are Caucasian, which are the people with european ancestry.
 - There are "?" in the data which means the race contains the Null values.
 - We will be needing to remove this from the data or we can also assign this with "Other" category.

""")


df.loc[df['race'] == '?', 'race'] = 'Other'

fig = plt.figure(figsize=(10, 4))
ax = sns.barplot(x=df['race'].value_counts().index,   y=df['race'].value_counts())
plt.xlabel('Race', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.title('Distribution of Race of Patients \n', size = 12)
st.pyplot(fig)



st.write(""" We replaced the Race containing value '?' with Other! """)
fig = plt.figure(figsize=(10, 4))
data = df['race'].value_counts().to_numpy()
labels = y=df['race'].unique()
colors = sns.color_palette('pastel')[0:5]
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
st.pyplot(fig)

st.write(""" # What is the Gender Distribution in Data? """)

st.write("""  According to Documentation, The values can be,
 - Male, 
 - Female  
 - Unknown/invalid
 """)



fig = plt.figure(figsize=(10, 4))
ax = sns.countplot(x='gender',   data= df)
plt.xlabel('Gender', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.title('Gender Distribution \n', size = 12)
st.pyplot(fig)


st.write("""  - We can see in the above figure that there are More than 50,000 Males in the data.
 - Females are close to 48,000.
 - There are some people whose gender is unknow, we can drop these rows as they are very few.
 """)

st.write(df['gender'].value_counts())

st.write("""  - There are only 3 Encounter for which we dont know the gender, It may create distribution error in the data. 
 - So it is better to drop these rows from the data """)

st.write(df[df['gender']!='Unknown/Invalid'])



st.write("""Drop the "Unknown/Invalid" gender of the data. """)

df.drop(df[df['gender'] == 'Unknown/Invalid'].index, inplace = True)

st.write(df.reset_index(inplace = True, drop = True))


st.write(df.head())

fig = plt.figure(figsize=(10, 4))
data = df['gender'].value_counts().to_numpy()
labels = y=df['gender'].unique()
colors = sns.color_palette('pastel')[0:5]
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
st.pyplot(fig)

st.write(' ### Relationship of Gender and Readmitted Overall ')


fig = plt.figure(figsize=(10, 4))
ax = sns.countplot(x="gender", hue="readmitted_2", data=df)
plt.xlabel('Gender', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.title('Gender vs Readmitted \n', size = 12)
st.pyplot(fig)


st.write(""" # What Age of People are there in data? """)


fig = plt.figure(figsize=(10, 4))
ax = sns.countplot(x='age',   data= df)
plt.xlabel('Age', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.title('Age Distribution \n', size = 12)
st.pyplot(fig)




# - As per the Literature, The problem of Readmission is common in Older People. 

st.write(""" ### RelationShip Between and Age and Readmission ? """)


fig = plt.figure(figsize=(10, 4))
sns.countplot(x="age", hue="readmitted_2", data=df)
st.pyplot(fig)


st.write(""" - As we mentioned above, The relationship of older Patients and Readmission is Strong as Mostly Older Patients are at high risk of Readmission.
  - And you can also see from the data the Mostly Older Patient are Readmitted, and younger people not tend to readmit.
 """)

st.write(""" # Lets Analyze Weight of the Patient ? """)



df.shape

st.write(df['weight'].value_counts())

st.write(""" - From value Counts We can see that the from around 101000 records, 98569 records dont have Weight Value. 
 - So, We will drop this column. 
 - If we will try to fill this column it can disturb the distribution of the data.
""")

st.write(""" Lets drop this column. """)


df.drop(columns = ['weight'], inplace = True)

st.write(""" # Understanding of Type of Admission of the Patient column. """)


st.write("""  As per the documentation, Integer identifier corresponding to 9 distinct values, for example:
 - emergency
 - urgent,
 - elective,
 - newborn,
 - not available

 This represents the Type of Admission of the Patient, Which means in which department patient if admitted to at the time of encounter. 
 
 As we dont have specific Id Defined even in the Documentation, we cannot map these value with Type for better undetstanding. 
 We will only see if which ID have most Encounters. """)


st.write(""" #plot  """)

fig = plt.figure(figsize=(10, 4))
sns.countplot(x='admission_type_id',   data= df)
plt.xlabel('Admission Type ID', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.title('Admission Type Id Distribution \n', size = 12)
st.pyplot(fig)

st.write(""" We can see in the above graph, The Id 1 have most of the encounters. From the literature review i assumed that the value should mean as Inpatient Encounter. Because mostly the Patients Admitted to the Inpatiet Department Readmitted after some Procedure. 
 """)

# <b>What is the Discharge Disposition ?</b>

# AS per the Documentation, Integer identifier corresponding to 29 distinct values, for example:
# - discharged to home
# - expired
# - not available 

# As per Literature, The Discharge Disposition means the facility to which patinet is discharged to. Patient can discharge to Home Health, etc. 

# In[46]:


len(df['discharge_disposition_id'].unique())


df['admission_source_id'].unique()  



print('There are', len(df['admission_source_id'].unique()), 'unique Admission Sources from which patient can be admitted.')


st.write(""" # What is meaning of time_in_hospital? """)

# 
st.write(""" As per Literaure, it is Integer number of days between admission and discharge.
 """)


st.write(df['time_in_hospital'].unique())


fig = plt.figure(figsize=(10, 4))
sns.countplot(x='time_in_hospital',   data= df)
plt.xlabel('Admission Type ID', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.title('Admission Type Id Distribution \n', size = 12)
st.pyplot(fig)


st.write(df['time_in_hospital'].mean())

st.write("""From the Graph and Mean of the Time in Hospital, We found that the majority of the people stays in hospital 3-4 Days.
""")

st.write(""" # What is the Relation of Stay in Hospital and Readmission? """)


sns.set(rc={'figure.figsize':(18,8.2)})
fig = plt.figure(figsize=(10, 4))
sns.countplot(x='time_in_hospital',  hue= 'readmitted_2',  data= df)
plt.xlabel('Time In Hospital', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Readmitted Count', size = 12)
plt.title('Time in Hospital vs Readmission \n', size = 12)
st.pyplot(fig)


st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure()
plt.title('Relationship between Time in Hospital and Readmission \n\n', size  = 14)
sns.set_style('darkgrid')
sns.displot(df, x="time_in_hospital", hue = 'readmitted_2', kind="kde")
st.pyplot()


st.write(""" Normal Time in Hospital for Not Readmitted and Readmitted is the same. This means that this parameter will not add value In our model.
  """)

df['payer_code'].value_counts()



df.drop(columns = ['payer_code'], inplace = True)

st.write(""" # What is medical Speciality?""")

st.write("""  Integer identifier of a specialty of the admitting physician, corresponding to 84 distinct
values, for example: 
 - cardiology
 - internal medicine 
 - family\general practice
 - surgeon""")



sns.set(rc={'figure.figsize':(18,8.2)})
fig = plt.figure(figsize=(10, 4))
ax = sns.countplot(x='medical_specialty',   data= df)
plt.xlabel('Medical Speciality', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.title('Medical Speciality Distribution \n', size = 12)
st.pyplot(fig)

st.write("""  By looking at the graph we can see that there are also many missing values in the data. We will remove this column.
 - As managing this colum with so many missing values will be not easy. """)


df.drop(columns =['medical_specialty'], inplace = True)

st.write(""" # What is num_lab_procedures """)


st.write(""" Number of lab tests performed during the encounter  """)


st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure()
plt.title('Distribution of Lab Procedures \n\n', size = 13)
sns.set_style('darkgrid')
sns.displot(df, x="num_lab_procedures", kind="kde")
st.pyplot()



st.write(""" -  As we can see that from the distribution plot. That the majority of the Patients have around 30 to 50 Labs Procedures. Lets look at it with respect to class.   
  """)


st.write(""" ### Trend of Lab Procedures with Readmission ?""")


st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure()
plt.title('Realtionship of Lab Procedures with Readmission \n\n', size = 13)
sns.set_style('darkgrid')
sns.displot(df, x="num_lab_procedures", hue= 'readmitted_2', kind="kde")
st.pyplot()


st.write(""" - The Distribution of Readmitted and Not Readmitted have the same trend.
 - The number of labs procedures will not play a vital role in creating contrastive behaviour between Readmitted and Not Readmitted.""")



st.write(""" # What is the relation of Number of Procedures and Readmission? """)

st.write(""" Number of procedures (other than lab tests) performed during the encounter
  """)

sns.set(rc={'figure.figsize':(18,8.2)})
fig = plt.figure(figsize=(10, 4))
ax = sns.countplot(x='num_procedures',  hue= 'readmitted_2',  data= df)
plt.xlabel('Number of Procedures', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Readmitted Count', size = 12)
plt.title('Number of Procedures vs Readmission \n', size = 14)
st.pyplot(fig)

st.write(""" - Number of Procedures is also not giving some vital signs of readmission with increase in procedure. 
- Majority of patients have 0 procedures which are can be Readmitted and not Readmitted
  """)


st.write(""" # What is the trend of Number of Emergency Visits ? """)

st.write(""" Number of emergency visits of the patient in the year preceding the encounter
 """)


st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure()
plt.title('Relation of Emergency Visits w.r.t Readmission \n\n', size = 13)
sns.set_style('darkgrid')
sns.displot(df, x="number_emergency", hue= 'readmitted_2', kind='kde')
st.pyplot()

st.write(""" - We can see that the distribution of Emergnecy Visits very Skewed.
 - Majority of the Patients have 0 Emergency `Visits. 
 - We will slice the data and look for trend in detail.
  """)

st.write(' ### What is relation when we look at Emergency Visits less than 5 ?' )



st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure()
plt.title('Relationship of Emergency Visits < 5 w.r.t Readmission \n\n', size = 13)
sns.set_style('darkgrid')
sns.displot(df.loc[df['number_emergency']<5], x="number_emergency", hue= 'readmitted_2', kind='kde')
st.pyplot()


st.write(""" - When the value is at 0 the number of Not Readmitted are higher than the Readmitted Patients. 
 - Now lets look at the patients with readmission greater than equal to 5.
""")


st.write(' ### What is relation when we look at Emergency Visits greater than 5 ? ')


st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure()
plt.title('Relationship of Emergency Visits >= 5 w.r.t Readmission \n\n', size = 13)
sns.set_style('darkgrid')
sns.displot(df.loc[df['number_emergency']>=5], x="number_emergency", hue= 'readmitted_2', kind='kde')
st.pyplot()


st.write(""" - We can see that The majority of the Encouters have Number of Readmission Visits nea 10 and they are Readmitted to hospital.
 - We can conclude that, if the Numer of emergency Visits Increased the Patient Most likely to readmit to the hospital.
""")


st.write(""" # Top 20 Diagnosis in the Readmitted = YES """)



fig = plt.figure(figsize=(10, 4))
ax = sns.barplot(x=df[df['readmitted_2'] == 'Yes']['diag_1'].value_counts().index[:20],
                 y=df[df['readmitted_2'] == 'Yes']['diag_1'].value_counts()[:20])
plt.xlabel('Primary Diagnosis Codes', size = 12)
plt.ylabel('Count', size = 12)
plt.title('Top 20 Primary Diagnosis Codes in Readmission = YES \n', size = 12)
st.pyplot(fig)


st.write(""" The Top Diagnosis Codes are 428, 414 and 786 in the Readmitted Patients.
 If we look at the ICD-9 Dictionary we will know that,
 - 428 = Congestive heart failure
 - 414 = Ischemic heart disease
 - 786 = Symptoms involving respiratory system and other chest symptoms
 - 486 = Pneumonia, organism unspecified 

 So Patients with Heart Disease and Chest Disease are more likely to readmit to the hospital.
""")


fig = plt.figure(figsize=(10, 4))
ax = sns.barplot(x=df[df['readmitted_2'] == 'No']['diag_1'].value_counts().index[:20],
                 y=df[df['readmitted_2'] == 'No']['diag_1'].value_counts()[:20])
plt.xlabel('Primary Diagnosis Codes', size = 12)
plt.ylabel('Count', size = 12)
plt.title('Top 20 Primary Diagnosis Codes in Readmission = No \n', size = 12)
st.pyplot(fig)


st.write(""" We can see from graph, Chest and Heart Diseases are also common in Patients who didnt Admitted.
""")


st.write(""" # What is change? """)

st.write(""" Indicates if there was a change in diabetic medications (either dosage or generic
 name). Values:
 
 - “change”  
 - “no change” """)


st.write(df['change'].value_counts())


fig = plt.figure(figsize=(10, 4))
ax = sns.countplot(x='change',  hue= 'readmitted_2',  data= df)
plt.xlabel('Diabetes Med', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Readmitted Count', size = 12)
plt.title('Diabetes Med vs Readmission \n', size = 12)
st.pyplot(fig)

st.write(""" # What is Diabetes Medication? """)

st.write("""  Indicates if there was any diabetic medication prescribed. Values:
 
 - “yes”
 - “no”
""")


fig = plt.figure(figsize=(10, 4))
ax = sns.countplot(x='diabetesMed',  hue= 'readmitted_2',  data= df)
plt.xlabel('Diabetes Med', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Readmitted Count', size = 12)
plt.title('Diabetes Med vs Readmission \n', size = 12)
st.pyplot(fig)

st.write(""" - From above figure we can see that, the Patient with Diabetes have the high amount of readmissions.
 """)

st.write(""" # Features for Medications? """)

st.write(""" For the generic names: 
 1. metformin 
 2. repaglinide 
 3. nateglinide
 4. chlorpropamide
 5. glimepiride
 6. acetohexamide
 7. glipizide
 8. glyburide
 9. tolbutamide 
 10. pioglitazone
 11. rosiglitazone
 12. acarbose
 13. miglitol
 14. troglitazone
 15. tolazamide
 16. examide
 17. sitagliptin
 18. insulin
 19. glyburide-metformin
 20. glipizide-metformin
 21. glimepiride-pioglitazone
 22. metformin-rosiglitazone
 23. metformin-pioglitazone
 
 The feature indicates whether the drug was prescribed or there was a change in the dosage. Values:
 
 - “up” if the dosage was increased during the encounter
 - “down” if the dosage was decreased
 - “steady” if the dosage did not change
 - “no” if the drug was not prescribed """)


st.write(""" Lets Analyze Distribution of each value in these columns! PLOT  """)



medStats = []
for i in df.iloc[:, 21:44].columns:
    labels = y=df[i].unique()
#     df[i]=df[i].replace("None", "No")
#     df[i]=df[i].replace("Norm", "Steady")
#     df[i]=df[i].replace(">300", "Up")
#     df[i]=df[i].replace(">200", "Down")
#     df[i]=df[i].replace(">7", "Down")
#     df[i]=df[i].replace(">8", "Up")
    # print('Name: ', i, '\t,Labels: ',labels)
    steady =(df[i]=='Steady').value_counts()[True] if 'Steady' in df[i].unique() else 0
    up = (df[i]=='Up').value_counts()[True] if 'Up' in df[i].unique() else 0
    down = (df[i]=='Down').value_counts()[True] if 'Down' in df[i].unique() else 0
    no = (df[i]=='No').value_counts()[True] if 'No' in df[i].unique() else 0
    medStats.append({'name':i , 'steady':steady, 'up':up, 'down':down, 'no':no})
    # print('Name:', i,'\t,Steady:', steady, '\t,Up:', up, '\t,Down:', down, '\t,No:', no)
# print(medStats)

graphs = ['steady' , 'up' , 'down' , 'no']
for graph in graphs:
    labels =[]
    values =[]
    fig = plt.figure(figsize=(10, 4))
    for i in medStats:
        labels.append(i['name'])
        values.append(i[graph])
        colors = sns.color_palette('pastel')[0:23]
        plt.bar(labels, values , color = colors)
        plt.xticks(rotation=90)
        # add graph title
        plt.title('Medication ' + graph)
        # plt.title('Medication Graphs')
        plt.ylabel('Count')
        plt.xlabel('Medication')
    st.pyplot(fig)





st.write(""" - From the above count plots, we can see that majority of the Medicines are not assigned to patients.
 - If one is assigned then it is assigned to very few people.
  """)

# st.write("""# Analyze Medicines with Class Variable Readmission """)

# for columnName in df.iloc[:, 21:44].columns:
#   fig = plt.figure(figsize=(10, 4))
#   g = sns.FacetGrid(df, col=columnName)
#   g.map(sns.histplot, "readmitted_2")
# #     plt.title(str(columnName) + 'vs Readmission', size = 13)
#   st.pyplot(fig)


st.write(""" ## Dropping Columns with almost no Information """)


st.write(df.drop(columns = ['acetohexamide', 'tolbutamide', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
                   'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                   'metformin-pioglitazone'], inplace = True))



st.write(df.shape)

st.write(""" ### Drop Diagnosis Codes with empty values""")


st.write(""" - As we have found Null values in the data, the diagnosis codes are not availale in the around 1500 rows.
 - So we will drop these rows.""")

df = df[~((df['diag_1'] == "?") | (df['diag_2'] == "?") | (df['diag_3'] == "?"))]


st.write(df.shape)

# df.to_csv('PreparedData.csv')


# In the start of the Analysis, we had 50 columns and 12 of them are dropped from the dataset as they didnt provide any useful information.

# In[91]:


# Make copy of data.
df_ = df.copy()


# ### 4. Transform the Categorical Features

# In[92]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()

st.write(""" # Transform Categorical Features """)
# <b> Transform Categorical Features </b>

# In[ ]:


categorical_features =['race', 'gender', 'age',
       'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
       'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
       'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin',
       'glyburide-metformin', 'change', 'diabetesMed'] 

for i in categorical_features:
    df_[i] = le.fit_transform(df_[i])


st.write(df_.head())
df_.head()


# Now we can see that in dataframe that the categorical values are encoded.

# <b> Transform Label Columns </b>

# In[ ]:


label = le.fit(df_['readmitted_2'])


# In[ ]:


df_['readmitted_2_encoded'] = label.transform(df_['readmitted_2'])


# After Label Encoding the values assigned to class values are :
#
# - 0 as No
# - 1 as yes

st.write(""" # Features Correaltion """)

# ### 5. Features Correaltion

# In[ ]:

# st.write(df_ = df_.drop(columns= ['encounter_id', 'patient_nbr', 'readmitted','readmitted_2']))

df_ = df_.drop(columns= ['encounter_id', 'patient_nbr', 'readmitted','readmitted_2'])



# In[ ]:


df_

st.write(""" ### Correlatiom between Numeical Features""")
# <b>Correlatiom between Numeical Features</b>

# In[ ]:

st.write(df_[['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
   'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']].corr())
# df_[['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
#     'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']].corr()


# In[ ]:

# st.write(df_.columns)

df_.columns


# #### Split the Dependednt and Independent Variables

# In[ ]:


X = df_.drop(columns= ['readmitted_2_encoded'])
Y = df_['readmitted_2_encoded']


# ### 6. Feature Scaling

# In[ ]:


from sklearn import preprocessing
scaled_X = preprocessing.StandardScaler().fit_transform(X)


# ### 7. Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(scaled_X, Y, test_size=0.25, random_state=42)





X_train.shape, X_test.shape, y_train.shape, y_test.shape

st.write(""" # Machine Learning Modeling """)

# <b>Import Libraries for Evaluation of the Models</b>

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix

st.write(""" ### Logistic Regression """)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
# Training
lr.fit(X_train, y_train)

lr_prediction = lr.predict(X_test)



st.write(classification_report(y_test, lr_prediction))


fig = plt.figure(figsize=(10, 4))
ax = sns.heatmap(confusion_matrix(y_test, lr_prediction), annot=True, fmt='', cmap='Blues')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
ax.set_title('Confusion Matrix of Logistic Regression \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
st.pyplot(fig)

st.write(""" ### Random Forest Classifier """)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 450, max_depth=9, random_state=43)
rf.fit(X_train, y_train)


rf_prediction =  rf.predict(X_test)


st.write(classification_report(y_test, rf_prediction))


fig = plt.figure(figsize=(10, 4))
ax = sns.heatmap(confusion_matrix(y_test, rf_prediction), annot=True, fmt='', cmap='Blues')
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
ax.set_title('Confusion Matrix of Random Forest \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
st.pyplot(fig)








#!/usr/bin/env python
# coding: utf-8

# ## Income Qualification
# 
# ## DESCRIPTION
# 
# Identify the level of income qualification needed for the families in Latin America.
# 
# ## Problem Statement Scenario:
# 
# Many social programs have a hard time ensuring that the right people are given enough aid. It’s tricky when a program focuses on the poorest segment of the population. This segment of the population can’t provide the necessary income and expense records to prove that they qualify.
# 
# In Latin America, a popular method called Proxy Means Test (PMT) uses an algorithm to verify income qualification. With PMT, agencies use a model that considers a family’s observable household attributes like the material of their walls and ceiling or the assets found in their homes to classify them and predict their level of need.
# 
# While this is an improvement, accuracy remains a problem as the region’s population grows and poverty declines.
# 
# The Inter-American Development Bank (IDB)believes that new methods beyond traditional econometrics, based on a dataset of Costa Rican household characteristics, might help improve PMT’s performance.
# 
# 
# 
# 
# 
# 

# ## Analysis Tasks to be performed:
#     
#     Identify the output variable.
#     Understand the type of data.
#     Check if there are any biases in your dataset.
#     Check whether all members of the house have the same poverty level.
#     Check if there is a house without a family head.
#     Set poverty level of the members and the head of the house within a family.
#     Count how many null values are existing in columns.
#     Remove null value rows of the target variable.
#     Predict the accuracy using random forest classifier.
#     Check the accuracy using random forest with cross validation.

# ## Core Data fields
#         Id - a unique identifier for each row.
#         Target - the target is an ordinal variable indicating groups of income levels.
#         1 = extreme poverty 2 = moderate poverty 3 = vulnerable households 4 = non vulnerable households
#         idhogar - this is a unique identifier for each household. This can be used to create household-wide features, etc. All rows in a given household will have a matching value for this identifier.
#         parentesco1 - indicates if this person is the head of the household.
# 
# ## Note: that ONLY the heads of household are used in scoring. All household members are included in train +test data sets, but only heads of households are scored.
# 
# ## Step 1: Understand the Data
# 1.1 Import necessary Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


import warnings
warnings.filterwarnings('ignore')


# In[3]:


# 1.2 Load Data
df_income_train = pd.read_csv("train.csv")
df_income_test =  pd.read_csv("test.csv")


# In[4]:


# 1.3 Explore Train dataset
# View first 5 records of train dataset
df_income_train.head()


# In[5]:


df_income_train.info()


# In[6]:


# 1.4 Explore Test dataset
# View first 5 records of test dataset

df_income_test.head()


# In[7]:


df_income_test.info()


# ## Looking at the train and test dataset we noticed that the following:
# ## Train dataset:
# Rows: 9557 entries, 0 to 9556
# Columns: 143 entries, Id to Target
# Column dtypes: float64(8), int64(130), object(5)
# 
# ## Test dataset:
# Rows: 23856 entries, 0 to 23855
# Columns: 142 entries, Id to agesq
# dtypes: float64(8), int64(129), object(5)
# 
# The important piece of information here is that we don’t have ‘Target’ feature in Test Dataset. There are 5 object type, 130(Train set)/ 129 (test set) integer type and 8 float type features. Lets look at those features next.

# In[8]:


#List the columns for different datatypes:
print('Integer Type: ')
print(df_income_train.select_dtypes(np.int64).columns)
print('\n')
print('Float Type: ')
print(df_income_train.select_dtypes(np.float64).columns)
print('\n')
print('Object Type: ')
print(df_income_train.select_dtypes(np.object).columns)


# In[9]:


df_income_train.select_dtypes('int64').head()


# ## 1.5 Find columns with null values

# In[10]:


#Find columns with null values
null_counts=df_income_train.select_dtypes('int64').isnull().sum()
null_counts[null_counts > 0]


# In[11]:


df_income_train.select_dtypes('float64').head()


# In[12]:


#Find columns with null values
null_counts=df_income_train.select_dtypes('float64').isnull().sum()
null_counts[null_counts > 0]


# In[13]:


df_income_train.select_dtypes('object').head()


# In[14]:


#Find columns with null values
null_counts=df_income_train.select_dtypes('object').isnull().sum()
null_counts[null_counts > 0]


# ## Looking at the different types of data and null values for each feature. We found the following: 
# 
# 1. No null values for Integer type features. 
# 
# 2. No null values for float type features. 
# 
# 3. For Object types v2a1 6860 v18q1 7342 rez_esc 7928 meaneduc 5 SQBmeaned 5

# We also noticed that object type features dependency, edjefe, edjefa have mixed values.
# Lets fix the data for features with null values and features with mixed values

# ## Step 3: Data Cleaning

# ## 3.1 Lets fix the column with mixed values.
# According to the documentation for these columns:
# dependency: Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
# edjefe: years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# edjefa: years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
# 
# For these three variables, it seems “yes” = 1 and “no” = 0. We can correct the variables using a mapping and convert to floats.

# In[15]:


mapping={'yes':1,'no':0}

for df in [df_income_train, df_income_test]:
    df['dependency'] =df['dependency'].replace(mapping).astype(np.float64)
    df['edjefe'] =df['edjefe'].replace(mapping).astype(np.float64)
    df['edjefa'] =df['edjefa'].replace(mapping).astype(np.float64)
    
df_income_train[['dependency','edjefe','edjefa']].describe()


# ## 3.2 Lets fix the column with null values
# According to the documentation for these columns:
# 
# v2a1 (total nulls: 6860) : Monthly rent payment
# v18q1 (total nulls: 7342) : number of tablets household owns
# rez_esc (total nulls: 7928) : Years behind in school
# meaneduc (total nulls: 5) : average years of education for adults (18+)
# SQBmeaned (total nulls: 5) : square of the mean years of education of adults (>=18) in the household 142

# In[16]:


# 1. Lets look at v2a1 (total nulls: 6860) : Monthly rent payment 
# why the null values, Lets look at few rows with nulls in v2a1
# Columns related to  Monthly rent payment
# tipovivi1, =1 own and fully paid house
# tipovivi2, "=1 own,  paying in installments"
# tipovivi3, =1 rented
# tipovivi4, =1 precarious 
# tipovivi5, "=1 other(assigned,  borrowed)"


# In[17]:


data = df_income_train[df_income_train['v2a1'].isnull()].head()

columns=['tipovivi1','tipovivi2','tipovivi3','tipovivi4','tipovivi5']
data[columns]


# In[18]:


# Variables indicating home ownership
own_variables = [x for x in df_income_train if x.startswith('tipo')]

# Plot of the home ownership variables for home missing rent payments
df_income_train.loc[df_income_train['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),
                                                                        color = 'green',
                                                              edgecolor = 'k', linewidth = 2);
plt.xticks([0, 1, 2, 3, 4],
           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],
          rotation = 20)
plt.title('Home Ownership Status for Households Missing Rent Payments', size = 18);


# In[19]:


#Looking at the above data it makes sense that when the house is fully paid, there will be no monthly rent payment.
#Lets add 0 for all the null values.
for df in [df_income_train, df_income_test]:
    df['v2a1'].fillna(value=0, inplace=True)

df_income_train[['v2a1']].isnull().sum()   


# In[20]:


# 2. Lets look at v18q1 (total nulls: 7342) : number of tablets household owns 
# why the null values, Lets look at few rows with nulls in v18q1
# Columns related to  number of tablets household owns 
# v18q, owns a tablet


# In[21]:


# Since this is a household variable, it only makes sense to look at it on a household level, 
# so we'll only select the rows for the head of household.

# Heads of household
heads = df_income_train.loc[df_income_train['parentesco1'] == 1].copy()
heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())


# In[22]:


plt.figure(figsize = (8, 6))
col='v18q1'
df_income_train[col].value_counts().sort_index().plot.bar(color = 'blue',
                                             edgecolor = 'k',
                                             linewidth = 2)
plt.xlabel(f'{col}'); plt.title(f'{col} Value Counts'); plt.ylabel('Count')
plt.show();


# In[23]:


#Looking at the above data it makes sense that when owns a tablet column is 0, there will be no number of tablets household owns.
#Lets add 0 for all the null values.
for df in [df_income_train, df_income_test]:
    df['v18q1'].fillna(value=0, inplace=True)

df_income_train[['v18q1']].isnull().sum()


# In[24]:


# 3. Lets look at rez_esc    (total nulls: 7928) : Years behind in school  
# why the null values, Lets look at few rows with nulls in rez_esc
# Columns related to Years behind in school 
# Age in years

# Lets look at the data with not null values first.
df_income_train[df_income_train['rez_esc'].notnull()]['age'].describe()


# In[25]:


#From the above , we see that when min age is 7 and max age is 17 for Years, then the 'behind in school' column has a value.
#Lets confirm
df_income_train.loc[df_income_train['rez_esc'].isnull()]['age'].describe()


# In[26]:


df_income_train.loc[(df_income_train['rez_esc'].isnull() & ((df_income_train['age'] > 7) & (df_income_train['age'] < 17)))]['age'].describe()
#There is one value that has Null for the 'behind in school' column with age between 7 and 17 


# In[27]:


df_income_train[(df_income_train['age'] ==10) & df_income_train['rez_esc'].isnull()].head()
df_income_train[(df_income_train['Id'] =='ID_f012e4242')].head()
#there is only one member in household for the member with age 10 and who is 'behind in school'. This explains why the member is 
#behind in school.


# In[28]:


#from above we see that  the 'behind in school' column has null values 
# Lets use the above to fix the data
for df in [df_income_train, df_income_test]:
    df['rez_esc'].fillna(value=0, inplace=True)
df_income_train[['rez_esc']].isnull().sum()


# In[29]:


#Lets look at meaneduc   (total nulls: 5) : average years of education for adults (18+)  
# why the null values, Lets look at few rows with nulls in meaneduc
# Columns related to average years of education for adults (18+)  
# edjefe, years of education of male head of household, based on the interaction of escolari (years of education),
#    head of household and gender, yes=1 and no=0
# edjefa, years of education of female head of household, based on the interaction of escolari (years of education), 
#    head of household and gender, yes=1 and no=0 
# instlevel1, =1 no level of education
# instlevel2, =1 incomplete primary


# In[30]:


data = df_income_train[df_income_train['meaneduc'].isnull()].head()

columns=['edjefe','edjefa','instlevel1','instlevel2']
data[columns][data[columns]['instlevel1']>0].describe()


# In[31]:


#from the above, we find that meaneduc is null when no level of education is 0
#Lets fix the data
for df in [df_income_train, df_income_test]:
    df['meaneduc'].fillna(value=0, inplace=True)
df_income_train[['meaneduc']].isnull().sum()


# In[32]:


#Lets look at SQBmeaned  (total nulls: 5) : square of the mean years of education of adults (>=18) in the household 142  
# why the null values, Lets look at few rows with nulls in SQBmeaned
# Columns related to average years of education for adults (18+)  
# edjefe, years of education of male head of household, based on the interaction of escolari (years of education),
#    head of household and gender, yes=1 and no=0
# edjefa, years of education of female head of household, based on the interaction of escolari (years of education), 
#    head of household and gender, yes=1 and no=0 
# instlevel1, =1 no level of education
# instlevel2, =1 incomplete primary 


# In[33]:


data = df_income_train[df_income_train['SQBmeaned'].isnull()].head()

columns=['edjefe','edjefa','instlevel1','instlevel2']
data[columns][data[columns]['instlevel1']>0].describe()


# In[34]:


#from the above, we find that SQBmeaned is null when no level of education is 0
#Lets fix the data
for df in [df_income_train, df_income_test]:
    df['SQBmeaned'].fillna(value=0, inplace=True)
df_income_train[['SQBmeaned']].isnull().sum()


# In[35]:


#Lets look at the overall data
null_counts = df_income_train.isnull().sum()
null_counts[null_counts > 0].sort_values(ascending=False)


# ## 3.3 Lets look at the target column
# Lets see if records belonging to same household has same target/score.

# In[36]:


# Groupby the household and figure out the number of unique values
all_equal = df_income_train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))


# In[37]:


#Lets check one household
df_income_train[df_income_train['idhogar'] == not_equal.index[0]][['idhogar', 'parentesco1', 'Target']]


# In[38]:


#Lets use Target value of the parent record (head of the household) and update rest. But before that lets check
# if all families has a head. 

households_head = df_income_train.groupby('idhogar')['parentesco1'].sum()

# Find households without a head
households_no_head = df_income_train.loc[df_income_train['idhogar'].isin(households_head[households_head == 0].index), :]

print('There are {} households without a head.'.format(households_no_head['idhogar'].nunique()))


# In[39]:


# Find households without a head and where Target value are different
households_no_head_equal = households_no_head.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
print('{} Households with no head have different Target value.'.format(sum(households_no_head_equal == False)))


# In[40]:


#Lets fix the data
#Set poverty level of the members and the head of the house within a family.
# Iterate through each household
for household in not_equal.index:
    # Find the correct label (for the head of household)
    true_target = int(df_income_train[(df_income_train['idhogar'] == household) & (df_income_train['parentesco1'] == 1.0)]['Target'])
    
    # Set the correct label for all members in the household
    df_income_train.loc[df_income_train['idhogar'] == household, 'Target'] = true_target
    
    
# Groupby the household and figure out the number of unique values
all_equal = df_income_train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))


# ## 3.3 Lets check for any bias in the dataset

# In[41]:


#Lets look at the dataset and plot head of household and Target
# 1 = extreme poverty 2 = moderate poverty 3 = vulnerable households 4 = non vulnerable households 
target_counts = heads['Target'].value_counts().sort_index()
target_counts


# In[42]:


target_counts.plot.bar(figsize = (8, 6),linewidth = 2,edgecolor = 'k',title="Target vs Total_Count")


# In[44]:


# extreme poverty is the smallest count in the train dataset. The dataset is biased.


# ## 3.4 Lets look at the Squared Variables
#     ‘SQBescolari’
#     ‘SQBage’
#     ‘SQBhogar_total’
#     ‘SQBedjefe’
#     ‘SQBhogar_nin’
#     ‘SQBovercrowding’
#     ‘SQBdependency’
#     ‘SQBmeaned’
#     ‘agesq’

# In[45]:


#Lets remove them
print(df_income_train.shape)
cols=['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 
        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']


for df in [df_income_train, df_income_test]:
    df.drop(columns = cols,inplace=True)

print(df_income_train.shape)


# In[46]:


id_ = ['Id', 'idhogar', 'Target']

ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone']

ind_ordered = ['rez_esc', 'escolari', 'age']

hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']


# In[47]:


#Check for redundant household variables
heads = df_income_train.loc[df_income_train['parentesco1'] == 1, :]
heads = heads[id_ + hh_bool + hh_cont + hh_ordered]
heads.shape


# In[48]:


# Create correlation matrix
corr_matrix = heads.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop


# In[49]:


corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9]


# In[50]:


sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9],
            annot=True, cmap = plt.cm.Accent_r, fmt='.3f');


# In[51]:


# There are several variables here having to do with the size of the house:
# r4t3, Total persons in the household
# tamhog, size of the household
# tamviv, number of persons living in the household
# hhsize, household size
# hogar_total, # of total individuals in the household
# These variables are all highly correlated with one another.


# In[52]:


cols=['tamhog', 'hogar_total', 'r4t3']
for df in [df_income_train, df_income_test]:
    df.drop(columns = cols,inplace=True)

df_income_train.shape


# In[53]:


#Check for redundant Individual variables
ind = df_income_train[id_ + ind_bool + ind_ordered]
ind.shape


# In[54]:


# Create correlation matrix
corr_matrix = ind.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]

to_drop


# In[55]:


# This is simply the opposite of male! We can remove the male flag.
for df in [df_income_train, df_income_test]:
    df.drop(columns = 'male',inplace=True)

df_income_train.shape    


# In[56]:


#lets check area1 and area2 also
# area1, =1 zona urbana 
# area2, =2 zona rural 
#area2 redundant because we have a column indicating if the house is in a urban zone

for df in [df_income_train, df_income_test]:
    df.drop(columns = 'area2',inplace=True)

df_income_train.shape


# In[57]:


#Finally lets delete 'Id', 'idhogar'
cols=['Id','idhogar']
for df in [df_income_train, df_income_test]:
    df.drop(columns = cols,inplace=True)

df_income_train.shape


# ## Step 4: Predict the accuracy using random forest classifier.

# In[58]:


x_features=df_income_train.iloc[:,0:-1]
y_features=df_income_train.iloc[:,-1]
print(x_features.shape)
print(y_features.shape)


# In[59]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report

x_train,x_test,y_train,y_test=train_test_split(x_features,y_features,test_size=0.2,random_state=1)
rmclassifier = RandomForestClassifier()


# In[60]:


rmclassifier.fit(x_train,y_train)


# In[61]:


RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)


# In[62]:


y_predict = rmclassifier.predict(x_test)


# In[63]:


print(accuracy_score(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))


# In[64]:


y_predict_testdata = rmclassifier.predict(df_income_test)


# In[65]:


y_predict_testdata


# ## Step 6: Check the accuracy using random forest with cross validation.

# In[66]:


from sklearn.model_selection import KFold,cross_val_score


# ## 6.1 Checking the score using default 10 trees

# In[67]:


seed=7
kfold=KFold(n_splits=5,random_state=seed,shuffle=True)

rmclassifier=RandomForestClassifier(random_state=10,n_jobs = -1)
print(cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy'))
results=cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy')
print(results.mean()*100)


# ## 6.2 Checking the score using 100 trees

# In[68]:


num_trees= 100

rmclassifier=RandomForestClassifier(n_estimators=100, random_state=10,n_jobs = -1)
print(cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy'))
results=cross_val_score(rmclassifier,x_features,y_features,cv=kfold,scoring='accuracy')
print(results.mean()*100)


# In[70]:


rmclassifier.fit(x_train,y_train)


# In[71]:


y_predict_testdata = rmclassifier.predict(df_income_test)
y_predict_testdata


# Looking at the accuracy score, RandomForestClassifier with cross validation has the highest accuracy score of 94.60%
# ## To get a better sense of what is going on inside the RandomForestClassifier model, lets visualize how our model uses the different features and which features have greater effect.

# In[72]:


rmclassifier.fit(x_features,y_features)
labels = list(x_features)
feature_importances = pd.DataFrame({'feature': labels, 'importance': rmclassifier.feature_importances_})
feature_importances=feature_importances[feature_importances.importance>0.015]
feature_importances.head()


# In[73]:


feature_importances.sort_values(by=['importance'], ascending=True, inplace=True)
feature_importances['positive'] = feature_importances['importance'] > 0
feature_importances.set_index('feature',inplace=True)
feature_importances.head()

feature_importances.importance.plot(kind='barh', figsize=(11, 6),color = feature_importances.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')


# ## From the above figure, meaneduc,dependency,overcrowding has significant influence on the model.
# 
# Reference: Jhimli Bora - https://jhimlib.github.io/IncomeQualification/
# 

#!/usr/bin/env python
# coding: utf-8

# Banking Dataset Lead Scoring

# Importing Libraries

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt  #Visualization
import seaborn as sns


# In[2]:


df=pd.read_csv('train.csv')


# In[3]:


df.columns


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:

#organizing null entries
nullCheck=pd.DataFrame()
nullCheck['Number of null values']=df.isnull().sum()
nullCheck['Percentage of Null Values']=(df.isnull().sum() / df.shape[0]) * 100
nullCheck=nullCheck.sort_values('Percentage of Null Values',ascending=False)
nullCheck


# loan amount and loan period = 0 for null values

# In[8]:

#fillna(0) - fill null entries with 0
df['Loan_Amount']=df['Loan_Amount'].fillna(0)
df['Loan_Period']=df['Loan_Period'].fillna(0)


# loan amount non zero       interest rate null

# In[9]:

#examining interest rates
df['Interest_Rate'].unique()


# In[10]:


df2=df.iloc[:,7:10]
df3=df.iloc[ :,16:19] #loan period
df2=pd.concat([df2,df3],axis='columns')

#df3 - records with not-null interest rates
df3=df2[df2['Interest_Rate'].isnull()==False]
df3=df3.dropna()


# In[11]:

'''INTERNAL MODEL TO PREDICT INTEREST RATES'''
Y=df3['Interest_Rate']
X=df3.drop(['Interest_Rate'],axis='columns')


# In[12]:


X


# In[13]:


Y


# In[14]:


# X=X.dropna()


# In[15]:

#ONE-HOT ENCODING FOR CATEGORICAL VALUES
dummies1=pd.get_dummies(X['Employer_Category1'])
dummies1=dummies1.drop(dummies1.columns[0],axis='columns')
X=X.drop(['Employer_Category1'],axis='columns')
X=pd.concat([X,dummies1],axis='columns')


# In[16]:


X


# In[17]:


from xgboost import XGBRegressor
xgbr=XGBRegressor()


# In[18]:


xgbr.fit(X,Y)


# In[19]:


df4=df2[df2['Interest_Rate'].isnull()==True]
df5=df4[df4['Loan_Amount']!=0]


# In[20]:


df5


# In[21]:


df2


# In[22]:


X_test=df5.drop(['Interest_Rate'],axis='columns')

dummies2=pd.get_dummies(X_test['Employer_Category1'])
dummies2=dummies2.drop(dummies2.columns[0],axis='columns')
X_test=X_test.drop(['Employer_Category1'],axis='columns')
X_test=pd.concat([X_test,dummies2],axis='columns')


# In[23]:


X_test


# In[24]:


Y_predicted=xgbr.predict(X_test)


# In[25]:


df


# In[26]:


df['Interest_Rate']=df['Interest_Rate'].fillna(-1)
df['EMI']=df['EMI'].fillna(-1)


# In[27]:


def fill_ir(df,Y_predicted,x):
    for i in range(df.shape[0]):  #16 and 18
        la=df.iloc[i,16]
        #df.iloc[i,18]   ir
        if(la==0):
          df.iloc[i,18]=0    
        elif ((la!=0) and (df.iloc[i,18]==-1)):
          df.iloc[i,18]=Y_predicted[x]
          x+=1
    df['Interest_Rate']=df['Interest_Rate'].round(1)


# In[28]:


fill_ir(df,Y_predicted,0)


# In[29]:


df


# In[ ]:





# In[30]:


def fill_emi(df):
    for i in range(df.shape[0]):
        la=df.iloc[i,16]
        if(la==0):
            df.iloc[i,19]=0
        elif(df.iloc[i,19]==-1):
            total=la+((la*df.iloc[i,19])/100)
            emi=total/(12*df.iloc[i,17])
            df.iloc[i,19]=emi
    df['EMI']=df['EMI'].round(1)
            


# In[31]:


fill_emi(df)


# In[32]:


df


# In[33]:


nullCheck=pd.DataFrame()
nullCheck['Number of null values']=df.isnull().sum()
nullCheck['Percentage of Null Values']=(df.isnull().sum() / df.shape[0]) * 100
nullCheck=nullCheck.sort_values('Percentage of Null Values',ascending=False)
nullCheck


# In[34]:


df['Primary_Bank_Type'].unique()


# In[35]:


df[df.duplicated()==True]


# In[36]:


df['Customer_Existing_Primary_Bank_Code'].unique()


# In[37]:


df=df.dropna()


# In[38]:


df


# In[39]:


df['City_Category'].unique()


# In[40]:


df=df.drop(['Employer_Code','ID','City_Code','Customer_Existing_Primary_Bank_Code','Var1',
            'Source','Employer_Category1'],axis='columns')


# In[41]:


df.columns


# In[42]:


df.info()


# In[43]:


# df[["dobday", "dobmonth", "dobyear"]] = df["DOB"].str.split("/", expand=True)
# df[["leadday", "leadmonth", "leadyear"]] = df["Lead_Creation_Date"].str.split("/", expand=True)


# In[44]:


# df=df.drop(['DOB','Lead_Creation_Date'],axis='columns')


# In[45]:


df=df.reset_index(drop=True)


# In[46]:


df


# In[47]:


df=df.drop(['DOB','Lead_Creation_Date'],axis='columns')


# # Categorical Encoding

# In[48]:


dictionary={ 'Gender':{'Male':1,'Female':0},
      'Contacted':{'Y':1,'N':0},
    'Primary_Bank_Type':{'P':1,'G':0}
}

df=df.replace(dictionary)


# In[49]:


df.City_Category.unique()


# In[50]:


df.Source_Category.unique()


# In[51]:


dummies1=pd.get_dummies(df.City_Category)
# dummies1=dummies1.drop(dummies1.columns[0],axis='columns')
# X=pd.concat([X,dummies1],axis='columns')

dummies1.columns={'City_A','City_B','City_C'}

dummies1=dummies1.drop(dummies1.columns[0],axis='columns')
df=pd.concat([df,dummies1],axis='columns')

dummies2=pd.get_dummies(df.Source_Category)

dummies2=dummies2.drop(dummies2.columns[0],axis='columns')
df=pd.concat([df,dummies2],axis='columns')


# In[52]:


df


# In[53]:


df=df.drop(['City_Category','Source_Category'],axis='columns')


# # Splitting Data

# In[54]:


Y=df['Approved']
X=df.drop(['Approved'],axis='columns')


# In[55]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=99)


# In[56]:


X_train


# # Model Training

# In[57]:


from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(X_train,Y_train)


# In[58]:


xgb.score(X_test,Y_test)


# In[59]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(xgb,X_train,Y_train,cv=5)
scores


# In[60]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)


# In[61]:


svc.score(X_test,Y_test)


# In[62]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(svc,X_train,Y_train,cv=5)
scores


# In[ ]:





# # Metrics

# In[ ]:





# In[ ]:





# In[63]:


Y_predicted=xgb.predict(X_test)

from sklearn import metrics
matrix=metrics.confusion_matrix(Y_predicted,Y_test)
sns.heatmap(matrix,annot=True,fmt=".2f")
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()


# In[64]:


xgb.fit(X,Y)


# In[65]:


test_df=pd.read_csv('test.csv')


# In[66]:


test_df


# In[67]:


nullCheck=pd.DataFrame()
nullCheck['Number of null values']=test_df.isnull().sum()
nullCheck['Percentage of Null Values']=(test_df.isnull().sum() / test_df.shape[0]) * 100
nullCheck=nullCheck.sort_values('Percentage of Null Values',ascending=False)
nullCheck


# In[68]:


test_df['Loan_Amount']=test_df['Loan_Amount'].fillna(0)
test_df['Loan_Period']=test_df['Loan_Period'].fillna(0)


# In[69]:


test_df2=test_df.iloc[:,7:10]
test_df3=test_df.iloc[ :,16:19] #loan period
test_df2=pd.concat([test_df2,test_df3],axis='columns')


# In[70]:


test_df4=test_df2[test_df2['Interest_Rate'].isnull()==True]
test_df5=test_df4[test_df4['Loan_Amount']!=0]


# In[71]:


test_df5


# In[72]:


X_test=test_df5.drop(['Interest_Rate'],axis='columns')

dummies2=pd.get_dummies(X_test['Employer_Category1'])
dummies2=dummies2.drop(dummies2.columns[0],axis='columns')
X_test=X_test.drop(['Employer_Category1'],axis='columns')
X_test=pd.concat([X_test,dummies2],axis='columns')


# In[73]:


X_test


# In[74]:


Y_predicted=xgbr.predict(X_test)


# In[75]:


test_df['Interest_Rate']=test_df['Interest_Rate'].fillna(-1)
test_df['EMI']=test_df['EMI'].fillna(-1)


# In[76]:


fill_ir(test_df,Y_predicted,0)
fill_emi(test_df)


# In[77]:


test_df=test_df.dropna()
final_df=test_df.copy()
test_df=test_df.drop(['Employer_Code','ID','City_Code','Customer_Existing_Primary_Bank_Code','Var1',
            'Source','Employer_Category1'],axis='columns')


# In[80]:


test_df=test_df.drop(['DOB','Lead_Creation_Date'],axis='columns')
test_df=test_df.reset_index(drop=True)


# In[81]:


test_df=test_df.replace(dictionary)
dummies1=pd.get_dummies(test_df.City_Category)
# dummies1=dummies1.drop(dummies1.columns[0],axis='columns')
# X=pd.concat([X,dummies1],axis='columns')

dummies1.columns={'City_A','City_B','City_C'}

dummies1=dummies1.drop(dummies1.columns[0],axis='columns')
test_df=pd.concat([test_df,dummies1],axis='columns')

dummies2=pd.get_dummies(test_df.Source_Category)

dummies2=dummies2.drop(dummies2.columns[0],axis='columns')
test_df=pd.concat([test_df,dummies2],axis='columns')

test_df=test_df.drop(['City_Category','Source_Category'],axis='columns')


# In[82]:


test_df


# In[83]:


probs=xgb.predict_proba(test_df)


# In[84]:


probs_array=[]
for i in probs:
    if i[0]>i[1]:
        probs_array.append((1-i[0])*100)
    else:
        probs_array.append((i[1])*100)


# In[85]:


probs_array = ["%.2f" % x for x in probs_array]

# print(probs_array)


# In[86]:


final_df['Scores']=probs_array


# In[87]:


final_df


# In[88]:


final_df.to_csv('results.csv')


# In[ ]:





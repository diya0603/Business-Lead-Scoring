import streamlit as st
import numpy as np
import plotly_express as px
import pandas as pd

st.set_page_config(
	page_title="Bank Lead Scoring",
	layout='wide',
	page_icon="💵")

st.set_option('deprecation.showfileUploaderEncoding',False)

hide_streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            MainMenu {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Lead Scores")

df=pd.read_csv('results.csv')
df=df.drop(df.columns[0],axis='columns')


st.sidebar.header("Apply Filters Here...")

attributes=df.columns.tolist()
attr=st.sidebar.multiselect('Attributes',options=attributes,default=attributes)

if attr=="":
	st.dataframe(df)
else:
	display_df=df.loc[:,attr]
	st.dataframe(display_df)
	

# hot_lead=[]
# cold_lead=[]

# for i in range(df.shape[0]):
# 	if (df.iloc[i,df.shape[1]-1]>=50):
# 		hot_lead.append(i)
# 	else:
# 		cold_lead.append(i)

# st.write(hot_lead)

# labels=["Hot Lead","Cold Lead"]
# values=[hot_lead,cold_lead]
# fig1=px.pie(df,values=values,names=labels) #hole=0.55
# st.plotly_chart(fig1)




# filter=st.sidebarselectbox("Filter results on", donut_test_df['Win Funnel'].unique().tolist())

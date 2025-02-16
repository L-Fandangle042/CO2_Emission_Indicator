import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import datetime


st.set_page_config(layout="wide")

# @st.experimental_memo
CO2_yearly_path = "data/CO2_simplified_by_name.xlsx"
df = pd.read_excel(CO2_yearly_path)

CH4_df = pd.read_excel("data/CH4_simplified.xlsx")
N2O_df = pd.read_excel("data/N2O_simplified.xlsx")

# @st.experimental_memo
# CO2_region_path = "/Users/anna/code/L-Fandangle042/CO2_Emission_Indicator/data/carbon_dioxide/CO2_region.xlsx"
# df_region = pd.read_excel(CO2_region_path)

st.header("Welcome to the CO2 Emissions predictor")
st.text("Select a country on the sidebar and click 'Predict' 🚀")
st.text('------------------------------------------------------')

# Country filter
st.sidebar.title("Filters")
country_selected = st.sidebar.selectbox("Select country", options=df['country'].unique()) #, default=countries_list)

# year = datetime.datetime.today().year
# YEARS = [year + i for i in range(28)]
# year_selected = st.sidebar.selectbox("Select year", options=YEARS) #, default=countries_list)

df_selection = df.query("country == @country_selected")
ch4_selection = CH4_df.query("Name == @country_selected")
n2o_selection = N2O_df.query("Name == @country_selected")

# Graph
if country_selected:
    st.write('Country Selected:', country_selected)

col1, col2 = st.columns(2)

with col1:
    st.header('CO2')
    fig = px.line(df_selection, x="year", y="CO2", color='country',
                title='CO2 emissions by country and year')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("CH4 and N2O")
    fig2 = go.Figure()

    # add line / trace 2 to figure
    fig2.add_trace(go.Scatter(
        x=ch4_selection['year'],
        y=ch4_selection['gas'],
        marker=dict(color="green"),
        name='CH4'
    ))

    # add line / trace 2 to figure
    fig2.add_trace(go.Scatter(
        x=n2o_selection['year'],
        y=n2o_selection['gas'],
        marker=dict(color="red"),
        name='N2O'
    ))

    st.plotly_chart(fig2, use_container_width=True)


# API implementation

# url = 'http://127.0.0.1:8000/predict'
url  = "https://co2project-vzzs3rfq7q-ew.a.run.app/predict"

if st.sidebar.button("Predict"):
    question = f'Will {country_selected} reach its environmental goals for CO2?'

    if country_selected == "Bhutan":
        st.subheader(question)
        st.subheader('✅ Yes')
    else:
        params = {"country": country_selected}
        response = requests.get(url, params=params)
        st.subheader(question + '\n')
        if response.text == "false":
            st.subheader("❌ No")
        else:
            st.subheader('✅ Yes')
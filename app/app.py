import streamlit as st
from datetime import datetime
from src import hello_world


# ---------------------------- Page configuration ----------------------------
st.set_page_config(
    page_title="Homepage", # name that apear in the tab in web navigator
    page_icon="🏡", # icon that apear in the tab in web navigator
    layout="centered",
    initial_sidebar_state="auto"
)


# ---------------------------- read env variables used in the app ----------------------------
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
PROJECT_GCP = os.environ.get("PROJECT_GCP", "")
REGION_GCP = os.environ.get("REGION_GCP", "")
BUCKET_GCP = os.environ.get("BUCKET_GCP", "")



# ---------------------------- invoke codes from script (testing - delete) ----------------------------
msg_script = hello_world.invoke_function()
st.write('---')
st.write(msg_script)
print('debugging main: ', msg_script)
st.write('---')


# ---------------------------- Main Tittle ----------------------------
# Give your app a title
st.title("PoC / MVP Auto Data Science Application")

# Header
st.header("Abstract")
st.write("This is a cloud application where the user can do auto data science with no code")
st.write("This application is located in cloud (GCP) and it can do exploratory data analysis, training forecast models, evaluation of this models and\
and can do simulation with the trained models")



# ---------------------------- Write text transversal each page ----------------------------
st.header("Content")

with st.expander("Upload data"):
    st.write("In this first step the user can upload a timeseries dataset with the objetive of to do forecasting of a variable of the dataset to X steps to\
    the future. When the user upload the dataset if it possible select what kind of analysis of the data do to this dataset and select what kind of models \
    fit to this data (TODO) and obviosly select the features and target in the data")

with st.expander("Exploratory Data Analysis (eda)"):
    st.write("Show different analyzes of the data, looking for patterns in them")

with st.expander("Send training job"):
    st.write("Given a dataset and the configuration of the models to train, send a job in vertex to train all the models. It is only one job and in this job \
             several models are trained")

with st.expander("Vertex Experiment"):
    st.write("After send a tranning job and it finished. It is possible see results saved in vertex experiments for each model trained. \
             See params, metrics and artifacts saved in vertex experiments for each model trained for each dataset")

with st.expander("Offline Evaluation of the trained models"):
    st.write("See differents offline evaluation analysis in the forecast models trained")

with st.expander("Simulations"):
    st.write("Given a model trained (selected by the user considering the metrics and offline evaluation of the model) do differents simulations to see how \
    the inference change, given changes in the data")
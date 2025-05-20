import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(os.path.abspath(".."))

@st.cache_data
def load_data():
    benin = pd.read_csv("data/benin_clean.csv")
    sierra_leone = pd.read_csv("data/sierraleone_clean.csv")
    togo = pd.read_csv("data/togo_clean.csv")
    
    benin["Country"] = "Benin"
    sierra_leone["Country"] = "Sierra Leone"
    togo["Country"] = "Togo"
    
    return pd.concat([benin, sierra_leone, togo])

def plot_boxplots(df, metric):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Country", y=metric, palette="pastel", ax=ax)
    st.pyplot(fig)

def show_summary(df):
    summary = df.groupby("Country")[["GHI", "DNI", "DHI"]].agg(["mean", "median", "std"])
    st.dataframe(summary.round(2))

def show_bar_chart(df):
    avg = df.groupby("Country")["GHI"].mean().sort_values(ascending=False)
    st.bar_chart(avg)
    
# st.write("Cross-Country Solar Potential Comparison")

st.title("Cross-Country Solar Potential Comparison")


df = load_data()

metric = st.selectbox("Select Solar Metric", ["GHI", "DNI", "DHI"])
plot_boxplots(df, metric)

if st.checkbox("Show Summary Table"):
    show_summary(df)

if st.button("Average GHI Ranking"):
    show_bar_chart(df)
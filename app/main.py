import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

class SolarDashboard:
    def __init__(self):
        self.df = SolarDashboard.load_data()  # cleaner call
        self.filtered_df = self.df

    @staticmethod
    @st.cache_data
    def load_data():
        benin = pd.read_csv("data/benin_clean.csv")
        sierra_leone = pd.read_csv("data/sierraleone_clean.csv")
        togo = pd.read_csv("data/togo_clean.csv")
        
        benin["Country"] = "Benin"
        sierra_leone["Country"] = "Sierra Leone"
        togo["Country"] = "Togo"
        
        return pd.concat([benin, sierra_leone, togo])
    def filter_data(self, selected_countries):
        self.filtered_df = self.df[self.df["Country"].isin(selected_countries)]

    def plot_boxplots(self, metric):
        fig, ax = plt.subplots()
        sns.boxplot(data=self.filtered_df, x="Country", y=metric, palette="pastel", ax=ax)
        st.pyplot(fig)

    def show_summary(self):
        summary = self.filtered_df.groupby("Country")[["GHI", "DNI", "DHI"]].agg(["mean", "median", "std"])
        st.dataframe(summary.round(2))

    def show_bar_chart(self):
        avg = self.filtered_df.groupby("Country")["GHI"].mean().sort_values(ascending=False)
        st.bar_chart(avg)

# Streamlit UI
st.title("Cross-Country Solar Potential Comparison")
dashboard = SolarDashboard()

all_countries = dashboard.df["Country"].unique().tolist()
selected = st.multiselect("Select Countries", options=all_countries, default=all_countries)
dashboard.filter_data(selected)

metric = st.selectbox("Select Solar Metric", ["GHI", "DNI", "DHI"])
dashboard.plot_boxplots(metric)

if st.checkbox("Show Summary Table"):
    dashboard.show_summary()

if st.button("Average GHI Ranking"):
    dashboard.show_bar_chart()

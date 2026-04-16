import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Mental Health Dashboard", layout="wide")

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("mhacs_dashboard.csv")
model = pickle.load(open("model.pkl", "rb"))

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------
st.sidebar.title("🔍 Filters")

stress_filter = st.sidebar.multiselect(
    "Stress Level", df['stress'].unique(), default=df['stress'].unique()
)

age_filter = st.sidebar.multiselect(
    "Age Group", df['age'].unique(), default=df['age'].unique()
)

gender_filter = st.sidebar.multiselect(
    "Gender", df['gender'].unique(), default=df['gender'].unique()
)

# Apply filters
df_filtered = df[
    (df['stress'].isin(stress_filter)) &
    (df['age'].isin(age_filter)) &
    (df['gender'].isin(gender_filter))
]

# ---------------------------
# HEADER
# ---------------------------
st.title("📊 Mental Health Intelligence Dashboard")
st.markdown("### Executive Insights (MHACS Canada Data)")

# ---------------------------
# KPI CARDS
# ---------------------------
col1, col2, col3 = st.columns(3)

total = len(df_filtered)
good = (df_filtered['mental_health_binary'] == 'Good').sum()
poor = (df_filtered['mental_health_binary'] == 'Poor').sum()

col1.metric("Population", total)
col2.metric("Good Mental Health", good)
col3.metric("Poor Mental Health", poor)

# ---------------------------
# 📊 ANIMATED CHART (YOUR REQUEST)
# ---------------------------
st.subheader("📊 Stress vs Mental Health (Animated)")

chart_data = df_filtered.groupby(
    ['stress','mental_health_binary']
).size().reset_index(name='count')

fig = px.bar(
    chart_data,
    x='stress',
    y='count',
    color='mental_health_binary',
    barmode='group',
    animation_frame='stress'
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 📊 AGE ANALYSIS
# ---------------------------
st.subheader("📈 Mental Health by Age")

age_chart = df_filtered.groupby(
    ['age','mental_health_binary']
).size().reset_index(name='count')

fig2 = px.bar(
    age_chart,
    x='age',
    y='count',
    color='mental_health_binary',
    barmode='group'
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------------------
# 🌍 REGION VIEW (IF EXISTS)
# ---------------------------
if 'region' in df.columns:
    st.subheader("🌍 Regional Distribution")

    region_chart = df_filtered.groupby(
        ['region','mental_health_binary']
    ).size().reset_index(name='count')

    fig3 = px.bar(
        region_chart,
        x='region',
        y='count',
        color='mental_health_binary',
        barmode='group'
    )

    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------
# 📥 DOWNLOAD BUTTON
# ---------------------------
st.subheader("📥 Download Filtered Data")

csv = df_filtered.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV Report",
    data=csv,
    file_name="mental_health_report.csv",
    mime="text/csv"
)

# ---------------------------
# 🔮 AI PREDICTION
# ---------------------------
st.subheader("🔮 AI Prediction Tool")

stress = st.selectbox(
    "Stress Level",
    ['Low','Moderate','Average','High','Very High']
)

age = st.selectbox(
    "Age Group",
    ['15–24','25–34','35–44','45–54','55–64','65+']
)

gender = st.selectbox(
    "Gender",
    ['Male','Female']
)

# Mapping back to numeric
stress_map = {'Low':1,'Moderate':2,'Average':3,'High':4,'Very High':5}
age_map = {'15–24':1,'25–34':2,'35–44':3,'45–54':4,'55–64':5,'65+':6}
gender_map = {'Male':1,'Female':2}

if st.button("Predict Outcome"):
    input_data = np.array([[stress_map[stress], age_map[age], gender_map[gender]]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Good Mental Health")
    else:
        st.error("⚠️ Poor Mental Health")

# ---------------------------
# 📌 INSIGHTS
# ---------------------------
st.subheader("📌 Executive Insights")

st.info("""
✔ Stress is the strongest predictor of mental health  
✔ High stress → significantly worse outcomes  
✔ Older individuals show better resilience  
✔ Socioeconomic factors influence mental health  
✔ Machine learning confirms these relationships  
""")
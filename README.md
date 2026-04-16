📊 MHACS Mental Health Analysis Dashboard
🧠 Project Overview

This capstone project analyzes mental health outcomes using the MHACS 2022 dataset. The goal is to identify key factors influencing mental health status and build an interactive dashboard that supports data-driven insights and decision-making.

The project combines statistical analysis, machine learning, and data visualization to explore relationships between stress levels, demographics, and mental health outcomes.

🎯 Objectives
Analyze mental health patterns across population groups
Identify key predictors of poor mental health
Perform statistical testing (Chi-Square) to validate relationships
Build predictive models (Logistic Regression & Random Forest)
Develop an interactive Streamlit dashboard for visualization
📂 Dataset
Source: MHACS 2022 (Mental Health and Access to Care Survey)
Type: Public health dataset (Canada)
Features include:
Stress level
Age group
Gender
Education level
Income level
Geographic region
🔍 Key Analysis Performed
1. Data Cleaning & Preparation
Converted categorical variables into usable formats
Created binary mental health indicator
Handled missing values and inconsistencies
2. Exploratory Data Analysis (EDA)
Distribution of mental health outcomes
Stress vs mental health relationships
Demographic breakdowns
3. Statistical Testing
Chi-Square Test used to assess significance
Result: Strong statistical relationship between stress and mental health
4. Predictive Modeling
📉 Logistic Regression
Identified key predictors:
Stress (strongest factor)
Age
Gender
Model statistically significant (p < 0.05)
🌲 Random Forest
Used for improved prediction accuracy
Captures non-linear relationships
Provides feature importance insights
📊 Dashboard Features (Streamlit)

The interactive dashboard includes:

🎛 Filters (Stress, Age, Gender, Region)
📈 Animated charts (Plotly)
🧑‍🤝‍🧑 Demographic insights
🌍 Regional distribution visualization
📥 Downloadable filtered dataset
🌙 Dark theme (professional UI)
🚀 How to Run the App
🔹 Local Run
pip install -r requirements.txt
streamlit run app.py
🌐 Live Deployment

The app is deployed using Streamlit Cloud.

👉 (Insert your live app link here after deployment)

Example:

https://your-app.streamlit.app
🛠️ Technologies Used
Python
Pandas & NumPy
Matplotlib & Plotly
Scikit-learn
Statsmodels
Streamlit
📈 Key Insights
Higher stress levels strongly correlate with poor mental health
Age and gender significantly influence outcomes
Urban vs rural differences observed in mental health distribution
Predictive models confirm stress as the most important factor
📌 Conclusion

This project demonstrates how data analytics and machine learning can be applied to public health data to uncover meaningful insights. The dashboard provides an accessible way for stakeholders to explore mental health trends and support evidence-based decisions.

👤 Author

Oludeji Fashoro
Master’s in Data Analytics
University of Niagara Falls

📧 Contact

dejifashoro@iauwu.com

⭐ Acknowledgements
MHACS 2022 dataset providers
Open-source Python community

https://mhacs-mental-health-analysis-svfssziftgy98b8b7hhd9p.streamlit.app/

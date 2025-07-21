import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide", page_title="📊 Marketing Mix Dashboard")

st.title("💼 Marketing Mix Modeling (MMM) Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("../data/simulated/marketing_mmm_dataset.csv")
    df['Week'] = pd.to_datetime(df['Week'])
    df.set_index('Week', inplace=True)
    return df

df = load_data()

# Show raw data
with st.expander("🔍 Preview Dataset"):
    st.dataframe(df.head())

# Adstock Function
def apply_adstock(series, decay=0.5):
    result = []
    accumulated = 0
    for x in series:
        accumulated = x + decay * accumulated
        result.append(accumulated)
    return result

# Apply adstock
channels = ['TV_Spend', 'Radio_Spend', 'Digital_Spend', 'Influencer_Spend', 'OOH_Spend']
for ch in channels:
    df[ch + '_adstock'] = apply_adstock(df[ch])

# Modeling
X = df[[c + '_adstock' for c in channels] + ['Seasonality_Index', 'Holiday_Flag', 'Competitor_Activity']]
y = df['Sales']
model = LinearRegression()
model.fit(X, y)

# Show channel coefficients
with st.expander("📌 Channel Effectiveness Coefficients"):
    coeffs = pd.Series(model.coef_, index=X.columns)
    st.write(coeffs)

# ROI Curves
st.subheader("📈 ROI Curves per Channel")

fig, ax = plt.subplots(figsize=(10, 5))
for ch in channels:
    spends = np.linspace(0, df[ch].max() * 1.5, 50)
    sales = [model.coef_[X.columns.get_loc(ch + '_adstock')] * apply_adstock([s]*10)[-1] for s in spends]
    ax.plot(spends, sales, label=ch.replace('_', ' '))
ax.set_title("ROI Curve per Channel")
ax.set_xlabel("Spend ($)")
ax.set_ylabel("Estimated Sales Contribution")
ax.legend()
st.pyplot(fig)

# Scenario Simulator
st.subheader("🧪 Scenario Simulation")

col1, col2 = st.columns(2)
adjustments = {}

with col1:
    for ch in channels[:3]:
        percent = st.slider(f"{ch} Change (%)", -100, 100, 0)
        adjustments[ch] = percent / 100

with col2:
    for ch in channels[3:]:
        percent = st.slider(f"{ch} Change (%)", -100, 100, 0)
        adjustments[ch] = percent / 100

def simulate_sales(adj):
    df_sim = df.copy()
    for ch in channels:
        df_sim[ch] *= (1 + adj[ch])
        df_sim[ch + '_adstock'] = apply_adstock(df_sim[ch])
    X_sim = df_sim[[ch + '_adstock' for ch in channels] + ['Seasonality_Index', 'Holiday_Flag', 'Competitor_Activity']]
    df_sim['Predicted_Sales'] = model.predict(X_sim)
    return df_sim

df_simulated = simulate_sales(adjustments)

# Plot comparison
fig2, ax2 = plt.subplots(figsize=(10, 4))
df[['Sales']].plot(ax=ax2, label="Actual Sales", color='blue')
df_simulated[['Predicted_Sales']].plot(ax=ax2, linestyle="--", color='red')
ax2.set_title("Actual vs. Simulated Sales")
st.pyplot(fig2)

# Optimized Budget (Optional Preview)
st.subheader("💰 Optimized Budget Allocation (Last Week Example)")

from scipy.optimize import minimize

last_week_spend = df[channels].iloc[-1].sum()
channel_coeffs = coeffs[[ch + '_adstock' for ch in channels]].values

def objective(x):
    return -np.dot(x, channel_coeffs)

bounds = [(0, None) for _ in channels]
constraints = {'type': 'eq', 'fun': lambda x: sum(x) - last_week_spend}
x0 = df[channels].iloc[-1].values

opt_result = minimize(objective, x0, bounds=bounds, constraints=constraints)
optimal_spend = dict(zip(channels, opt_result.x))

st.write("Optimal Budget Split:")
st.json({ch: f"${val:,.0f}" for ch, val in optimal_spend.items()})

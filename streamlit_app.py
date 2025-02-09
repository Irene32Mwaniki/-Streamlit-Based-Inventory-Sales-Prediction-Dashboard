import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

#######################################
# PAGE SETUP
#######################################

st.set_page_config(page_title="Inventory Dashboard", page_icon=":bar_chart:", layout="wide")

st.title("📊 Chemical Inventory Dashboard")
#st.markdown("_Prototype v1.0_")

st.markdown("""
### 🔍 About This Dashboard
This is a real-time interactive inventory dashboard for an SME allowing it to:

- Monitor stock levels and track sales trends.
- Analyze wastage and its impact on revenue.
- Identify best and worst-selling products.
- Determine stock turnover rates to optimize inventory management.
- Predict future sales trends and recommend reorder levels.
- Get real-time reorder recommendations to prevent stock shortages.


""")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is None:
    st.info(" Upload a file through config", icon="ℹ️")
    st.stop()


#######################################
# DATA LOADING
#######################################

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

df = load_data(uploaded_file)

# Ensure correct data types
df['Sales Quantity'] = df['Sales Quantity'].astype(float)
df['Starting Stock'] = df['Starting Stock'].astype(float)
df['Ending Stock'] = df['Ending Stock'].astype(float)
df['Normal Waste'] = df['Normal Waste'].astype(float)
df['Abnormal Waste'] = df['Abnormal Waste'].astype(float)

df['Total Waste'] = df['Normal Waste'] + df['Abnormal Waste']
df['Wastage Percentage'] = (df['Total Waste'] / df['Starting Stock']) * 100
df['Stock Turnover Rate'] = df['Sales Quantity'] / df['Starting Stock']

# Reorder recommendation
def reorder_status(row):
    return "Yes" if row['Ending Stock'] <= (0.2 * row['Starting Stock']) else "No"
df['Reorder Needed'] = df.apply(reorder_status, axis=1)

# Sidebar Filters
selected_month = st.sidebar.selectbox("Select Month", df["Month"].unique())
selected_sku = st.sidebar.selectbox("Select Stock Unit Code (SKU ID)", df["SKU ID"].unique())
filtered_df = df[(df["Month"] == selected_month) & (df["SKU ID"] == selected_sku)]

# Ensure correct data types
df['Sales Quantity'] = df['Sales Quantity'].astype(float)
df['Starting Stock'] = df['Starting Stock'].astype(float)
df['Ending Stock'] = df['Ending Stock'].astype(float)
df['Normal Waste'] = df['Normal Waste'].astype(float)
df['Abnormal Waste'] = df['Abnormal Waste'].astype(float)

# Ensure 'Month' column exists and convert to categorical
if 'Month' in df.columns:
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    df = df.dropna(subset=['Month'])  # Drop missing values in Month column
    df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
    
    if df['Month'].isnull().all():
        st.error("❌ 'Month' column values are not matching expected month names. Please check data format.")
        st.stop()
else:
    st.error("❌ 'Month' column not found in the dataset. Please check the CSV file format.")
    st.stop()

#######################################
# DASHBOARD METRICS WITH DONUT CHARTS
#######################################

st.header("🔹 Key Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Starting Stock", int(filtered_df["Starting Stock"].values[0]), delta_color="inverse")
col2.metric("📉 Sales Quantity", int(filtered_df["Sales Quantity"].values[0]), delta_color="off")
col3.metric("🔄 Stock Turnover", round(filtered_df["Stock Turnover Rate"].values[0], 2), delta_color="normal")
col4.metric("📊 Wastage %", f"{round(filtered_df['Wastage Percentage'].values[0], 2)}%", delta_color="inverse")

# Bar Chart for Waste Analysis
st.header("📊 Waste Distribution Analysis")
fig_waste = px.bar(df, x='SKU ID', y=['Normal Waste', 'Abnormal Waste'], barmode='stack', title="Waste Distribution by SKU")
st.plotly_chart(fig_waste, use_container_width=True)

# Donut Charts for Waste Analysis
st.header("🍩 Waste Analysis")
fig_donut = px.pie(df, names='SKU ID', values='Total Waste', hole=0.4, title="Waste Distribution by SKU")
st.plotly_chart(fig_donut, use_container_width=True)

# Dropdown for Specific Chemical Analysis
st.header("🔍 Check Specific Product Waste")
selected_chemical = st.selectbox("Select Chemical", df["SKU ID"].unique())
df_chemical = df[df["SKU ID"] == selected_chemical]
fig_bar = px.bar(df_chemical, x="Month", y="Total Waste", title=f"Waste Trend for {selected_chemical}")
st.plotly_chart(fig_bar, use_container_width=True)

st.header("📢 Reorder Recommendation")
if filtered_df["Reorder Needed"].values[0] == "Yes":
    st.error("⚠️ Reorder Needed!")
else:
    st.success("✅ No Reorder Needed.")


#######################################
# OPTIMAL REORDER QUANTITY RECOMMENDATION
#######################################

st.header("🔄 Optimal Reorder Quantity")

avg_sales = df.groupby('SKU ID')['Sales Quantity'].mean()
reorder_levels = avg_sales * 1.5  # Set reorder level as 1.5 times the average sales
reorder_df = pd.DataFrame({
    'SKU ID': avg_sales.index,
    'Avg Monthly Sales': avg_sales.values,
    'Recommended Reorder Level': reorder_levels.values
})

st.write("### 📦 Recommended Reorder Levels Based on Sales")
st.table(reorder_df)

#######################################
# FINANCIAL ANALYSIS
#######################################

st.header("💰 Financial Analysis")

price_per_unit = {1002: 10.5, 1005: 12.0}  # Example pricing
df['Price Per Unit'] = df['SKU ID'].map(price_per_unit)
df['Revenue'] = df['Sales Quantity'] * df['Price Per Unit']
df['Profit'] = df['Revenue'] - ((df['Normal Waste'] + df['Abnormal Waste']) * df['Price Per Unit'])

revenue_summary = df.groupby('SKU ID')[['Revenue', 'Profit']].sum().reset_index()
fig_finance = px.bar(revenue_summary, x='SKU ID', y=['Revenue', 'Profit'], title="Revenue & Profit by SKU", barmode='group')

st.plotly_chart(fig_finance, use_container_width=True)

st.write("### 🔍 Revenue Impact of Wastage")
wastage_cost = (df['Total Waste'] * df['Price Per Unit']).sum()
st.warning(f"⚠️ Total Revenue Lost Due to Wastage: ${wastage_cost:,.2f}")

#######################################
# VISUALIZATIONS
#######################################

st.header("📈 Monthly Sales Trend")
sales_trend = df.groupby("Month")["Sales Quantity"].sum().reset_index()
chart_option = st.radio("Select Chart Type", ["Line Chart", "Bar Chart"])
if chart_option == "Line Chart":
    fig1 = px.line(sales_trend, x="Month", y="Sales Quantity", markers=True, title="Monthly Sales Trend")
else:
    fig1 = px.bar(sales_trend, x="Month", y="Sales Quantity", title="Monthly Sales Trend")
st.plotly_chart(fig1, use_container_width=True)

st.header("🏆 Best & Worst Selling SKUs")
best_sellers = df.groupby("SKU ID")["Sales Quantity"].sum().nlargest(5).reset_index()
worst_sellers = df.groupby("SKU ID")["Sales Quantity"].sum().nsmallest(5).reset_index()

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔥 Top 5 Best Sellers")
    st.table(best_sellers)
with col2:
    st.subheader("❄️ Top 5 Worst Sellers")
    st.table(worst_sellers)

st.header("💰 Revenue & Profit Analysis")
df['Revenue'] = df['Sales Quantity'] * 10  # Assume price per unit is $10 for now
df['Profit'] = df['Revenue'] - (df['Total Waste'] * 5)  # Assume waste costs $5 per unit
fig2 = px.bar(df, x="SKU ID", y=["Revenue", "Profit"], barmode="group", title="Revenue & Profit by SKU")
st.plotly_chart(fig2, use_container_width=True)

#######################################
# BEST & WORST PERFORMING MONTHS
#######################################

st.header("📆 Best & Worst Performing Months")
monthly_sales = df.groupby("Month")["Sales Quantity"].sum().reset_index()
monthly_sales = monthly_sales.sort_values(by="Sales Quantity", ascending=False)

best_month = monthly_sales.iloc[0]
worst_month = monthly_sales.iloc[-1]

st.success(f"🏆 Best Performing Month: {best_month['Month']} with {best_month['Sales Quantity']} units sold")
st.error(f"📉 Worst Performing Month: {worst_month['Month']} with {worst_month['Sales Quantity']} units sold")

fig_months = px.bar(monthly_sales, x="Month", y="Sales Quantity", title="Monthly Sales Performance", color="Sales Quantity", color_continuous_scale="blues")
st.plotly_chart(fig_months, use_container_width=True)

#######################################
# SALES PREDICTION USING LINEAR REGRESSION
#######################################

st.header("📈 Sales Forecasting")

if 'Month' in df.columns:
    sales_trend = df.groupby('Month')['Sales Quantity'].sum().reset_index()
    
    if not sales_trend.empty:
        X = np.arange(len(sales_trend)).reshape(-1, 1)  # Convert month index to numerical values
        y = sales_trend['Sales Quantity']
        model = LinearRegression()
        model.fit(X, y)

        future_months = np.array(range(len(sales_trend), len(sales_trend) + 3)).reshape(-1, 1)
        sales_predictions = model.predict(future_months)

        future_df = pd.DataFrame({
            'Month': ['Next Month 1', 'Next Month 2', 'Next Month 3'],
            'Sales Quantity Prediction': sales_predictions
        })

        st.write("### 📊 Predicted Sales for Next 3 Months")
        st.table(future_df)

        # Merge historical and predicted data
        sales_trend = pd.concat([sales_trend, future_df.rename(columns={'Sales Quantity Prediction': 'Sales Quantity'})])
        
        # Visualization selection
        chart_option = st.radio("Select Chart Type for Sales Forecast", ["Line Chart", "Bar Chart"], horizontal=True)
        
        if chart_option == "Line Chart":
            fig_forecast = px.line(sales_trend, x='Month', y='Sales Quantity', markers=True, title="Sales Trend and Prediction")
        else:
            fig_forecast = px.bar(sales_trend, x='Month', y='Sales Quantity', title="Sales Trend and Prediction")
        
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("⚠️ Not enough data to generate sales predictions.")
else:
    st.error("❌ 'Month' column not found. Ensure 'Month' values are correct.")
    st.stop()
#######################################
# DATA EXPORT
#######################################

st.download_button(
    label="📥 Download Processed Data",
    data=df.to_csv(index=False),
    file_name="chemical_inventory_analysis.csv",
    mime="text/csv"
)

st.markdown("**📌 Note:** This dashboard helps monitor inventory performance and optimize stock levels.")



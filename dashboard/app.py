import streamlit as st
import pandas as pd
import plotly.express as px
import pygwalker as pyg
import os

st.set_page_config(page_title="Hotel Dynamic Pricing & Retention Dashboard", layout="wide")

# Custom styling for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏨 Travel & Hospitality: Customer Retention & Pricing Dashboard")

st.sidebar.markdown("### 📂 Use Your Own Data")
st.sidebar.info("Upload a CSV with similar columns (e.g., `adr`, `is_canceled`, `lead_time`) to analyze your own data!")
uploaded_file = st.sidebar.file_uploader("Drag and drop your CSV file here", type=['csv'])

# Load Data
@st.cache_data
def load_data(file_upload):
    # If user uploads a file, use that!
    if file_upload is not None:
        try:
            df = pd.read_csv(file_upload)
            if 'arrival_date_year' in df.columns and 'arrival_date_month' in df.columns:
                month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
                             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
                df['arrival_month_num'] = df['arrival_date_month'].map(month_map)
                df['date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_month_num'].astype(str) + '-01')
            return df
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return None

    # Otherwise fallback to the default project data
    file_path = 'data/cleaned_hotel_bookings.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Combine year and month for time series if available
        if 'arrival_date_year' in df.columns and 'arrival_date_month' in df.columns:
            month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
                         'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
            df['arrival_month_num'] = df['arrival_date_month'].map(month_map)
            df['date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_month_num'].astype(str) + '-01')
        return df
    else:
        return None

df = load_data(uploaded_file)

if df is None:
    st.error("Data not found! Please ensure 'setup_data.py' has been run and notebooks executed.")
    st.stop()

# Key Metrics
st.markdown("### 📊 Executive Overview")
col1, col2, col3, col4 = st.columns(4)
total_bookings = len(df)
cancellation_rate = df['is_canceled'].mean() * 100
avg_adr = df['adr'].mean()
avg_lead_time = df['lead_time'].mean()

col1.metric("Total Bookings", f"{total_bookings:,}")
col2.metric("Cancellation Rate", f"{cancellation_rate:.1f}%")
col3.metric("Average Daily Rate (ADR)", f"${avg_adr:.2f}")
col4.metric("Avg Lead Time (Days)", f"{avg_lead_time:.0f}")

st.markdown("---")

# Row 1: Seasonal Pricing and Demand
col_ts1, col_ts2 = st.columns(2)

with col_ts1:
    st.markdown("#### Seasonal Pricing Trend (ADR)")
    if 'date' in df.columns:
        ts_adr = df.groupby('date')['adr'].mean().reset_index()
        fig_adr = px.line(ts_adr, x='date', y='adr', markers=True, title='Average Daily Rate over Time',
                          line_shape='spline', color_discrete_sequence=['#2ecc71'])
        st.plotly_chart(fig_adr, use_container_width=True)
    else:
        st.info("Time series data not available.")

with col_ts2:
    st.markdown("#### Demand / Booking Volume Trend")
    if 'date' in df.columns:
        ts_vol = df.groupby('date')['is_canceled'].count().reset_index()
        ts_vol.rename(columns={'is_canceled': 'Booking Volume'}, inplace=True)
        fig_vol = px.bar(ts_vol, x='date', y='Booking Volume', title='Booking Volume over Time',
                         color_discrete_sequence=['#3498db'])
        st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("---")

# Row 2: Customer Retention and Cancellation Analysis
col_churn1, col_churn2 = st.columns(2)

with col_churn1:
    st.markdown("#### Cancellation by Market Segment")
    segment_cancel = df.groupby('market_segment')['is_canceled'].mean().reset_index()
    segment_cancel['Cancellation Rate (%)'] = segment_cancel['is_canceled'] * 100
    fig_seg = px.bar(segment_cancel, x='market_segment', y='Cancellation Rate (%)',
                     color='market_segment', title='Which segments cancel the most?')
    st.plotly_chart(fig_seg, use_container_width=True)

with col_churn2:
    st.markdown("#### Lead Time vs. Cancellation Risk")
    # Group lead time into bins
    bins = [0, 7, 30, 90, 180, 365, 1000]
    labels = ['0-7 Days', '8-30 Days', '31-90 Days', '91-180 Days', '181-365 Days', '365+ Days']
    df['lead_time_bin'] = pd.cut(df['lead_time'], bins=bins, labels=labels)
    lead_cancel = df.groupby('lead_time_bin')['is_canceled'].mean().reset_index()
    lead_cancel['Cancellation Rate (%)'] = lead_cancel['is_canceled'] * 100
    fig_lead = px.line(lead_cancel, x='lead_time_bin', y='Cancellation Rate (%)', markers=True,
                       title='Does booking earlier increase cancellation risk?',
                       line_shape='spline', color_discrete_sequence=['#e74c3c'])
    st.plotly_chart(fig_lead, use_container_width=True)

st.markdown("---")
st.markdown("### 🔍 My Custom Analysis")
col_custom1, col_custom2 = st.columns(2)

with col_custom1:
    st.markdown("#### Cancellations by Deposit Type")
    # Group the data by deposit_type and find the cancellation average
    deposit_cancel = df.groupby('deposit_type')['is_canceled'].mean().reset_index()
    deposit_cancel['Cancellation Rate (%)'] = deposit_cancel['is_canceled'] * 100
    
    # Create a bar chart using Plotly
    fig_dep = px.bar(deposit_cancel, x='deposit_type', y='Cancellation Rate (%)',
                     color='deposit_type', title='Impact of Deposit Type on Churn')
    st.plotly_chart(fig_dep, use_container_width=True)

with col_custom2:
    st.markdown("#### Room Types Requested")
    # Count the number of each reserved room type
    room_demand = df['reserved_room_type'].value_counts().reset_index()
    room_demand.columns = ['Room Type', 'Count']
    
    # Create a pie chart
    fig_room = px.pie(room_demand, values='Count', names='Room Type', 
                      title='Most Popular Room Types')
    st.plotly_chart(fig_room, use_container_width=True)

st.markdown("---")
st.markdown("### 🛠️ Drag & Drop Chart Builder")
st.info("Use the interface below to drag and drop your data columns into the X and Y axes (just like Tableau) to create your own custom charts instantly!")
if df is not None:
    from pygwalker.api.streamlit import StreamlitRenderer
    pyg_app = StreamlitRenderer(df)
    pyg_app.explorer()

st.markdown("---")
st.markdown("### 🎯 Strategic Recommendations")
st.info("""
- **Dynamic Pricing**: ADR drops significantly during off-peak winter months. Implement targeted discounts to stimulate demand rather than flat rate cuts.
- **Retention Campaigns**: Bookings with a lead time of >90 days exhibit a much higher cancellation risk. Deploy automated email drip campaigns (e.g., local guides, upgrades) 30 days prior to arrival for these early planners to lock them in.
- **Deposit Strategies**: The Transient market segment has high churn. Require non-refundable deposits or partial upfront payments during peak summer months.
""")

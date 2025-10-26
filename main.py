import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os
import requests

# Page configuration
st.set_page_config(page_title="Smart Energy Manager", layout="wide", page_icon="⚡")

# Initialize OpenAI API Key
def get_openai_api_key():
    api_key = None
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    elif "OPENAI_API_KEY" in os.environ:
        api_key = os.environ["OPENAI_API_KEY"]
    return api_key

# Load and preprocess data
@st.cache_data
def load_data(file):
    """Load and preprocess energy consumption data"""
    df = pd.read_csv(file, sep=';', low_memory=False)
    
    # Combine Date and Time columns
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    
    # Convert numeric columns
    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Extract time features
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Month'] = df['DateTime'].dt.month
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    return df

def engineer_features(df):
    """Create additional features for modeling"""
    df = df.copy()
    
    # Calculate total sub-metering
    df['Total_SubMetering'] = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
    
    # Calculate unmeasured power (other appliances)
    df['Other_Power'] = df['Global_active_power'] - (df['Total_SubMetering'] / 1000)
    df['Other_Power'] = df['Other_Power'].clip(lower=0)
    
    # Calculate percentage contributions
    total_power = df['Global_active_power'] * 1000
    df['Kitchen_Pct'] = (df['Sub_metering_1'] / total_power * 100).fillna(0)
    df['Laundry_Pct'] = (df['Sub_metering_2'] / total_power * 100).fillna(0)
    df['Climate_Pct'] = (df['Sub_metering_3'] / total_power * 100).fillna(0)
    df['Other_Pct'] = (df['Other_Power'] * 1000 / total_power * 100).fillna(0)
    
    return df

def train_forecasting_model(df):
    """Train Random Forest model for energy forecasting"""
    df_model = df.copy()
    
    # Features for prediction
    feature_cols = ['Hour', 'DayOfWeek', 'Month', 'IsWeekend', 
                    'Global_reactive_power', 'Voltage', 'Global_intensity']
    
    X = df_model[feature_cols]
    y = df_model['Global_active_power']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Calculate metrics
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    return model, scaler, train_score, test_score, feature_cols

def predict_next_month(df, model, scaler, feature_cols):
    """Predict energy consumption for next month"""
    last_date = df['DateTime'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30*24, freq='h')
    
    future_df = pd.DataFrame({
        'DateTime': future_dates,
        'Hour': future_dates.hour,
        'DayOfWeek': future_dates.dayofweek,
        'Month': future_dates.month,
        'IsWeekend': future_dates.dayofweek.isin([5, 6]).astype(int)
    })
    
    # Use average values for other features
    future_df['Global_reactive_power'] = df['Global_reactive_power'].mean()
    future_df['Voltage'] = df['Voltage'].mean()
    future_df['Global_intensity'] = df['Global_intensity'].mean()
    
    X_future = future_df[feature_cols]
    X_future_scaled = scaler.transform(X_future)
    
    predictions = model.predict(X_future_scaled)
    future_df['Predicted_Power'] = predictions
    
    return future_df

def analyze_appliance_usage(df):
    """Analyze energy usage by appliance category"""
    analysis = {
        'Kitchen (Sub_metering_1)': df['Sub_metering_1'].sum(),
        'Laundry (Sub_metering_2)': df['Sub_metering_2'].sum(),
        'Climate Control (Sub_metering_3)': df['Sub_metering_3'].sum(),
        'Other Appliances': df['Other_Power'].sum() * 1000
    }
    
    total = sum(analysis.values())
    percentages = {k: (v/total)*100 for k, v in analysis.items()}
    
    return analysis, percentages

def generate_energy_context(df, analysis, percentages):
    """Generate context for LLM about user's energy consumption"""
    df_daily = df.groupby(df['DateTime'].dt.date).agg({
        'Global_active_power': 'sum',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum'
    }).reset_index()
    
    avg_daily = df_daily['Global_active_power'].mean()
    max_daily = df_daily['Global_active_power'].max()
    
    peak_hours = df.groupby('Hour')['Global_active_power'].mean().nlargest(3)
    
    context = f"""
Energy Consumption Analysis:

Total Consumption Breakdown:
- Kitchen appliances: {percentages['Kitchen (Sub_metering_1)']:.1f}% ({analysis['Kitchen (Sub_metering_1)']/1000:.2f} kWh)
- Laundry appliances: {percentages['Laundry (Sub_metering_2)']:.1f}% ({analysis['Laundry (Sub_metering_2)']/1000:.2f} kWh)
- Climate control (AC/Heating): {percentages['Climate Control (Sub_metering_3)']:.1f}% ({analysis['Climate Control (Sub_metering_3)']/1000:.2f} kWh)
- Other appliances: {percentages['Other Appliances']:.1f}% ({analysis['Other Appliances']/1000:.2f} kWh)

Usage Patterns:
- Average daily consumption: {avg_daily:.2f} kW
- Maximum daily consumption: {max_daily:.2f} kW
- Peak usage hours: {', '.join([f'{int(h)}:00' for h in peak_hours.index])}

Data period: {df['DateTime'].min().strftime('%Y-%m-%d')} to {df['DateTime'].max().strftime('%Y-%m-%d')}
"""
    return context

def chat_with_energy_assistant(user_message, energy_context, chat_history):
    """Chat with OpenAI GPT-4 energy assistant using requests"""
    api_key = get_openai_api_key()
    
    if not api_key:
        return "⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY in environment variables or Streamlit secrets."
    
    system_prompt = f"""You are an expert energy efficiency advisor helping users reduce their electricity consumption and save money.

Based on the user's energy consumption data:
{energy_context}

Your role:
1. Provide personalized, actionable recommendations to reduce energy consumption
2. Identify high-usage patterns and suggest optimizations
3. Recommend energy-efficient appliances and practices
4. Provide feasibility analysis for renewable energy solutions (solar panels, etc.)
5. Estimate potential savings and ROI
6. Encourage sustainable behavior with positive reinforcement
7. Answer questions about energy management clearly and concisely

Be friendly, practical, and focus on measurable improvements. Use specific numbers from the data when making recommendations."""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history
    for msg in chat_history:
        messages.append(msg)
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4-turbo-preview",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: API returned status code {response.status_code}. {response.text}"
    
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"

# Streamlit UI
def main():
    st.title("⚡ Smart Energy Management System")
    st.markdown("### AI-Powered Energy Analytics & Sustainability Assistant")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Upload")
        uploaded_file = st.file_uploader("Upload your energy data (CSV)", type=['csv'])
        
        st.markdown("---")
        st.markdown("### 📋 System Features")
        st.markdown("""
        - 🔍 Traditional AI Analysis
        - 📈 Energy Forecasting
        - 🤖 AI Chatbot Assistant
        - 💡 Personalized Recommendations
        - 🌱 Renewable Energy Guidance
        """)
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading and processing data..."):
            df = load_data(uploaded_file)
            df = engineer_features(df)
            st.success(f"✅ Loaded {len(df):,} records")
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analytics", "🔮 Forecasting", "🤖 Your AI Assistant", "🌱 Sustainability"])
        
        # Tab 1: Analytics
        with tab1:
            st.header("Energy Consumption Analytics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_power = df['Global_active_power'].mean()
                st.metric("Avg Power (kW)", f"{avg_power:.2f}")
            
            with col2:
                total_kwh = df['Global_active_power'].sum() / 60  # Convert to kWh
                st.metric("Total Energy (kWh)", f"{total_kwh:.0f}")
            
            with col3:
                peak_power = df['Global_active_power'].max()
                st.metric("Peak Power (kW)", f"{peak_power:.2f}")
            
            with col4:
                avg_daily = df.groupby(df['DateTime'].dt.date)['Global_active_power'].sum().mean()
                st.metric("Avg Daily (kW)", f"{avg_daily:.2f}")
            
            # Appliance breakdown
            st.subheader("Appliance Energy Breakdown")
            analysis, percentages = analyze_appliance_usage(df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(percentages.keys()),
                    values=list(percentages.values()),
                    hole=0.4
                )])
                fig_pie.update_layout(title="Energy Consumption by Category")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar chart
                fig_bar = go.Figure(data=[go.Bar(
                    x=list(percentages.keys()),
                    y=list(percentages.values()),
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
                )])
                fig_bar.update_layout(
                    title="Percentage Distribution",
                    yaxis_title="Percentage (%)",
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Time series
            st.subheader("Energy Consumption Over Time")
            df_daily = df.groupby(df['DateTime'].dt.date)['Global_active_power'].mean().reset_index()
            df_daily.columns = ['Date', 'Power']
            
            fig_time = px.line(df_daily, x='Date', y='Power', 
                              title='Daily Average Power Consumption')
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Hourly pattern
            st.subheader("Hourly Usage Pattern")
            hourly_avg = df.groupby('Hour')['Global_active_power'].mean()
            
            fig_hourly = go.Figure(data=[go.Bar(x=hourly_avg.index, y=hourly_avg.values)])
            fig_hourly.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Average Power (kW)"
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Tab 2: Forecasting
        with tab2:
            st.header("Energy Consumption Forecasting")
            
            if st.button("🔮 Train Forecasting Model", type="primary"):
                with st.spinner("Training model... Please wait...."):
                    model, scaler, train_score, test_score, feature_cols = train_forecasting_model(df)
                    
                    st.session_state['model'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['feature_cols'] = feature_cols
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Training Accuracy", f"{train_score*100:.2f}%")
                    with col2:
                        st.metric("Testing Accuracy", f"{test_score*100:.2f}%")
                    
                    st.success("✅ Model trained successfully!")
            
            if 'model' in st.session_state:
                st.subheader("Next Month Prediction")
                
                future_df = predict_next_month(df, st.session_state['model'], 
                                               st.session_state['scaler'], 
                                               st.session_state['feature_cols'])
                
                # Calculate prediction summary
                current_month_avg = df.groupby(df['DateTime'].dt.date)['Global_active_power'].sum().mean()
                predicted_month_avg = future_df.groupby(future_df['DateTime'].dt.date)['Predicted_Power'].sum().mean()
                change_pct = ((predicted_month_avg - current_month_avg) / current_month_avg) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Avg Daily", f"{current_month_avg:.2f} kW")
                with col2:
                    st.metric("Predicted Avg Daily", f"{predicted_month_avg:.2f} kW")
                with col3:
                    st.metric("Expected Change", f"{change_pct:+.1f}%", 
                             delta=f"{change_pct:+.1f}%")
                
                # Plot prediction
                future_daily = future_df.groupby(future_df['DateTime'].dt.date)['Predicted_Power'].mean().reset_index()
                future_daily.columns = ['Date', 'Power']
                
                fig_pred = px.line(future_daily, x='Date', y='Power',
                                  title='Predicted Daily Energy Consumption (Next 30 Days)')
                st.plotly_chart(fig_pred, use_container_width=True)
        
        # Tab 3: AI Assistant
        with tab3:
            st.header("🤖 Energy Efficiency AI Assistant")
            
            api_key = get_openai_api_key()
            
            if not api_key:
                st.warning("⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY in environment variables or Streamlit secrets.")
                st.info("""
                **To set up the API key:**
                
                1. Get your API key from: https://platform.openai.com/api-keys
                2. Set it as environment variable:
                   ```
                   set OPENAI_API_KEY=your-key-here
                   ```
                3. Or create `.streamlit/secrets.toml` with:
                   ```
                   OPENAI_API_KEY = "your-key-here"
                   ```
                """)
            else:
                # Initialize chat history
                if 'chat_history' not in st.session_state:
                    st.session_state['chat_history'] = []
                
                # Generate energy context
                analysis, percentages = analyze_appliance_usage(df)
                energy_context = generate_energy_context(df, analysis, percentages)
                
                # Display chat history
                st.markdown("### 💬 Chat History")
                for i, message in enumerate(st.session_state['chat_history']):
                    if message["role"] == "user":
                        st.markdown(f"**🙋 You:** {message['content']}")
                    else:
                        st.markdown(f"**🤖 Assistant:** {message['content']}")
                    st.markdown("---")
                
                # Chat input using form (compatible with older Streamlit)
                with st.form(key='chat_form', clear_on_submit=True):
                    user_input = st.text_input("Ask me anything about your energy consumption:", 
                                               key='user_input', 
                                               placeholder="Type your question here...")
                    submit_button = st.form_submit_button("Send 📤")
                
                if submit_button and user_input:
                    # Get AI response
                    with st.spinner("🤔 Thinking..."):
                        response = chat_with_energy_assistant(
                            user_input, energy_context, 
                            st.session_state['chat_history']
                        )
                    
                    # Update chat history
                    st.session_state['chat_history'].append({"role": "user", "content": user_input})
                    st.session_state['chat_history'].append({"role": "assistant", "content": response})
                    
                    # Rerun to display new messages
                    st.experimental_rerun()
                
                # Suggested questions
                st.markdown("---")
                st.markdown("### 💡 Suggested Questions")
                col1, col2 = st.columns(2)
                
                suggestions = [
                    "What are my top 3 energy-saving opportunities?",
                    "How can I reduce my climate control costs?",
                    "Is solar panel installation worth it for my consumption?",
                    "What appliances should I replace first?",
                    "How much can I save by optimizing my usage patterns?"
                ]
                
                for idx, suggestion in enumerate(suggestions):
                    col = col1 if idx % 2 == 0 else col2
                    with col:
                        if st.button(suggestion, key=f"suggest_{idx}"):
                            # Add to chat history
                            st.session_state['chat_history'].append({"role": "user", "content": suggestion})
                            
                            # Get response
                            response = chat_with_energy_assistant(
                                suggestion, energy_context, 
                                st.session_state['chat_history'][:-1]
                            )
                            st.session_state['chat_history'].append({"role": "assistant", "content": response})
                            st.experimental_rerun()
        
        # Tab 4: Sustainability
        with tab4:
            st.header("🌱 Sustainability & Renewable Energy")
            
            # Solar panel feasibility
            st.subheader("☀️ Solar Panel Feasibility Analysis")
            
            avg_daily_kwh = df.groupby(df['DateTime'].dt.date)['Global_active_power'].sum().mean() / 60
            monthly_kwh = avg_daily_kwh * 30
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Current Consumption")
                st.metric("Monthly Usage", f"{monthly_kwh:.0f} kWh")
                st.metric("Annual Usage", f"{monthly_kwh * 12:.0f} kWh")
            
            with col2:
                st.markdown("#### Solar Potential")
                # Assume 5 kW system, 4 peak sun hours, 80% efficiency
                solar_capacity = 5
                daily_solar = solar_capacity * 4 * 0.8
                monthly_solar = daily_solar * 30
                
                st.metric("Recommended System", f"{solar_capacity} kW")
                st.metric("Est. Monthly Generation", f"{monthly_solar:.0f} kWh")
                st.metric("Coverage", f"{(monthly_solar/monthly_kwh)*100:.0f}%")
            
            # ROI Calculation (Malaysia)
            st.subheader("💰 Return on Investment (Malaysia)")

            system_cost = solar_capacity * 1000 * 3.5  # RM3.5 per watt (typical installed cost in Malaysia)
            electricity_rate = 0.218  # RM0.218 per kWh (TNB domestic tariff)
            annual_savings = monthly_solar * 12 * electricity_rate
            payback_period = system_cost / annual_savings

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("System Cost", f"RM{system_cost:,.0f}")
            with col2:
                st.metric("Annual Savings", f"RM{annual_savings:,.0f}")
            with col3:
                st.metric("Payback Period", f"{payback_period:.1f} years")

                # Energy efficiency tips
            st.subheader("⚡ Quick Energy Saving Tips")

            
            tips = [
                ("🌡️", "Climate Control", "Set AC to 24°C instead of 22°C to save 10-15%"),
                ("💡", "Lighting", "Switch to LED bulbs - use 75% less energy"),
                ("🧺", "Laundry", "Use cold water for washing - saves 90% energy per load"),
                ("🔌", "Standby Power", "Unplug devices not in use - phantom load costs 5-10%"),
                ("⏰", "Peak Hours", "Run heavy appliances during off-peak hours")
            ]
            
            for icon, title, tip in tips:
                st.markdown(f"**{icon} {title}:** {tip}")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to Smart Energy Management System! 👋
        
        ### How it works:
        
        1. **Upload Your Data** 📤
           - Upload your energy consumption CSV file using the sidebar
           - The system will automatically analyze your usage patterns
        
        2. **Explore Analytics** 📊
           - View detailed breakdowns of your energy consumption
           - Identify which appliances use the most energy
           - Understand your usage patterns by time of day
        
        3. **Get Predictions** 🔮
           - Train AI models to forecast future consumption
           - Predict next month's energy usage
           - Plan ahead and budget accordingly
        
        4. **Chat with AI Assistant** 🤖
           - Get personalized energy-saving recommendations
           - Ask questions about your consumption
           - Receive actionable insights to reduce bills
        
        5. **Explore Renewable Options** 🌱
           - Analyze solar panel feasibility
           - Calculate ROI for renewable investments
           - Get sustainability recommendations
        
        ### Get Started
        Upload your energy consumption data using the sidebar to begin! 
        
        *Expected format: Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3*
        """)

if __name__ == "__main__":
    main()
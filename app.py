import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import json
from risk_calculator import RiskCalculator
from care_plan_generator import CarePlanGenerator
from population_analytics import PopulationAnalytics
from watson_integration import WatsonHealthcareAI
from utils import generate_sample_csv, calculate_bmi, validate_health_metrics

# Page configuration
st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'risk_results' not in st.session_state:
    st.session_state.risk_results = {}
if 'care_plan' not in st.session_state:
    st.session_state.care_plan = {}
if 'watson_insights' not in st.session_state:
    st.session_state.watson_insights = {}

# Initialize Watson AI
@st.cache_resource
def init_watson():
    return WatsonHealthcareAI()

def main():
    st.title("🏥 Healthcare AI Assistant")
    st.markdown("**Comprehensive AI-powered risk assessment and care planning platform**")
    
    # Educational disclaimer
    st.warning("""
    ⚠️ **Important Medical Disclaimer**: This tool is for educational and informational purposes only. 
    It does not provide medical advice, diagnosis, or treatment. Always consult with qualified healthcare 
    professionals for medical decisions and treatment plans.
    """)
    
    # Sidebar for patient intake
    with st.sidebar:
        st.header("📋 Patient Intake Form")
        
        # Basic Information
        st.subheader("Basic Information")
        age = st.number_input("Age", min_value=18, max_value=120, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        # Physical Measurements
        st.subheader("Physical Measurements")
        height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
        
        # Calculate BMI
        bmi = calculate_bmi(height_cm, weight_kg)
        st.metric("BMI", f"{bmi:.1f}")
        
        # Vital Signs
        st.subheader("Vital Signs")
        systolic_bp = st.number_input("Systolic Blood Pressure", min_value=70, max_value=250, value=120)
        diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, value=80)
        resting_hr = st.number_input("Resting Heart Rate", min_value=40, max_value=200, value=70)
        
        # Laboratory Values
        st.subheader("Laboratory Values")
        glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50, max_value=400, value=90)
        cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
        hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=150, value=50)
        ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=50, max_value=300, value=100)
        
        # Lifestyle Factors
        st.subheader("Lifestyle Factors")
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        exercise_days = st.slider("Exercise Days per Week", 0, 7, 3)
        alcohol_drinks = st.slider("Alcoholic Drinks per Week", 0, 20, 2)
        
        # Family History
        st.subheader("Family History")
        family_diabetes = st.checkbox("Family History of Diabetes")
        family_heart_disease = st.checkbox("Family History of Heart Disease")
        family_hypertension = st.checkbox("Family History of Hypertension")
        
        # Medical History
        st.subheader("Medical History")
        current_medications = st.text_area("Current Medications", placeholder="List current medications...")
        allergies = st.text_area("Allergies", placeholder="List known allergies...")
        
        # Calculate button
        if st.button("🔍 Calculate Risk Assessment", type="primary"):
            # Compile patient data
            patient_data = {
                'age': age,
                'gender': gender,
                'height_cm': height_cm,
                'weight_kg': weight_kg,
                'bmi': bmi,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'resting_hr': resting_hr,
                'glucose': glucose,
                'cholesterol': cholesterol,
                'hdl': hdl,
                'ldl': ldl,
                'smoking': smoking,
                'exercise_days': exercise_days,
                'alcohol_drinks': alcohol_drinks,
                'family_diabetes': family_diabetes,
                'family_heart_disease': family_heart_disease,
                'family_hypertension': family_hypertension,
                'current_medications': current_medications,
                'allergies': allergies
            }
            
            # Validate data
            validation_result = validate_health_metrics(patient_data)
            if not validation_result['valid']:
                st.error(f"Validation Error: {validation_result['message']}")
            else:
                st.session_state.patient_data = patient_data
                
                # Calculate risks
                calculator = RiskCalculator()
                st.session_state.risk_results = calculator.calculate_all_risks(patient_data)
                
                # Generate care plan
                care_generator = CarePlanGenerator()
                st.session_state.care_plan = care_generator.generate_care_plan(
                    patient_data, st.session_state.risk_results
                )
                
                # Generate Watson AI insights
                watson_ai = init_watson()
                st.session_state.watson_insights = watson_ai.generate_health_insights(
                    patient_data, st.session_state.risk_results
                )
                
                st.success("✅ Risk assessment completed with AI insights!")
                st.rerun()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Risk Assessment", 
        "📋 Care Plan", 
        "🤖 AI Insights",
        "📊 Population Analytics", 
        "🔧 Tools & Resources"
    ])
    
    with tab1:
        display_risk_assessment()
    
    with tab2:
        display_care_plan()
    
    with tab3:
        display_watson_insights()
    
    with tab4:
        display_population_analytics()
    
    with tab5:
        display_tools_resources()

def display_risk_assessment():
    st.header("🎯 Individual Risk Assessment")
    
    if not st.session_state.risk_results:
        st.info("👈 Please complete the patient intake form in the sidebar to view risk assessment results.")
        return
    
    risk_results = st.session_state.risk_results
    patient_data = st.session_state.patient_data
    
    # Risk Summary Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        diabetes_risk = risk_results['diabetes']['risk_percentage']
        color = "🔴" if diabetes_risk >= 70 else "🟡" if diabetes_risk >= 40 else "🟢"
        st.metric(
            f"{color} Diabetes Risk",
            f"{diabetes_risk:.1f}%",
            delta=f"Risk Level: {risk_results['diabetes']['risk_level']}"
        )
    
    with col2:
        heart_risk = risk_results['heart_disease']['risk_percentage']
        color = "🔴" if heart_risk >= 70 else "🟡" if heart_risk >= 40 else "🟢"
        st.metric(
            f"{color} Heart Disease Risk",
            f"{heart_risk:.1f}%",
            delta=f"Risk Level: {risk_results['heart_disease']['risk_level']}"
        )
    
    with col3:
        hypertension_risk = risk_results['hypertension']['risk_percentage']
        color = "🔴" if hypertension_risk >= 70 else "🟡" if hypertension_risk >= 40 else "🟢"
        st.metric(
            f"{color} Hypertension Risk",
            f"{hypertension_risk:.1f}%",
            delta=f"Risk Level: {risk_results['hypertension']['risk_level']}"
        )
    
    # Risk Visualization
    st.subheader("📈 Risk Breakdown")
    
    # Create risk comparison chart
    risk_data = pd.DataFrame({
        'Condition': ['Diabetes', 'Heart Disease', 'Hypertension'],
        'Risk Percentage': [
            risk_results['diabetes']['risk_percentage'],
            risk_results['heart_disease']['risk_percentage'],
            risk_results['hypertension']['risk_percentage']
        ]
    })
    
    fig = px.bar(
        risk_data,
        x='Condition',
        y='Risk Percentage',
        color='Risk Percentage',
        color_continuous_scale='RdYlGn_r',
        title="Risk Assessment Summary"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Risk Factors
    st.subheader("🔍 Risk Factor Analysis")
    
    for condition in ['diabetes', 'heart_disease', 'hypertension']:
        with st.expander(f"{condition.replace('_', ' ').title()} Risk Factors"):
            result = risk_results[condition]
            
            st.write("**Primary Risk Factors:**")
            for factor, score in result['risk_factors'].items():
                if score > 0:
                    st.write(f"• {factor.replace('_', ' ').title()}: {score:.2f} points")
            
            st.write(f"**Total Risk Score:** {result['total_score']:.2f}")
            st.write(f"**Risk Percentage:** {result['risk_percentage']:.1f}%")
            st.write(f"**Risk Level:** {result['risk_level']}")
    
    # Download report
    if st.button("📄 Download Risk Assessment Report"):
        report_data = {
            'patient_data': st.session_state.patient_data,
            'risk_results': st.session_state.risk_results,
            'generated_at': datetime.now().isoformat()
        }
        
        json_str = json.dumps(report_data, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="risk_assessment_report.json">Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)

def display_care_plan():
    st.header("📋 Personalized Care Plan")
    
    if not st.session_state.care_plan:
        st.info("👈 Please complete the risk assessment first to generate a personalized care plan.")
        return
    
    care_plan = st.session_state.care_plan
    
    # Care Plan Summary
    st.subheader("🎯 Care Plan Overview")
    st.write(f"**Priority Level:** {care_plan['priority_level']}")
    st.write(f"**Primary Focus:** {care_plan['primary_focus']}")
    
    # Recommendations by Category
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💊 Medical Recommendations")
        for rec in care_plan['medical_recommendations']:
            st.write(f"• {rec}")
        
        st.subheader("🏃 Lifestyle Modifications")
        for rec in care_plan['lifestyle_recommendations']:
            st.write(f"• {rec}")
    
    with col2:
        st.subheader("🥗 Dietary Guidelines")
        for rec in care_plan['dietary_recommendations']:
            st.write(f"• {rec}")
        
        st.subheader("📅 Follow-up Schedule")
        for rec in care_plan['follow_up_schedule']:
            st.write(f"• {rec}")
    
    # Monitoring Parameters
    st.subheader("📊 Key Monitoring Parameters")
    monitoring_df = pd.DataFrame(care_plan['monitoring_parameters'])
    st.dataframe(monitoring_df, use_container_width=True)
    
    # Educational Resources
    st.subheader("📚 Educational Resources")
    for resource in care_plan['educational_resources']:
        st.write(f"• {resource}")
    
    # Download care plan
    if st.button("📄 Download Care Plan"):
        care_plan_json = json.dumps(care_plan, indent=2)
        b64 = base64.b64encode(care_plan_json.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="care_plan.json">Download Care Plan</a>'
        st.markdown(href, unsafe_allow_html=True)

def display_watson_insights():
    st.header("🤖 Watson AI Health Insights")
    
    if not st.session_state.watson_insights:
        st.info("👈 Please complete the risk assessment first to generate AI-powered insights.")
        return
    
    insights = st.session_state.watson_insights
    
    # AI Summary
    st.subheader("🧠 AI Analysis Summary")
    st.info(insights.get('ai_summary', 'AI analysis in progress...'))
    
    # Priority Recommendations
    if insights.get('priority_recommendations'):
        st.subheader("⚡ Priority Recommendations")
        for i, rec in enumerate(insights['priority_recommendations'], 1):
            st.write(f"**{i}.** {rec}")
    
    # Personalized Advice
    if insights.get('personalized_advice'):
        st.subheader("👤 Personalized Health Advice")
        for advice in insights['personalized_advice']:
            st.write(f"• {advice}")
    
    # Key Health Entities
    if insights.get('key_health_entities'):
        st.subheader("🔍 Key Health Factors Identified")
        entities_df = pd.DataFrame(insights['key_health_entities'])
        if not entities_df.empty:
            st.dataframe(entities_df, use_container_width=True)
    
    # Risk Factors Analysis
    if insights.get('risk_factors_identified'):
        st.subheader("⚠️ Risk Factors Analysis")
        risk_factors_df = pd.DataFrame(insights['risk_factors_identified'])
        if not risk_factors_df.empty:
            # Create visualization
            fig = px.bar(
                risk_factors_df.head(10),
                x='relevance',
                y='factor',
                color='sentiment',
                orientation='h',
                title="Top Risk Factors by Relevance"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment Analysis
    if insights.get('sentiment_analysis'):
        st.subheader("📊 Health Profile Sentiment Analysis")
        sentiment = insights['sentiment_analysis']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Sentiment", sentiment.get('overall_sentiment', 'neutral').title())
        with col2:
            confidence = sentiment.get('confidence', 0)
            st.metric("Confidence Score", f"{abs(confidence):.2f}")
        
        # Sentiment explanation
        if sentiment.get('overall_sentiment') == 'negative':
            st.warning("⚠️ Analysis indicates concerning health patterns requiring attention.")
        elif sentiment.get('overall_sentiment') == 'positive':
            st.success("✅ Analysis shows generally positive health indicators.")
        else:
            st.info("ℹ️ Analysis shows neutral health profile with mixed indicators.")
    
    # Watson Status
    st.subheader("🔧 AI Service Status")
    watson_ai = init_watson()
    if watson_ai.watson_available:
        st.success("✅ IBM Watson AI is connected and analyzing your health data")
        st.write("**Active Watson Services:**")
        st.write("• Natural Language Understanding for health text analysis")
        st.write("• Advanced pattern recognition for risk assessment")
        st.write("• Personalized recommendation engine")
    else:
        st.warning("⚠️ Watson AI services are not fully configured")
        st.write("Enhanced AI features require proper Watson credentials setup.")

def display_population_analytics():
    st.header("📊 Population Health Analytics")
    
    # File upload section
    st.subheader("📁 Upload Population Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file with patient data",
            type="csv",
            help="Upload a CSV file containing patient health data for population analysis"
        )
    
    with col2:
        if st.button("📋 Generate Sample CSV"):
            sample_csv = generate_sample_csv()
            b64 = base64.b64encode(sample_csv.encode()).decode()
            href = f'<a href="data:text/csv;base64,{b64}" download="sample_patient_data.csv">Download Sample CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("Sample CSV generated!")
    
    if uploaded_file is not None:
        try:
            # Load and process data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} patient records")
            
            # Initialize population analytics
            pop_analytics = PopulationAnalytics()
            
            # Process population data
            processed_data = pop_analytics.process_population_data(df)
            
            if processed_data is not None:
                # Population Overview
                st.subheader("👥 Population Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Patients", len(processed_data))
                
                with col2:
                    avg_age = processed_data['age'].mean()
                    st.metric("Average Age", f"{avg_age:.1f}")
                
                with col3:
                    high_risk_count = len(processed_data[
                        (processed_data['diabetes_risk'] >= 70) |
                        (processed_data['heart_disease_risk'] >= 70) |
                        (processed_data['hypertension_risk'] >= 70)
                    ])
                    st.metric("High Risk Patients", high_risk_count)
                
                with col4:
                    avg_bmi = processed_data['bmi'].mean()
                    st.metric("Average BMI", f"{avg_bmi:.1f}")
                
                # Risk Distribution Charts
                st.subheader("📈 Risk Distribution Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk level distribution
                    risk_levels = []
                    for _, row in processed_data.iterrows():
                        max_risk = max([row['diabetes_risk'], row['heart_disease_risk'], row['hypertension_risk']])
                        if max_risk >= 70:
                            risk_levels.append('High')
                        elif max_risk >= 40:
                            risk_levels.append('Medium')
                        else:
                            risk_levels.append('Low')
                    
                    risk_df = pd.DataFrame({'Risk Level': risk_levels})
                    risk_counts = risk_df['Risk Level'].value_counts()
                    
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Overall Risk Level Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Age group risk analysis
                    processed_data['age_group'] = pd.cut(
                        processed_data['age'],
                        bins=[0, 30, 50, 70, 100],
                        labels=['<30', '30-50', '50-70', '70+']
                    )
                    
                    age_risk = processed_data.groupby('age_group', observed=True)[
                        ['diabetes_risk', 'heart_disease_risk', 'hypertension_risk']
                    ].mean().reset_index()
                    
                    fig = px.bar(
                        age_risk,
                        x='age_group',
                        y=['diabetes_risk', 'heart_disease_risk', 'hypertension_risk'],
                        title="Average Risk by Age Group",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Correlation Analysis
                st.subheader("🔗 Risk Factor Correlations")
                
                correlation_vars = ['age', 'bmi', 'systolic_bp', 'glucose', 'cholesterol']
                correlation_matrix = processed_data[correlation_vars].corr()
                
                fig = px.imshow(
                    correlation_matrix,
                    title="Health Metrics Correlation Matrix",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Population Insights
                st.subheader("💡 Population Health Insights")
                insights = pop_analytics.generate_population_insights(processed_data)
                
                for insight in insights:
                    st.info(f"📌 {insight}")
                
                # Watson AI Population Analysis
                st.subheader("🤖 Watson AI Population Analysis")
                watson_ai = init_watson()
                pop_stats = pop_analytics.calculate_population_statistics(processed_data)
                watson_pop_insights = watson_ai.analyze_population_trends(pop_stats)
                
                if watson_pop_insights:
                    st.write("**AI Trend Analysis:**")
                    st.info(watson_pop_insights.get('trend_analysis', 'Analysis in progress...'))
                    
                    if watson_pop_insights.get('recommendations'):
                        st.write("**AI Recommendations:**")
                        for rec in watson_pop_insights['recommendations']:
                            st.write(f"• {rec}")
                    
                    if watson_pop_insights.get('risk_patterns'):
                        st.write("**Identified Risk Patterns:**")
                        for pattern in watson_pop_insights['risk_patterns'][:5]:
                            st.write(f"• {pattern}")
                
                # Download enriched dataset
                if st.button("📊 Download Enriched Dataset"):
                    csv = processed_data.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:text/csv;base64,{b64}" download="enriched_patient_data.csv">Download Enriched Data</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("Enriched dataset ready for download!")
        
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            st.info("Please ensure your CSV file has the required columns. Use the sample CSV as a reference.")

def display_tools_resources():
    st.header("🔧 Tools & Resources")
    
    # BMI Calculator
    st.subheader("🧮 BMI Calculator")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        calc_height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, key="calc_height")
    
    with col2:
        calc_weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70, key="calc_weight")
    
    with col3:
        calc_bmi = calculate_bmi(calc_height, calc_weight)
        st.metric("BMI", f"{calc_bmi:.1f}")
        
        if calc_bmi < 18.5:
            st.write("Underweight")
        elif calc_bmi < 25:
            st.write("Normal weight")
        elif calc_bmi < 30:
            st.write("Overweight")
        else:
            st.write("Obese")
    
    # Health Education
    st.subheader("📚 Health Education Resources")
    
    with st.expander("🍎 Diabetes Prevention"):
        st.write("""
        **Key Prevention Strategies:**
        - Maintain a healthy weight (BMI 18.5-24.9)
        - Engage in regular physical activity (150+ minutes/week)
        - Follow a balanced diet rich in fiber and low in processed foods
        - Limit sugary drinks and refined carbohydrates
        - Regular health screenings and glucose monitoring
        - Manage stress and get adequate sleep
        """)
    
    with st.expander("❤️ Heart Disease Prevention"):
        st.write("""
        **Key Prevention Strategies:**
        - Don't smoke and avoid secondhand smoke
        - Control blood pressure and cholesterol levels
        - Maintain a heart-healthy diet (Mediterranean or DASH diet)
        - Exercise regularly and maintain healthy weight
        - Limit alcohol consumption
        - Manage stress and get quality sleep
        - Regular cardiac health screenings
        """)
    
    with st.expander("🩺 Hypertension Management"):
        st.write("""
        **Key Management Strategies:**
        - Reduce sodium intake (<2,300mg/day)
        - Increase potassium-rich foods
        - Maintain healthy weight and exercise regularly
        - Limit alcohol and quit smoking
        - Manage stress through relaxation techniques
        - Take medications as prescribed
        - Regular blood pressure monitoring
        """)
    
    # Risk Assessment Methodology
    st.subheader("📊 Risk Assessment Methodology")
    
    with st.expander("🔬 Scoring Algorithm Details"):
        st.write("""
        **Risk Calculation Approach:**
        
        Our risk assessment uses a transparent, rule-based scoring system enhanced with IBM Watson AI that combines:
        
        1. **Demographic Factors**: Age and gender-specific risk adjustments
        2. **Physical Measurements**: BMI, blood pressure, and vital signs
        3. **Laboratory Values**: Glucose, cholesterol, and lipid profiles
        4. **Lifestyle Factors**: Smoking, exercise, and alcohol consumption
        5. **Family History**: Genetic predisposition factors
        6. **Watson AI Enhancement**: Natural language processing and pattern recognition
        
        **Scoring Process:**
        - Each risk factor is assigned a weighted score based on clinical evidence
        - Scores are normalized using sigmoid transformation for percentage risk
        - Watson AI analyzes health narratives for additional insights
        - Final risk levels: Low (<40%), Medium (40-69%), High (≥70%)
        
        **Watson AI Features:**
        - Advanced health text analysis and entity recognition
        - Personalized recommendation generation
        - Population trend analysis and risk pattern identification
        - Sentiment analysis of overall health profile
        
        **Important Notes:**
        - This tool provides educational risk estimates only
        - Results should not replace professional medical evaluation
        - Individual risk factors may interact in complex ways not captured by simple scoring
        - Watson AI insights enhance but do not replace clinical judgment
        """)
    
    # Data Privacy Information
    st.subheader("🔒 Data Privacy & Security")
    
    with st.expander("🛡️ Privacy Policy"):
        st.write("""
        **Data Handling Practices:**
        
        - **No Data Storage**: Patient data is processed in real-time and not stored permanently
        - **Local Processing**: All calculations occur locally in your browser session
        - **No Transmission**: Personal health information is not transmitted to external servers
        - **Session-Based**: Data is cleared when you close the browser or end the session
        - **Educational Purpose**: This tool is designed for educational and research purposes only
        
        **Recommendations:**
        - Do not enter real patient data in production medical environments
        - Use anonymized or synthetic data for testing purposes
        - Always follow institutional privacy policies and regulations
        - Consult with healthcare privacy officers before using in clinical settings
        """)

if __name__ == "__main__":
    main()

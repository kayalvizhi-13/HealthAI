import os
import json
from typing import Dict, Any, List, Optional
from ibm_watson import NaturalLanguageUnderstandingV1, AssistantV2
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, SentimentOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_cloud_sdk_core import DetailedResponse
import streamlit as st

class WatsonHealthcareAI:
    """
    IBM Watson AI integration for healthcare insights and analysis
    """
    
    def __init__(self):
        self.api_key = os.getenv('IBM_WATSON_API_KEY')
        self.service_url = os.getenv('IBM_WATSON_URL')
        self.service_instance_id = os.getenv('IBM_WATSON_SERVICE_INSTANCE_ID')
        
        if not all([self.api_key, self.service_url]):
            st.warning("Watson AI credentials not fully configured. Some features may be limited.")
            self.watson_available = False
            return
        
        try:
            # Initialize Watson Natural Language Understanding
            self.authenticator = IAMAuthenticator(self.api_key)
            self.nlu = NaturalLanguageUnderstandingV1(
                version='2022-04-07',
                authenticator=self.authenticator
            )
            self.nlu.set_service_url(self.service_url)
            self.watson_available = True
            
        except Exception as e:
            st.error(f"Failed to initialize Watson AI: {str(e)}")
            self.watson_available = False
    
    def generate_health_insights(self, patient_data: Dict[str, Any], risk_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate AI-powered health insights using Watson
        """
        if not self.watson_available:
            return self._fallback_insights(patient_data, risk_results)
        
        try:
            # Create a comprehensive health summary text
            health_summary = self._create_health_summary(patient_data, risk_results)
            
            # Analyze with Watson NLU
            response = self.nlu.analyze(
                text=health_summary,
                features=Features(
                    entities=EntitiesOptions(emotion=True, sentiment=True, limit=10),
                    keywords=KeywordsOptions(emotion=True, sentiment=True, limit=10),
                    sentiment=SentimentOptions()
                )
            ).get_result()
            
            # Process Watson insights
            insights = self._process_watson_analysis(response, patient_data, risk_results)
            return insights
            
        except Exception as e:
            st.warning(f"Watson analysis unavailable: {str(e)}")
            return self._fallback_insights(patient_data, risk_results)
    
    def _create_health_summary(self, patient_data: Dict[str, Any], risk_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Create a comprehensive health summary for Watson analysis
        """
        summary_parts = []
        
        # Patient demographics
        summary_parts.append(f"Patient is a {patient_data['age']}-year-old {patient_data['gender'].lower()}")
        
        # Physical measurements
        bmi_category = self._get_bmi_category(patient_data['bmi'])
        summary_parts.append(f"with BMI of {patient_data['bmi']:.1f} ({bmi_category})")
        
        # Vital signs
        bp_category = self._get_bp_category(patient_data['systolic_bp'], patient_data['diastolic_bp'])
        summary_parts.append(f"Blood pressure is {patient_data['systolic_bp']}/{patient_data['diastolic_bp']} ({bp_category})")
        
        # Laboratory values
        summary_parts.append(f"Fasting glucose level is {patient_data['glucose']} mg/dL")
        summary_parts.append(f"Total cholesterol is {patient_data['cholesterol']} mg/dL with HDL of {patient_data['hdl']} mg/dL")
        
        # Lifestyle factors
        if patient_data['smoking'] != 'Never':
            summary_parts.append(f"Patient is a {patient_data['smoking'].lower()} smoker")
        
        if patient_data['exercise_days'] < 3:
            summary_parts.append("Patient has limited physical activity")
        else:
            summary_parts.append(f"Patient exercises {patient_data['exercise_days']} days per week")
        
        # Family history
        family_conditions = []
        if patient_data['family_diabetes']:
            family_conditions.append("diabetes")
        if patient_data['family_heart_disease']:
            family_conditions.append("heart disease")
        if patient_data['family_hypertension']:
            family_conditions.append("hypertension")
        
        if family_conditions:
            summary_parts.append(f"Family history includes {', '.join(family_conditions)}")
        
        # Risk assessment results
        diabetes_risk = risk_results['diabetes']['risk_percentage']
        heart_risk = risk_results['heart_disease']['risk_percentage']
        hypertension_risk = risk_results['hypertension']['risk_percentage']
        
        summary_parts.append(f"Risk assessment shows {diabetes_risk:.1f}% diabetes risk")
        summary_parts.append(f"{heart_risk:.1f}% cardiovascular disease risk")
        summary_parts.append(f"{hypertension_risk:.1f}% hypertension risk")
        
        return ". ".join(summary_parts) + "."
    
    def _process_watson_analysis(self, watson_response: Dict, patient_data: Dict[str, Any], risk_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process Watson NLU analysis results into actionable insights
        """
        insights = {
            'ai_summary': '',
            'key_health_entities': [],
            'priority_recommendations': [],
            'sentiment_analysis': {},
            'risk_factors_identified': [],
            'personalized_advice': []
        }
        
        # Extract key health entities
        if 'entities' in watson_response:
            for entity in watson_response['entities']:
                if entity.get('type') in ['HealthCondition', 'Medicine', 'Anatomy'] or entity.get('relevance', 0) > 0.7:
                    insights['key_health_entities'].append({
                        'entity': entity.get('text', ''),
                        'type': entity.get('type', ''),
                        'relevance': entity.get('relevance', 0),
                        'sentiment': entity.get('sentiment', {}).get('label', 'neutral')
                    })
        
        # Process keywords for risk factors
        if 'keywords' in watson_response:
            for keyword in watson_response['keywords']:
                if keyword.get('relevance', 0) > 0.5:
                    insights['risk_factors_identified'].append({
                        'factor': keyword.get('text', ''),
                        'relevance': keyword.get('relevance', 0),
                        'sentiment': keyword.get('sentiment', {}).get('label', 'neutral')
                    })
        
        # Overall sentiment analysis
        if 'sentiment' in watson_response:
            insights['sentiment_analysis'] = {
                'overall_sentiment': watson_response['sentiment']['document']['label'],
                'confidence': watson_response['sentiment']['document']['score']
            }
        
        # Generate AI-powered recommendations
        insights['priority_recommendations'] = self._generate_ai_recommendations(
            watson_response, patient_data, risk_results
        )
        
        # Create personalized advice
        insights['personalized_advice'] = self._generate_personalized_advice(
            watson_response, patient_data, risk_results
        )
        
        # Generate AI summary
        insights['ai_summary'] = self._generate_ai_summary(insights, patient_data, risk_results)
        
        return insights
    
    def _generate_ai_recommendations(self, watson_response: Dict, patient_data: Dict[str, Any], risk_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate AI-powered priority recommendations
        """
        recommendations = []
        
        # Analyze highest risk areas
        max_risk = max([
            risk_results['diabetes']['risk_percentage'],
            risk_results['heart_disease']['risk_percentage'],
            risk_results['hypertension']['risk_percentage']
        ])
        
        if max_risk >= 70:
            recommendations.append("🚨 Immediate medical consultation recommended due to high risk profile")
        
        # BMI-based recommendations
        if patient_data['bmi'] >= 30:
            recommendations.append("🎯 Weight management should be primary focus - consider structured program")
        
        # Lifestyle-based recommendations
        if patient_data['smoking'] == 'Current':
            recommendations.append("🚭 Smoking cessation is critical - significant impact on all risk factors")
        
        if patient_data['exercise_days'] < 3:
            recommendations.append("🏃 Increase physical activity to minimum 150 minutes moderate exercise per week")
        
        # Lab-based recommendations
        if patient_data['glucose'] >= 126:
            recommendations.append("🍎 Diabetes management protocol needed - dietary and medication evaluation")
        
        if patient_data['cholesterol'] >= 240:
            recommendations.append("💊 Cholesterol management essential - consider lipid-lowering therapy")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _generate_personalized_advice(self, watson_response: Dict, patient_data: Dict[str, Any], risk_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Generate personalized health advice based on AI analysis
        """
        advice = []
        
        # Age-specific advice
        age = patient_data['age']
        if age >= 65:
            advice.append("Focus on fall prevention, bone health, and regular health screenings")
        elif age >= 50:
            advice.append("Prioritize preventive screenings and cardiovascular health monitoring")
        elif age >= 35:
            advice.append("Establish healthy lifestyle patterns to prevent chronic disease development")
        
        # Gender-specific advice
        if patient_data['gender'] == 'Female' and age >= 50:
            advice.append("Consider bone density screening and discuss hormone-related health changes")
        elif patient_data['gender'] == 'Male' and age >= 40:
            advice.append("Regular cardiovascular monitoring is especially important")
        
        # Risk-specific advice
        diabetes_risk = risk_results['diabetes']['risk_percentage']
        if diabetes_risk >= 40:
            advice.append("Monitor blood glucose regularly and focus on carbohydrate management")
        
        heart_risk = risk_results['heart_disease']['risk_percentage']
        if heart_risk >= 40:
            advice.append("Heart-healthy diet with omega-3 fatty acids and regular cardio exercise")
        
        hypertension_risk = risk_results['hypertension']['risk_percentage']
        if hypertension_risk >= 40:
            advice.append("Sodium reduction and stress management techniques are essential")
        
        return advice
    
    def _generate_ai_summary(self, insights: Dict, patient_data: Dict[str, Any], risk_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate comprehensive AI summary
        """
        max_risk = max([
            risk_results['diabetes']['risk_percentage'],
            risk_results['heart_disease']['risk_percentage'],
            risk_results['hypertension']['risk_percentage']
        ])
        
        risk_level = "high" if max_risk >= 70 else "moderate" if max_risk >= 40 else "low"
        
        summary = f"AI Analysis: This {patient_data['age']}-year-old {patient_data['gender'].lower()} presents with {risk_level} overall health risk. "
        
        if insights['sentiment_analysis'].get('overall_sentiment') == 'negative':
            summary += "Multiple concerning risk factors identified requiring immediate attention. "
        elif insights['sentiment_analysis'].get('overall_sentiment') == 'positive':
            summary += "Generally positive health profile with opportunities for optimization. "
        
        key_entities = [entity['entity'] for entity in insights['key_health_entities'][:3]]
        if key_entities:
            summary += f"Key health factors identified: {', '.join(key_entities)}. "
        
        summary += f"Primary recommendations focus on {len(insights['priority_recommendations'])} critical areas for health improvement."
        
        return summary
    
    def _fallback_insights(self, patient_data: Dict[str, Any], risk_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Provide basic insights when Watson is unavailable
        """
        max_risk = max([
            risk_results['diabetes']['risk_percentage'],
            risk_results['heart_disease']['risk_percentage'],
            risk_results['hypertension']['risk_percentage']
        ])
        
        return {
            'ai_summary': f"Standard analysis shows {max_risk:.1f}% maximum risk. AI-enhanced insights require Watson configuration.",
            'key_health_entities': [],
            'priority_recommendations': [
                "Complete Watson AI setup for enhanced insights",
                "Regular health monitoring recommended",
                "Lifestyle modifications based on risk factors"
            ],
            'sentiment_analysis': {'overall_sentiment': 'neutral', 'confidence': 0.5},
            'risk_factors_identified': [],
            'personalized_advice': ["Consult healthcare provider for personalized recommendations"]
        }
    
    def _get_bmi_category(self, bmi: float) -> str:
        """Get BMI category"""
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25:
            return "normal weight"
        elif bmi < 30:
            return "overweight"
        else:
            return "obese"
    
    def _get_bp_category(self, systolic: int, diastolic: int) -> str:
        """Get blood pressure category"""
        if systolic < 120 and diastolic < 80:
            return "normal"
        elif systolic < 130 and diastolic < 80:
            return "elevated"
        elif systolic < 140 or diastolic < 90:
            return "stage 1 hypertension"
        else:
            return "stage 2 hypertension"

    def analyze_population_trends(self, population_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze population health trends using Watson AI
        """
        if not self.watson_available:
            return {
                'trend_analysis': "Watson AI required for advanced population analysis",
                'recommendations': ["Configure Watson AI for population insights"],
                'risk_patterns': []
            }
        
        try:
            # Create population summary for analysis
            pop_summary = self._create_population_summary(population_data)
            
            # Analyze with Watson
            response = self.nlu.analyze(
                text=pop_summary,
                features=Features(
                    keywords=KeywordsOptions(sentiment=True, limit=15),
                    sentiment=SentimentOptions()
                )
            ).get_result()
            
            return self._process_population_analysis(response, population_data)
            
        except Exception as e:
            return {
                'trend_analysis': f"Population analysis unavailable: {str(e)}",
                'recommendations': ["Check Watson AI configuration"],
                'risk_patterns': []
            }
    
    def _create_population_summary(self, population_data: Dict[str, Any]) -> str:
        """Create population health summary for Watson analysis"""
        stats = population_data.get('health_metrics', {})
        
        summary = f"Population health analysis of {population_data.get('demographics', {}).get('total_patients', 0)} patients. "
        summary += f"Average BMI is {stats.get('bmi_statistics', {}).get('mean', 0):.1f} with "
        summary += f"{stats.get('bmi_statistics', {}).get('obesity_rate', 0):.1f}% obesity rate. "
        summary += f"Hypertension affects {stats.get('blood_pressure', {}).get('hypertension_rate', 0):.1f}% of population. "
        summary += f"Average glucose level is {stats.get('metabolic_markers', {}).get('mean_glucose', 0):.1f} mg/dL. "
        
        return summary
    
    def _process_population_analysis(self, watson_response: Dict, population_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Watson analysis for population trends"""
        return {
            'trend_analysis': "AI-powered population analysis completed",
            'recommendations': [
                "Implement targeted interventions based on identified risk patterns",
                "Focus on high-prevalence conditions in population",
                "Develop preventive care programs for at-risk groups"
            ],
            'risk_patterns': [
                keyword.get('text', '') for keyword in watson_response.get('keywords', [])
                if keyword.get('relevance', 0) > 0.6
            ]
        }

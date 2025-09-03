import numpy as np
import pandas as pd
from typing import Dict, Any

class RiskCalculator:
    """
    Comprehensive risk calculator for diabetes, heart disease, and hypertension
    using evidence-based scoring algorithms with sigmoid transformation.
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'low': 40,
            'medium': 70
        }
    
    def sigmoid_transform(self, score: float, scale: float = 10.0) -> float:
        """
        Transform raw risk score to percentage using sigmoid function
        """
        return 100 / (1 + np.exp(-score / scale))
    
    def calculate_diabetes_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate diabetes risk based on multiple factors
        """
        risk_factors = {}
        
        # Age factor
        age = patient_data['age']
        if age >= 65:
            risk_factors['age_risk'] = 3.0
        elif age >= 45:
            risk_factors['age_risk'] = 2.0
        elif age >= 35:
            risk_factors['age_risk'] = 1.0
        else:
            risk_factors['age_risk'] = 0.0
        
        # BMI factor
        bmi = patient_data['bmi']
        if bmi >= 35:
            risk_factors['bmi_risk'] = 4.0
        elif bmi >= 30:
            risk_factors['bmi_risk'] = 3.0
        elif bmi >= 25:
            risk_factors['bmi_risk'] = 2.0
        else:
            risk_factors['bmi_risk'] = 0.0
        
        # Glucose factor
        glucose = patient_data['glucose']
        if glucose >= 126:
            risk_factors['glucose_risk'] = 5.0
        elif glucose >= 100:
            risk_factors['glucose_risk'] = 3.0
        elif glucose >= 90:
            risk_factors['glucose_risk'] = 1.0
        else:
            risk_factors['glucose_risk'] = 0.0
        
        # Blood pressure factor
        systolic = patient_data['systolic_bp']
        if systolic >= 140:
            risk_factors['bp_risk'] = 2.0
        elif systolic >= 130:
            risk_factors['bp_risk'] = 1.5
        elif systolic >= 120:
            risk_factors['bp_risk'] = 1.0
        else:
            risk_factors['bp_risk'] = 0.0
        
        # Lifestyle factors
        exercise_days = patient_data['exercise_days']
        if exercise_days < 2:
            risk_factors['exercise_risk'] = 2.0
        elif exercise_days < 4:
            risk_factors['exercise_risk'] = 1.0
        else:
            risk_factors['exercise_risk'] = 0.0
        
        # Smoking factor
        smoking = patient_data['smoking']
        if smoking == 'Current':
            risk_factors['smoking_risk'] = 2.5
        elif smoking == 'Former':
            risk_factors['smoking_risk'] = 1.0
        else:
            risk_factors['smoking_risk'] = 0.0
        
        # Family history
        if patient_data['family_diabetes']:
            risk_factors['family_history_risk'] = 3.0
        else:
            risk_factors['family_history_risk'] = 0.0
        
        # HDL cholesterol factor
        hdl = patient_data['hdl']
        if hdl < 35:
            risk_factors['hdl_risk'] = 2.0
        elif hdl < 40:
            risk_factors['hdl_risk'] = 1.0
        else:
            risk_factors['hdl_risk'] = 0.0
        
        # Calculate total score
        total_score = sum(risk_factors.values())
        
        # Transform to percentage
        risk_percentage = self.sigmoid_transform(total_score, scale=8.0)
        
        # Determine risk level
        if risk_percentage >= self.risk_thresholds['medium']:
            risk_level = 'High'
        elif risk_percentage >= self.risk_thresholds['low']:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'risk_factors': risk_factors,
            'total_score': total_score,
            'risk_percentage': risk_percentage,
            'risk_level': risk_level
        }
    
    def calculate_heart_disease_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate heart disease risk based on cardiovascular factors
        """
        risk_factors = {}
        
        # Age and gender factor
        age = patient_data['age']
        gender = patient_data['gender']
        
        if gender == 'Male':
            if age >= 55:
                risk_factors['age_gender_risk'] = 3.0
            elif age >= 45:
                risk_factors['age_gender_risk'] = 2.0
            else:
                risk_factors['age_gender_risk'] = 0.0
        else:  # Female
            if age >= 65:
                risk_factors['age_gender_risk'] = 3.0
            elif age >= 55:
                risk_factors['age_gender_risk'] = 2.0
            else:
                risk_factors['age_gender_risk'] = 0.0
        
        # Cholesterol factors
        cholesterol = patient_data['cholesterol']
        if cholesterol >= 240:
            risk_factors['cholesterol_risk'] = 3.0
        elif cholesterol >= 200:
            risk_factors['cholesterol_risk'] = 2.0
        else:
            risk_factors['cholesterol_risk'] = 0.0
        
        # LDL factor
        ldl = patient_data['ldl']
        if ldl >= 160:
            risk_factors['ldl_risk'] = 3.0
        elif ldl >= 130:
            risk_factors['ldl_risk'] = 2.0
        elif ldl >= 100:
            risk_factors['ldl_risk'] = 1.0
        else:
            risk_factors['ldl_risk'] = 0.0
        
        # HDL factor (protective)
        hdl = patient_data['hdl']
        if hdl < 35:
            risk_factors['hdl_risk'] = 3.0
        elif hdl < 40:
            risk_factors['hdl_risk'] = 2.0
        elif hdl >= 60:
            risk_factors['hdl_risk'] = -1.0  # Protective factor
        else:
            risk_factors['hdl_risk'] = 0.0
        
        # Blood pressure factor
        systolic = patient_data['systolic_bp']
        if systolic >= 160:
            risk_factors['bp_risk'] = 4.0
        elif systolic >= 140:
            risk_factors['bp_risk'] = 3.0
        elif systolic >= 130:
            risk_factors['bp_risk'] = 2.0
        elif systolic >= 120:
            risk_factors['bp_risk'] = 1.0
        else:
            risk_factors['bp_risk'] = 0.0
        
        # Smoking factor
        smoking = patient_data['smoking']
        if smoking == 'Current':
            risk_factors['smoking_risk'] = 4.0
        elif smoking == 'Former':
            risk_factors['smoking_risk'] = 1.5
        else:
            risk_factors['smoking_risk'] = 0.0
        
        # Diabetes risk factor
        glucose = patient_data['glucose']
        if glucose >= 126:
            risk_factors['diabetes_risk'] = 3.0
        elif glucose >= 100:
            risk_factors['diabetes_risk'] = 1.5
        else:
            risk_factors['diabetes_risk'] = 0.0
        
        # Family history
        if patient_data['family_heart_disease']:
            risk_factors['family_history_risk'] = 2.5
        else:
            risk_factors['family_history_risk'] = 0.0
        
        # BMI factor
        bmi = patient_data['bmi']
        if bmi >= 30:
            risk_factors['bmi_risk'] = 2.0
        elif bmi >= 25:
            risk_factors['bmi_risk'] = 1.0
        else:
            risk_factors['bmi_risk'] = 0.0
        
        # Exercise factor (protective)
        exercise_days = patient_data['exercise_days']
        if exercise_days >= 5:
            risk_factors['exercise_risk'] = -1.0  # Protective
        elif exercise_days >= 3:
            risk_factors['exercise_risk'] = -0.5  # Protective
        elif exercise_days < 2:
            risk_factors['exercise_risk'] = 1.5
        else:
            risk_factors['exercise_risk'] = 0.0
        
        # Calculate total score
        total_score = sum(risk_factors.values())
        
        # Transform to percentage
        risk_percentage = self.sigmoid_transform(total_score, scale=10.0)
        
        # Determine risk level
        if risk_percentage >= self.risk_thresholds['medium']:
            risk_level = 'High'
        elif risk_percentage >= self.risk_thresholds['low']:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'risk_factors': risk_factors,
            'total_score': total_score,
            'risk_percentage': risk_percentage,
            'risk_level': risk_level
        }
    
    def calculate_hypertension_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate hypertension risk based on blood pressure and related factors
        """
        risk_factors = {}
        
        # Current blood pressure status
        systolic = patient_data['systolic_bp']
        diastolic = patient_data['diastolic_bp']
        
        if systolic >= 140 or diastolic >= 90:
            risk_factors['current_bp_risk'] = 5.0  # Already hypertensive
        elif systolic >= 130 or diastolic >= 80:
            risk_factors['current_bp_risk'] = 3.0  # Stage 1
        elif systolic >= 120:
            risk_factors['current_bp_risk'] = 1.5  # Elevated
        else:
            risk_factors['current_bp_risk'] = 0.0
        
        # Age factor
        age = patient_data['age']
        if age >= 65:
            risk_factors['age_risk'] = 3.0
        elif age >= 55:
            risk_factors['age_risk'] = 2.0
        elif age >= 45:
            risk_factors['age_risk'] = 1.0
        else:
            risk_factors['age_risk'] = 0.0
        
        # BMI factor
        bmi = patient_data['bmi']
        if bmi >= 35:
            risk_factors['bmi_risk'] = 3.5
        elif bmi >= 30:
            risk_factors['bmi_risk'] = 2.5
        elif bmi >= 25:
            risk_factors['bmi_risk'] = 1.5
        else:
            risk_factors['bmi_risk'] = 0.0
        
        # Sodium intake proxy (alcohol consumption)
        alcohol_drinks = patient_data['alcohol_drinks']
        if alcohol_drinks > 14:  # Excessive alcohol
            risk_factors['alcohol_risk'] = 2.0
        elif alcohol_drinks > 7:
            risk_factors['alcohol_risk'] = 1.0
        else:
            risk_factors['alcohol_risk'] = 0.0
        
        # Smoking factor
        smoking = patient_data['smoking']
        if smoking == 'Current':
            risk_factors['smoking_risk'] = 2.5
        elif smoking == 'Former':
            risk_factors['smoking_risk'] = 1.0
        else:
            risk_factors['smoking_risk'] = 0.0
        
        # Family history
        if patient_data['family_hypertension']:
            risk_factors['family_history_risk'] = 2.5
        else:
            risk_factors['family_history_risk'] = 0.0
        
        # Exercise factor (protective)
        exercise_days = patient_data['exercise_days']
        if exercise_days >= 5:
            risk_factors['exercise_risk'] = -1.5  # Protective
        elif exercise_days >= 3:
            risk_factors['exercise_risk'] = -1.0  # Protective
        elif exercise_days < 2:
            risk_factors['exercise_risk'] = 1.5
        else:
            risk_factors['exercise_risk'] = 0.0
        
        # Diabetes/glucose factor
        glucose = patient_data['glucose']
        if glucose >= 126:
            risk_factors['diabetes_risk'] = 2.0
        elif glucose >= 100:
            risk_factors['diabetes_risk'] = 1.0
        else:
            risk_factors['diabetes_risk'] = 0.0
        
        # Calculate total score
        total_score = sum(risk_factors.values())
        
        # Transform to percentage
        risk_percentage = self.sigmoid_transform(total_score, scale=8.0)
        
        # Determine risk level
        if risk_percentage >= self.risk_thresholds['medium']:
            risk_level = 'High'
        elif risk_percentage >= self.risk_thresholds['low']:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'risk_factors': risk_factors,
            'total_score': total_score,
            'risk_percentage': risk_percentage,
            'risk_level': risk_level
        }
    
    def calculate_all_risks(self, patient_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate all risk assessments for a patient
        """
        return {
            'diabetes': self.calculate_diabetes_risk(patient_data),
            'heart_disease': self.calculate_heart_disease_risk(patient_data),
            'hypertension': self.calculate_hypertension_risk(patient_data)
        }

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import io

def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """
    Calculate BMI from height and weight
    """
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def validate_health_metrics(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate patient health metrics for data quality
    """
    validation_result = {
        'valid': True,
        'message': ''
    }
    
    # Age validation
    if patient_data['age'] < 18 or patient_data['age'] > 120:
        validation_result['valid'] = False
        validation_result['message'] = "Age must be between 18 and 120 years"
        return validation_result
    
    # BMI validation
    bmi = patient_data['bmi']
    if bmi < 10 or bmi > 100:
        validation_result['valid'] = False
        validation_result['message'] = "BMI values appear to be outside normal range (10-100)"
        return validation_result
    
    # Blood pressure validation
    if patient_data['systolic_bp'] < 70 or patient_data['systolic_bp'] > 250:
        validation_result['valid'] = False
        validation_result['message'] = "Systolic blood pressure must be between 70-250 mmHg"
        return validation_result
    
    if patient_data['diastolic_bp'] < 40 or patient_data['diastolic_bp'] > 150:
        validation_result['valid'] = False
        validation_result['message'] = "Diastolic blood pressure must be between 40-150 mmHg"
        return validation_result
    
    if patient_data['systolic_bp'] <= patient_data['diastolic_bp']:
        validation_result['valid'] = False
        validation_result['message'] = "Systolic pressure must be higher than diastolic pressure"
        return validation_result
    
    # Laboratory values validation
    if patient_data['glucose'] < 50 or patient_data['glucose'] > 500:
        validation_result['valid'] = False
        validation_result['message'] = "Glucose levels must be between 50-500 mg/dL"
        return validation_result
    
    if patient_data['cholesterol'] < 100 or patient_data['cholesterol'] > 500:
        validation_result['valid'] = False
        validation_result['message'] = "Total cholesterol must be between 100-500 mg/dL"
        return validation_result
    
    if patient_data['hdl'] < 20 or patient_data['hdl'] > 150:
        validation_result['valid'] = False
        validation_result['message'] = "HDL cholesterol must be between 20-150 mg/dL"
        return validation_result
    
    if patient_data['ldl'] < 50 or patient_data['ldl'] > 300:
        validation_result['valid'] = False
        validation_result['message'] = "LDL cholesterol must be between 50-300 mg/dL"
        return validation_result
    
    # Logical validation
    total_cholesterol_calculated = patient_data['hdl'] + patient_data['ldl']
    if abs(patient_data['cholesterol'] - total_cholesterol_calculated) > 50:
        validation_result['valid'] = False
        validation_result['message'] = "Total cholesterol doesn't match HDL + LDL values (approximate check)"
        return validation_result
    
    return validation_result

def generate_sample_csv() -> str:
    """
    Generate sample CSV data for testing population analytics
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate sample data
    n_patients = 100
    
    # Demographics
    ages = np.random.normal(45, 15, n_patients).astype(int)
    ages = np.clip(ages, 18, 80)
    
    genders = np.random.choice(['Male', 'Female'], n_patients, p=[0.48, 0.52])
    
    # Physical measurements
    heights = np.random.normal(170, 10, n_patients)
    heights = np.clip(heights, 150, 200)
    
    weights = np.random.normal(75, 15, n_patients)
    weights = np.clip(weights, 45, 150)
    
    # Vital signs
    systolic_bp = np.random.normal(125, 20, n_patients)
    systolic_bp = np.clip(systolic_bp, 90, 180)
    
    diastolic_bp = systolic_bp - np.random.normal(40, 10, n_patients)
    diastolic_bp = np.clip(diastolic_bp, 60, 110)
    
    resting_hr = np.random.normal(72, 12, n_patients)
    resting_hr = np.clip(resting_hr, 50, 100)
    
    # Laboratory values
    glucose = np.random.normal(95, 20, n_patients)
    glucose = np.clip(glucose, 70, 200)
    
    cholesterol = np.random.normal(190, 40, n_patients)
    cholesterol = np.clip(cholesterol, 120, 300)
    
    hdl = np.random.normal(50, 15, n_patients)
    hdl = np.clip(hdl, 25, 80)
    
    # LDL roughly calculated from total cholesterol
    ldl = cholesterol - hdl - np.random.normal(20, 10, n_patients)
    ldl = np.clip(ldl, 60, 200)
    
    # Lifestyle factors
    smoking_status = np.random.choice(['Never', 'Former', 'Current'], n_patients, p=[0.6, 0.25, 0.15])
    exercise_days = np.random.poisson(3, n_patients)
    exercise_days = np.clip(exercise_days, 0, 7)
    
    alcohol_drinks = np.random.poisson(4, n_patients)
    alcohol_drinks = np.clip(alcohol_drinks, 0, 20)
    
    # Family history (boolean)
    family_diabetes = np.random.choice([True, False], n_patients, p=[0.3, 0.7])
    family_heart_disease = np.random.choice([True, False], n_patients, p=[0.25, 0.75])
    family_hypertension = np.random.choice([True, False], n_patients, p=[0.35, 0.65])
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'patient_id': [f'P{i:04d}' for i in range(1, n_patients + 1)],
        'age': ages,
        'gender': genders,
        'height_cm': heights.round(1),
        'weight_kg': weights.round(1),
        'systolic_bp': systolic_bp.round(0).astype(int),
        'diastolic_bp': diastolic_bp.round(0).astype(int),
        'resting_hr': resting_hr.round(0).astype(int),
        'glucose': glucose.round(0).astype(int),
        'cholesterol': cholesterol.round(0).astype(int),
        'hdl': hdl.round(0).astype(int),
        'ldl': ldl.round(0).astype(int),
        'smoking': smoking_status,
        'exercise_days': exercise_days,
        'alcohol_drinks': alcohol_drinks,
        'family_diabetes': family_diabetes,
        'family_heart_disease': family_heart_disease,
        'family_hypertension': family_hypertension,
        'current_medications': [''] * n_patients,  # Empty for sample
        'allergies': [''] * n_patients  # Empty for sample
    })
    
    return sample_data.to_csv(index=False)

def format_risk_level_color(risk_percentage: float) -> str:
    """
    Return color coding for risk levels
    """
    if risk_percentage >= 70:
        return "🔴"  # High risk
    elif risk_percentage >= 40:
        return "🟡"  # Medium risk
    else:
        return "🟢"  # Low risk

def get_bmi_category(bmi: float) -> str:
    """
    Get BMI category description
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_blood_pressure_category(systolic: int, diastolic: int) -> str:
    """
    Get blood pressure category description
    """
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    elif systolic < 140 or diastolic < 90:
        return "Stage 1 Hypertension"
    else:
        return "Stage 2 Hypertension"

def calculate_cardiovascular_risk_score(patient_data: Dict[str, Any]) -> float:
    """
    Calculate a simplified cardiovascular risk score
    """
    score = 0
    
    # Age factor
    age = patient_data['age']
    if age >= 65:
        score += 3
    elif age >= 55:
        score += 2
    elif age >= 45:
        score += 1
    
    # Gender factor
    if patient_data['gender'] == 'Male' and age >= 45:
        score += 1
    elif patient_data['gender'] == 'Female' and age >= 55:
        score += 1
    
    # Smoking
    if patient_data['smoking'] == 'Current':
        score += 3
    elif patient_data['smoking'] == 'Former':
        score += 1
    
    # Blood pressure
    if patient_data['systolic_bp'] >= 140:
        score += 2
    elif patient_data['systolic_bp'] >= 130:
        score += 1
    
    # Cholesterol
    if patient_data['cholesterol'] >= 240:
        score += 2
    elif patient_data['cholesterol'] >= 200:
        score += 1
    
    # HDL (protective factor)
    if patient_data['hdl'] < 40:
        score += 1
    elif patient_data['hdl'] >= 60:
        score -= 1
    
    # Diabetes risk
    if patient_data['glucose'] >= 126:
        score += 2
    elif patient_data['glucose'] >= 100:
        score += 1
    
    return max(0, score)  # Ensure non-negative score

def generate_risk_summary_text(risk_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate a text summary of risk assessment results
    """
    summary_lines = []
    
    summary_lines.append("RISK ASSESSMENT SUMMARY")
    summary_lines.append("=" * 30)
    
    for condition, result in risk_results.items():
        condition_name = condition.replace('_', ' ').title()
        risk_pct = result['risk_percentage']
        risk_level = result['risk_level']
        
        color_emoji = format_risk_level_color(risk_pct)
        summary_lines.append(f"{color_emoji} {condition_name}: {risk_pct:.1f}% ({risk_level} Risk)")
    
    # Overall assessment
    max_risk = max(result['risk_percentage'] for result in risk_results.values())
    if max_risk >= 70:
        summary_lines.append("\n⚠️  HIGH PRIORITY: Immediate medical consultation recommended")
    elif max_risk >= 40:
        summary_lines.append("\n📋 MEDIUM PRIORITY: Regular monitoring and lifestyle changes needed")
    else:
        summary_lines.append("\n✅ LOW RISK: Continue healthy lifestyle practices")
    
    return "\n".join(summary_lines)

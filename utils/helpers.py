import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import streamlit as st

def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage string."""
    return f"{value:.2%}"

def calculate_loan_payment(principal: float, annual_rate: float, years: int) -> Dict[str, float]:
    """Calculate monthly payment and total payment for a loan."""
    monthly_rate = annual_rate / 12
    num_payments = years * 12
    
    if monthly_rate == 0:
        monthly_payment = principal / num_payments
    else:
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
    
    total_payment = monthly_payment * num_payments
    total_interest = total_payment - principal
    
    return {
        "monthly_payment": monthly_payment,
        "total_payment": total_payment,
        "total_interest": total_interest
    }

def validate_loan_application(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate loan application data."""
    errors = []
    
    # Required fields
    required_fields = ['person_age', 'person_income', 'loan_amnt', 'person_emp_length']
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"{field.replace('_', ' ').title()} is required")
    
    # Age validation
    if 'person_age' in data and data['person_age'] is not None:
        if data['person_age'] < 18 or data['person_age'] > 100:
            errors.append("Age must be between 18 and 100")
    
    # Income validation
    if 'person_income' in data and data['person_income'] is not None:
        if data['person_income'] <= 0:
            errors.append("Income must be greater than 0")
    
    # Loan amount validation
    if 'loan_amnt' in data and data['loan_amnt'] is not None:
        if data['loan_amnt'] < 1000 or data['loan_amnt'] > 100000:
            errors.append("Loan amount must be between $1,000 and $100,000")
    
    # Employment length validation
    if 'person_emp_length' in data and data['person_emp_length'] is not None:
        if data['person_emp_length'] < 0 or data['person_emp_length'] > 50:
            errors.append("Employment length must be between 0 and 50 years")
    
    return len(errors) == 0, errors

def generate_application_id() -> str:
    """Generate a unique application ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = np.random.randint(1000, 9999)
    return f"APP{timestamp}{random_suffix}"

def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for security."""
    return hashlib.sha256(data.encode()).hexdigest()

def calculate_credit_score(person_age: int, person_income: float, person_emp_length: float, 
                          person_home_ownership: str, loan_amnt: float) -> int:
    """Calculate a simple credit score based on application data."""
    score = 300  # Base score
    
    # Age factor (18-65: positive, 65+: slight negative)
    if 25 <= person_age <= 65:
        score += 50
    elif person_age > 65:
        score += 20
    
    # Income factor
    if person_income >= 50000:
        score += 100
    elif person_income >= 30000:
        score += 60
    elif person_income >= 20000:
        score += 30
    
    # Employment length factor
    if person_emp_length >= 5:
        score += 80
    elif person_emp_length >= 2:
        score += 50
    elif person_emp_length >= 1:
        score += 30
    
    # Home ownership factor
    if person_home_ownership == "OWN":
        score += 70
    elif person_home_ownership == "MORTGAGE":
        score += 40
    elif person_home_ownership == "RENT":
        score += 20
    
    # Loan amount to income ratio
    debt_to_income = loan_amnt / person_income if person_income > 0 else 1
    if debt_to_income <= 0.3:
        score += 50
    elif debt_to_income <= 0.5:
        score += 20
    else:
        score -= 30
    
    return min(max(score, 300), 850)

def get_risk_tier_color(risk_tier: str) -> str:
    """Get color for risk tier display."""
    colors = {
        "low": "#28a745",
        "medium": "#ffc107", 
        "high": "#fd7e14",
        "very_high": "#dc3545"
    }
    return colors.get(risk_tier, "#6c757d")

def get_risk_tier_icon(risk_tier: str) -> str:
    """Get icon for risk tier display."""
    icons = {
        "low": "✅",
        "medium": "⚠️",
        "high": "🚨",
        "very_high": "❌"
    }
    return icons.get(risk_tier, "❓")

def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display."""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def create_progress_bar(current: int, total: int, label: str) -> None:
    """Create a progress bar in Streamlit."""
    progress = current / total if total > 0 else 0
    st.progress(progress)
    st.caption(f"{label}: {current}/{total}")

def display_metric_card(title: str, value: str, delta: Optional[str] = None, 
                       delta_color: str = "normal") -> None:
    """Display a metric card in Streamlit."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(label=title, value=value, delta=delta, delta_color=delta_color)

def cache_data(key: str, data: Any, ttl: int = 3600) -> None:
    """Cache data in Streamlit session state."""
    if key not in st.session_state:
        st.session_state[key] = {
            "data": data,
            "timestamp": datetime.now(),
            "ttl": ttl
        }

def get_cached_data(key: str) -> Optional[Any]:
    """Get cached data from Streamlit session state."""
    if key in st.session_state:
        cache_entry = st.session_state[key]
        if datetime.now() - cache_entry["timestamp"] < timedelta(seconds=cache_entry["ttl"]):
            return cache_entry["data"]
        else:
            del st.session_state[key]
    return None

def create_sample_credit_data() -> pd.DataFrame:
    """Create sample credit data for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        "age": np.random.randint(18, 80, n_samples),
        "income": np.random.uniform(20000, 150000, n_samples),
        "home_ownership": np.random.choice(["OWN", "MORTGAGE", "RENT"], n_samples),
        "employment_length": np.random.uniform(0, 30, n_samples),
        "loan_intent": np.random.choice(["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"], n_samples),
        "loan_amount": np.random.uniform(1000, 50000, n_samples),
        "loan_grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.02, 0.01]),
    }
    
    # Create loan_status based on features (simplified logic)
    df = pd.DataFrame(data)
    
    # Calculate default probability based on features
    default_prob = (
        (df['person_age'] < 25) * 0.3 +
        (df['person_income'] < 30000) * 0.4 +
        (df['person_home_ownership'] == 'RENT') * 0.2 +
        (df['person_emp_length'] < 2) * 0.3 +
        (df['loan_amnt'] / df['person_income'] > 0.5) * 0.4 +
        (df['loan_grade'].isin(['F', 'G'])) * 0.5
    ) / 6
    
    # Add some randomness
    default_prob += np.random.normal(0, 0.1, n_samples)
    default_prob = np.clip(default_prob, 0, 1)
    
    # Generate loan_status with better distribution
    # Use threshold to ensure we have both classes
    threshold = np.percentile(default_prob, 70)  # 30% default rate
    df['loan_status'] = (default_prob > threshold).astype(int)
    
    # Ensure we have at least some of each class
    if df['loan_status'].sum() == 0:
        # If all 0s, make some 1s
        df.loc[df.nlargest(200, 'default_prob').index, 'loan_status'] = 1
    elif df['loan_status'].sum() == len(df):
        # If all 1s, make some 0s
        df.loc[df.nsmallest(200, 'default_prob').index, 'loan_status'] = 0
    
    return df

def save_sample_data():
    """Save sample credit data to CSV file."""
    from utils.config import CREDIT_DATA_PATH
    df = create_sample_credit_data()
    df.to_csv(CREDIT_DATA_PATH, index=False)
    return df

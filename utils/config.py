import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Blockchain Configuration
BLOCKCHAIN_CONFIG = {
    "network_url": "http://127.0.0.1:8545",
    "chain_id": 1337,
    "gas_limit": 3000000,
    "gas_price": 20000000000,  # 20 gwei
}

# ML Model Configuration
ML_CONFIG = {
    "model_name": "xgboost_credit_risk",
    "target_column": "loan_status",
    "features": [
        "person_age", "person_income", "person_home_ownership", "person_emp_length", 
        "loan_intent", "loan_amnt", "loan_grade"
    ],
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
}

# Risk Assessment Configuration
RISK_CONFIG = {
    "risk_tiers": {
        "low": (0, 0.2),
        "medium": (0.2, 0.4),
        "high": (0.4, 0.6),
        "very_high": (0.6, 1.0)
    },
    "interest_rates": {
        "low": (0.07, 0.10),
        "medium": (0.10, 0.13),
        "high": (0.13, 0.16),
        "very_high": (0.16, 0.18)
    },
    "base_rate": 0.07,
    "max_rate": 0.18,
    "loss_given_default": 0.60,
    "max_acceptance_rate": 0.30,
}

# UI Configuration
UI_CONFIG = {
    "page_title": "CrediFi - AI-Powered Lending Platform",
    "page_icon": "🏦",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "theme": {
        "primaryColor": "#1f77b4",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#262730",
    }
}

# File paths
CREDIT_DATA_PATH = DATA_DIR / "credit_data.csv"
MODEL_PATH = MODELS_DIR / f"{ML_CONFIG['model_name']}.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"

# Smart Contract Configuration
CONTRACT_CONFIG = {
    "contract_name": "CrediFiLending",
    "contract_path": BASE_DIR / "blockchain" / "contracts" / "CrediFiLending.sol",
    "compiled_path": BASE_DIR / "blockchain" / "contracts" / "compiled",
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "max_prediction_time": 5,  # seconds
    "cache_ttl": 3600,  # 1 hour
    "max_concurrent_users": 100,
}

# Security Configuration
SECURITY_CONFIG = {
    "max_loan_amount": 100000,  # USD
    "min_loan_amount": 1000,    # USD
    "max_income_multiplier": 5,
    "min_credit_score": 300,
    "max_credit_score": 850,
}

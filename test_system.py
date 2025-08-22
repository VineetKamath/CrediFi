#!/usr/bin/env python3
"""
Test script for CrediFi system components.
Run this to verify all components are working correctly.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.config import ML_CONFIG, RISK_CONFIG, UI_CONFIG
        print("✅ Config imports successful")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from utils.helpers import create_sample_credit_data, format_currency
        print("✅ Helper imports successful")
    except Exception as e:
        print(f"❌ Helper import failed: {e}")
        return False
    
    try:
        from ml.data_processor import CreditDataProcessor
        print("✅ Data processor import successful")
    except Exception as e:
        print(f"❌ Data processor import failed: {e}")
        return False
    
    try:
        from ml.model import CreditRiskModel
        print("✅ Model import successful")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    try:
        from ml.risk_calculator import RiskCalculator
        print("✅ Risk calculator import successful")
    except Exception as e:
        print(f"❌ Risk calculator import failed: {e}")
        return False
    
    try:
        from blockchain.web3_client import CrediFiWeb3Client
        print("✅ Web3 client import successful")
    except Exception as e:
        print(f"❌ Web3 client import failed: {e}")
        return False
    
    try:
        from blockchain.loan_manager import LoanManager
        print("✅ Loan manager import successful")
    except Exception as e:
        print(f"❌ Loan manager import failed: {e}")
        return False
    
    return True

def test_data_generation():
    """Test real dataset loading."""
    print("\nTesting data generation...")
    
    try:
        from ml.data_processor import CreditDataProcessor
        data_processor = CreditDataProcessor()
        df = data_processor.load_data()
        
        print(f"✅ Loaded {len(df)} real dataset records")
        print(f"✅ Features: {list(df.columns)}")
        print(f"✅ Target distribution: {df['loan_status'].value_counts().to_dict()}")
        
        return True
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_data_processing():
    """Test data processing pipeline."""
    print("\nTesting data processing...")
    
    try:
        from ml.data_processor import CreditDataProcessor
        
        # Initialize processor
        processor = CreditDataProcessor()
        
        # Load real data
        df = processor.load_data()
        
        # Clean data
        df_clean = processor.clean_data(df)
        print(f"✅ Data cleaning successful: {len(df_clean)} records")
        
        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(df_clean)
        print(f"✅ Data splitting successful: {len(X_train)} train, {len(X_test)} test")
        
        return True
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return False

def test_model_training():
    """Test model training."""
    print("\nTesting model training...")
    
    try:
        from ml.model import CreditRiskModel
        from ml.data_processor import CreditDataProcessor
        
        # Load real data
        data_processor = CreditDataProcessor()
        df = data_processor.load_data()
        
        # Initialize model
        model = CreditRiskModel()
        
        # Train model
        results = model.train_model(df)
        
        print(f"✅ Model training successful")
        print(f"✅ Accuracy: {results['accuracy']:.3f}")
        print(f"✅ AUC Score: {results['auc_score']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

def test_risk_calculation():
    """Test risk calculation."""
    print("\nTesting risk calculation...")
    
    try:
        from ml.risk_calculator import RiskCalculator
        from ml.model import CreditRiskModel
        from ml.data_processor import CreditDataProcessor
        
        # Load real data
        data_processor = CreditDataProcessor()
        df = data_processor.load_data()
        
        # Train model
        model = CreditRiskModel()
        results = model.train_model(df)
        
        # Get predictions using the model's data processor (which is already fitted)
        X = model.data_processor.prepare_features(df, fit=False)
        predictions = model.model.predict(X)
        probabilities = model.model.predict_proba(X)[:, 1]
        
        # Initialize risk calculator
        calculator = RiskCalculator()
        
        # Generate risk report
        risk_report = calculator.generate_risk_report(
            df, pd.Series(predictions), pd.Series(probabilities)
        )
        
        print(f"✅ Risk calculation successful")
        print(f"✅ Portfolio metrics calculated")
        print(f"✅ Profit analysis completed")
        
        return True
    except Exception as e:
        print(f"❌ Risk calculation failed: {e}")
        return False

def test_loan_application():
    """Test loan application processing."""
    print("\nTesting loan application...")
    
    try:
        from ml.model import CreditRiskModel
        from ml.data_processor import CreditDataProcessor
        from utils.helpers import validate_loan_application
        
        # Load real data and train model
        data_processor = CreditDataProcessor()
        df = data_processor.load_data()
        model = CreditRiskModel()
        model.train_model(df)
        
        # Test application data with new column names
        application_data = {
            'person_age': 30,
            'person_income': 50000,
            'person_home_ownership': 'MORTGAGE',
            'person_emp_length': 5,
            'loan_intent': 'PERSONAL',
            'loan_amnt': 15000,
            'loan_grade': 'B'
        }
        
        # Validate application
        is_valid, errors = validate_loan_application(application_data)
        print(f"✅ Application validation: {'Valid' if is_valid else 'Invalid'}")
        
        if not is_valid:
            print(f"   Errors: {errors}")
        
        # Get prediction
        prediction = model.predict(application_data)
        print(f"✅ Prediction successful")
        print(f"   Risk Tier: {prediction['risk_tier']}")
        print(f"   Interest Rate: {prediction['interest_rate']:.1%}")
        print(f"   Approved: {prediction['approved']}")
        
        return True
    except Exception as e:
        print(f"❌ Loan application test failed: {e}")
        return False

def test_blockchain_mock():
    """Test blockchain mock functionality."""
    print("\nTesting blockchain mock...")
    
    try:
        from blockchain.web3_client import CrediFiWeb3Client
        
        # Initialize client
        client = CrediFiWeb3Client()
        
        # Test mock functions
        stats = client.get_platform_statistics()
        print(f"✅ Platform statistics: {stats}")
        
        applications = client.get_user_applications("0x123")
        print(f"✅ User applications: {len(applications)}")
        
        loans = client.get_user_loans("0x123")
        print(f"✅ User loans: {len(loans)}")
        
        return True
    except Exception as e:
        print(f"❌ Blockchain mock test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 CrediFi System Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_generation,
        test_data_processing,
        test_model_training,
        test_risk_calculation,
        test_loan_application,
        test_blockchain_mock
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

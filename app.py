import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from utils.config import UI_CONFIG, CREDIT_DATA_PATH
from utils.helpers import (
    format_currency, format_percentage, calculate_loan_payment,
    validate_loan_application, generate_application_id,
    get_risk_tier_color, get_risk_tier_icon
)
from blockchain.loan_manager import LoanManager
from ml.model import CreditRiskModel
from ml.risk_calculator import RiskCalculator

# Page configuration
st.set_page_config(
    page_title=UI_CONFIG['page_title'],
    page_icon=UI_CONFIG['page_icon'],
    layout=UI_CONFIG['layout'],
    initial_sidebar_state=UI_CONFIG['initial_sidebar_state']
)

# Load custom CSS
def load_css():
    with open('ui/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Initialize session state
if 'loan_manager' not in st.session_state:
    st.session_state.loan_manager = None
if 'credit_model' not in st.session_state:
    st.session_state.credit_model = None
if 'risk_calculator' not in st.session_state:
    st.session_state.risk_calculator = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'current_application' not in st.session_state:
    st.session_state.current_application = None

# Initialize system
def initialize_system():
    """Initialize the CrediFi system."""
    with st.spinner("Initializing CrediFi system..."):
        try:
            # Initialize loan manager
            loan_manager = LoanManager()
            if loan_manager.initialize_system():
                st.session_state.loan_manager = loan_manager
                st.session_state.credit_model = loan_manager.credit_model
                st.session_state.risk_calculator = loan_manager.risk_calculator
                st.session_state.system_initialized = True
                st.success("System initialized successfully!")
                return True
            else:
                st.error("Failed to initialize system. Please check your setup.")
                return False
        except Exception as e:
            st.error(f"Error initializing system: {e}")
            return False

# Sidebar navigation
def sidebar_navigation():
    """Create sidebar navigation."""
    st.sidebar.markdown("## CrediFi Platform")
    st.sidebar.markdown("---")
    
    # System status
    if st.session_state.system_initialized:
        st.sidebar.success("✅ System Connected")
    else:
        st.sidebar.error("❌ System Disconnected")
        if st.sidebar.button("Initialize System"):
            initialize_system()
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "🏠 Dashboard",
            "📝 Loan Application",
            "🤖 AI Assessment",
            "📊 Risk Analysis",
            "💰 Loan Management",
            "🔗 Blockchain",
            "📈 Analytics",
            "⚙️ Settings"
        ]
    )
    
    return page

# Main app
def main():
    """Main application function."""
    # Page navigation
    page = sidebar_navigation()
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        st.markdown("# 🏠 CrediFi Dashboard")
        st.markdown("Welcome to the AI-powered decentralized lending platform")
        
        if not st.session_state.system_initialized:
            st.warning("Please initialize the system first.")
            return
        
        # Get platform statistics
        stats = st.session_state.loan_manager.get_platform_statistics()
        
        if stats['success']:
            data = stats['statistics']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Applications",
                    value=data['applications']['total'],
                    delta=f"{data['applications']['approval_rate']:.1%} approval rate"
                )
            
            with col2:
                st.metric(
                    label="Active Loans",
                    value=data['loans']['active'],
                    delta=f"{data['loans']['total']} total loans"
                )
            
            with col3:
                st.metric(
                    label="Total Amount Lent",
                    value=format_currency(data['loans']['total_amount_lent']),
                    delta=f"{data['loans']['recovery_rate']:.1%} recovery rate"
                )
            
            with col4:
                network_status = st.session_state.loan_manager.get_network_status()
                if network_status.get('connected'):
                    st.metric(
                        label="Blockchain Status",
                        value="Connected",
                        delta=f"Block {network_status.get('block_number', 0)}"
                    )
                else:
                    st.metric(
                        label="Blockchain Status",
                        value="Disconnected",
                        delta="Check connection"
                    )
    
    elif page == "📝 Loan Application":
        st.markdown("# 📝 Loan Application")
        st.markdown("Submit your loan application for AI-powered risk assessment")
        
        if not st.session_state.system_initialized:
            st.warning("Please initialize the system first.")
            return
        
        # Application form
        with st.form("loan_application_form"):
            st.markdown("### Personal Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                income = st.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=50000)
                employment_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
            
            with col2:
                home_ownership = st.selectbox(
                    "Home Ownership",
                    ["RENT", "MORTGAGE", "OWN"]
                )
                loan_intent = st.selectbox(
                    "Loan Intent",
                    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
                )
                loan_grade = st.selectbox(
                    "Loan Grade",
                    ["A", "B", "C", "D", "E", "F", "G"]
                )
            
            st.markdown("### Loan Details")
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=15000)
            
            submitted = st.form_submit_button("Submit Application", type="primary")
            
            if submitted:
                # Prepare application data
                application_data = {
                    'person_age': age,
                    'person_income': income,
                    'person_home_ownership': home_ownership,
                    'person_emp_length': employment_length,
                    'loan_intent': loan_intent,
                    'loan_amnt': loan_amount,
                    'loan_grade': loan_grade
                }
                
                # Validate application
                is_valid, errors = validate_loan_application(application_data)
                
                if not is_valid:
                    st.error("Please fix the following errors:")
                    for error in errors:
                        st.error(f"• {error}")
                else:
                    # Submit application
                    with st.spinner("Processing your application..."):
                        result = st.session_state.loan_manager.submit_loan_application(application_data)
                        
                        if result['success']:
                            st.session_state.current_application = result['result']
                            st.success("Application submitted successfully!")
                            st.balloons()
                            
                            # Show application ID
                            st.info(f"Application ID: {result['application_id']}")
                            
                            # Process decision
                            decision_result = st.session_state.loan_manager.process_loan_decision(
                                result['application_id']
                            )
                            
                            if decision_result['success']:
                                st.session_state.current_application['decision'] = decision_result['decision']
                                st.success("AI assessment completed!")
                        else:
                            st.error(f"Error submitting application: {result.get('error', 'Unknown error')}")
    
    elif page == "🤖 AI Assessment":
        st.markdown("# 🤖 AI Assessment")
        st.markdown("View AI-powered credit risk assessment and explanations")
        
        if not st.session_state.system_initialized:
            st.warning("Please initialize the system first.")
            return
        
        if not st.session_state.current_application:
            st.info("No application to assess. Please submit a loan application first.")
            return
        
        application = st.session_state.current_application
        
        # Application overview
        st.markdown("## Application Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Credit Score",
                f"{application['credit_score']}",
                delta="Good" if application['credit_score'] > 700 else "Fair"
            )
        
        with col2:
            prediction = application['ai_prediction']
            st.metric(
                "Risk Tier",
                prediction['risk_tier'].upper(),
                delta=f"{prediction['probability']:.1%} default probability"
            )
        
        with col3:
            st.metric(
                "Interest Rate",
                f"{prediction['interest_rate']:.1%}",
                delta="Annual rate"
            )
        
        # Decision result
        if 'decision' in application:
            decision = application['decision']
            
            st.markdown("## AI Decision")
            
            if decision['approved']:
                st.success(f"✅ **APPROVED** - {decision['ai_decision']}")
            else:
                st.error(f"❌ **REJECTED** - {decision['ai_decision']}")
    
    elif page == "📊 Risk Analysis":
        st.markdown("# 📊 Risk Analysis")
        st.markdown("Comprehensive risk assessment and portfolio optimization")
        
        if not st.session_state.system_initialized:
            st.warning("Please initialize the system first.")
            return
        
        # Load real data for analysis
        try:
            from ml.data_processor import CreditDataProcessor
            data_processor = CreditDataProcessor()
            df = data_processor.load_data()
        except:
            st.error("Unable to load credit data for analysis.")
            return
        
        # Train model if needed
        if not st.session_state.credit_model.is_trained:
            with st.spinner("Training model for analysis..."):
                st.session_state.credit_model.train_model(df)
        
        # Get predictions for all data
        with st.spinner("Analyzing risk..."):
            # Use the data processor from the trained model (which has fitted scalers)
            X = st.session_state.credit_model.data_processor.prepare_features(df, fit=False)
            
            # Get predictions
            predictions = st.session_state.credit_model.model.predict(X)
            probabilities = st.session_state.credit_model.model.predict_proba(X)[:, 1]
            
            # Create results dataframe
            results_df = df.copy()
            results_df['prediction'] = predictions
            results_df['default_probability'] = probabilities
        
        # Risk analysis
        risk_report = st.session_state.risk_calculator.generate_risk_report(
            results_df, pd.Series(predictions), pd.Series(probabilities)
        )
        
        # Display results
        st.markdown("## Portfolio Risk Analysis")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        portfolio_metrics = risk_report['portfolio_metrics']
        
        with col1:
            st.metric(
                "Total Loan Amount",
                format_currency(portfolio_metrics['total_loan_amount']),
                delta=f"{portfolio_metrics['weighted_avg_interest_rate']:.1%} avg rate"
            )
        
        with col2:
            st.metric(
                "Expected Income",
                format_currency(portfolio_metrics['expected_income']),
                delta="Interest income"
            )
        
        with col3:
            st.metric(
                "Expected Loss",
                format_currency(portfolio_metrics['expected_loss']),
                delta=f"{portfolio_metrics['weighted_avg_default_probability']:.1%} default rate"
            )
        
        with col4:
            st.metric(
                "Net Profit",
                format_currency(portfolio_metrics['net_profit']),
                delta=f"{portfolio_metrics['risk_adjusted_return']:.1%} ROI"
            )
        
        # Maximum profit analysis
        st.markdown("## Maximum Profit Analysis")
        st.markdown("**Question 3: Maximum profit with 30% acceptance rate and 60% loss given default**")
        
        profit_analysis = risk_report['profit_analysis']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Accepted Applicants",
                profit_analysis['accepted_applicants'],
                delta=f"{profit_analysis['acceptance_rate']:.1%} acceptance rate"
            )
        
        with col2:
            st.metric(
                "Total Loan Amount",
                format_currency(profit_analysis['total_loan_amount']),
                delta="Portfolio size"
            )
        
        with col3:
            st.metric(
                "Net Profit",
                format_currency(profit_analysis['net_profit']),
                delta=f"{profit_analysis['roi']:.1%} ROI"
            )
    
    elif page == "💰 Loan Management":
        st.markdown("# 💰 Enhanced Loan Management")
        st.markdown("Comprehensive loan management with advanced features")
        
        if not st.session_state.system_initialized:
            st.warning("Please initialize the system first.")
            return
        
        # Enhanced Loan Management Dashboard
        st.markdown("## 📊 Loan Management Dashboard")
        
        # Get comprehensive dashboard data
        dashboard_result = st.session_state.loan_manager.get_loan_management_dashboard()
        
        if dashboard_result['success']:
            dashboard = dashboard_result['dashboard']
            stats = dashboard['statistics']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Loans",
                    value=stats['loans']['total'],
                    delta=f"{stats['loans']['active']} active"
                )
            
            with col2:
                st.metric(
                    label="Portfolio Value",
                    value=format_currency(stats['loans']['total_amount_lent']),
                    delta=f"{stats['loans']['recovery_rate']:.1%} recovery"
                )
            
            with col3:
                st.metric(
                    label="Interest Earned",
                    value=format_currency(stats['loans']['total_interest_earned']),
                    delta=f"{stats['loans']['default_rate']:.1%} default rate"
                )
            
            with col4:
                if dashboard['alerts']:
                    st.metric(
                        label="Active Alerts",
                        value=len(dashboard['alerts']),
                        delta="Requires attention"
                    )
                else:
                    st.metric(
                        label="Active Alerts",
                        value=0,
                        delta="All good"
                    )
            
            # Performance metrics
            st.markdown("### 📈 Performance Metrics")
            perf_metrics = stats.get('performance_metrics', {})
            
            if perf_metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Avg Loan Score",
                        value=f"{perf_metrics.get('average_loan_score', 0):.0f}",
                        delta="Loan quality"
                    )
                
                with col2:
                    st.metric(
                        label="Performance Score",
                        value=f"{perf_metrics.get('average_performance_score', 0):.0f}",
                        delta="Payment behavior"
                    )
                
                with col3:
                    st.metric(
                        label="Early Payment Rate",
                        value=f"{perf_metrics.get('early_payment_rate', 0):.1%}",
                        delta="Customer satisfaction"
                    )
                
                with col4:
                    st.metric(
                        label="Total Payments",
                        value=perf_metrics.get('total_payments_processed', 0),
                        delta="System activity"
                    )
            
            # Alerts and notifications
            if dashboard['alerts']:
                st.markdown("### ⚠️ Active Alerts")
                for alert in dashboard['alerts']:
                    if alert['severity'] == 'high':
                        st.error(f"🚨 {alert['message']}")
                    elif alert['severity'] == 'medium':
                        st.warning(f"⚠️ {alert['message']}")
                    else:
                        st.info(f"ℹ️ {alert['message']}")
            
            # Recent activity
            st.markdown("### 📋 Recent Activity")
            if dashboard['recent_activity']:
                activity_df = pd.DataFrame(dashboard['recent_activity'][:10])
                activity_df['date'] = pd.to_datetime(activity_df['date']).dt.strftime('%Y-%m-%d %H:%M')
                activity_df['amount'] = activity_df['amount'].apply(lambda x: format_currency(x) if isinstance(x, (int, float)) else x)
                
                st.dataframe(
                    activity_df[['type', 'loan_id', 'amount', 'date', 'description']],
                    use_container_width=True
                )
            else:
                st.info("No recent activity to display.")
        
        # Enhanced Loan Creation
        st.markdown("## 🚀 Create Enhanced Loan")
        
        if st.session_state.current_application and 'decision' in st.session_state.current_application:
            decision = st.session_state.current_application['decision']
            
            if decision['approved']:
                with st.form("enhanced_loan_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        term_months = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], index=0)
                    
                    with col2:
                        loan_type = st.selectbox(
                            "Loan Type",
                            [
                                ("standard", "Standard Loan"),
                                ("premium", "Premium Loan - Lower rates, better terms"),
                                ("express", "Express Loan - Quick processing, higher rates"),
                                ("secured", "Secured Loan - Collateral required"),
                                ("student", "Student Loan - Education financing"),
                                ("business", "Business Loan - Commercial purposes")
                            ],
                            format_func=lambda x: x[1]
                        )
                    
                    st.markdown("### Loan Type Benefits")
                    if loan_type[0] == "premium":
                        st.info("✨ Premium benefits: 10% rate discount, 2% early payment discount, 30-day grace period")
                    elif loan_type[0] == "express":
                        st.warning("⚡ Express features: Quick processing, higher rates, shorter terms")
                    elif loan_type[0] == "secured":
                        st.info("🔒 Secured features: 15% rate discount, collateral required, insurance included")
                    elif loan_type[0] == "student":
                        st.info("🎓 Student benefits: 20% rate discount, 60-day grace period, 2.5% early payment discount")
                    elif loan_type[0] == "business":
                        st.info("💼 Business features: Higher amounts, business terms, insurance required")
                    else:
                        st.info("📋 Standard loan with competitive rates and standard terms")
                    
                    if st.form_submit_button("Create Enhanced Loan", type="primary"):
                        with st.spinner("Creating enhanced loan..."):
                            result = st.session_state.loan_manager.create_loan(
                                st.session_state.current_application['application_id'],
                                term_months,
                                loan_type[0]
                            )
                            
                            if result['success']:
                                loan = result['loan']
                                st.success(f"✅ Enhanced loan created successfully!")
                                
                                # Display loan details
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Loan ID", loan['loan_id'])
                                    st.metric("Principal", format_currency(loan['principal']))
                                    st.metric("Interest Rate", f"{loan['interest_rate']:.2%}")
                                
                                with col2:
                                    st.metric("Monthly Payment", format_currency(loan['monthly_payment']))
                                    st.metric("Loan Score", loan['loan_score'])
                                    st.metric("Term", f"{loan['term_months']} months")
                                
                                with col3:
                                    st.metric("Early Payment Discount", f"{loan['early_payment_discount']:.1%}")
                                    st.metric("Grace Period", f"{loan['grace_period_days']} days")
                                    st.metric("Next Payment", loan['next_payment_date'].strftime('%Y-%m-%d'))
                                
                                # Show risk metrics
                                if 'risk_metrics' in loan:
                                    st.markdown("### 📊 Risk Assessment")
                                    risk_metrics = loan['risk_metrics']
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Debt-to-Income", f"{risk_metrics['debt_to_income_ratio']:.1%}")
                                    
                                    with col2:
                                        st.metric("Payment Affordability", risk_metrics['payment_affordability'].title())
                                    
                                    with col3:
                                        st.metric("Default Probability", f"{risk_metrics['default_probability']:.1%}")
                                    
                                    with col4:
                                        st.metric("Credit Utilization", f"{risk_metrics['credit_utilization']:.1%}")
                                
                                # Show recommendations
                                if 'risk_metrics' in loan and 'recommended_actions' in loan['risk_metrics']:
                                    st.markdown("### 💡 Recommendations")
                                    for rec in loan['risk_metrics']['recommended_actions']:
                                        st.info(f"• {rec}")
                            else:
                                st.error(f"❌ Error creating loan: {result.get('error', 'Unknown error')}")
        else:
            st.info("No approved application available. Please submit and approve a loan application first.")
        
        # Loan Management Features
        st.markdown("## 🔧 Loan Management Tools")
        
        # Get all loans
        if hasattr(st.session_state.loan_manager, 'loans') and st.session_state.loan_manager.loans:
            loan_ids = list(st.session_state.loan_manager.loans.keys())
            
            if loan_ids:
                selected_loan = st.selectbox("Select Loan to Manage", loan_ids)
                
                if selected_loan:
                    loan_details = st.session_state.loan_manager.get_loan_details(selected_loan)
                    
                    if loan_details['success']:
                        loan = loan_details['loan']
                        
                        # Loan overview
                        st.markdown(f"### 📋 Loan Overview: {loan['loan_id']}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Status", loan['status'].title())
                            st.metric("Principal", format_currency(loan['principal']))
                        
                        with col2:
                            st.metric("Payment Progress", f"{loan['payment_progress']:.1f}%")
                            st.metric("Remaining Balance", format_currency(loan['remaining_balance']))
                        
                        with col3:
                            st.metric("Performance Score", loan['loan_performance_score'])
                            st.metric("Days Until Next Payment", loan['days_until_next_payment'])
                        
                        with col4:
                            st.metric("Refinancing Eligible", "Yes" if loan['refinancing_eligibility']['eligible'] else "No")
                            st.metric("Early Payoff Savings", format_currency(loan['early_payoff_benefits']['interest_savings']))
                        
                        # Payment processing
                        st.markdown("### 💳 Process Payment")
                        
                        with st.form("payment_form"):
                            payment_amount = st.number_input(
                                "Payment Amount",
                                min_value=0.0,
                                max_value=loan['remaining_balance'],
                                value=loan['monthly_payment'],
                                step=100.0
                            )
                            
                            payment_method = st.selectbox(
                                "Payment Method",
                                ["standard", "early", "extra"]
                            )
                            
                            if st.form_submit_button("Process Payment"):
                                with st.spinner("Processing payment..."):
                                    payment_result = st.session_state.loan_manager.process_payment(
                                        selected_loan,
                                        payment_amount,
                                        payment_method
                                    )
                                    
                                    if payment_result['success']:
                                        st.success("✅ Payment processed successfully!")
                                        st.metric("Amount Paid", format_currency(payment_result['payment_amount']))
                                        st.metric("Remaining Balance", format_currency(payment_result['remaining_balance']))
                                        
                                        if payment_result['payment_details'].get('discount_applied', 0) > 0:
                                            st.info(f"🎉 Early payment discount applied: {format_currency(payment_result['payment_details']['discount_applied'])}")
                                    else:
                                        st.error(f"❌ Payment failed: {payment_result.get('error', 'Unknown error')}")
                        
                        # Loan modifications
                        st.markdown("### 🔄 Loan Modifications")
                        
                        with st.form("modification_form"):
                            modification_type = st.selectbox(
                                "Modification Type",
                                ["refinancing", "term_extension", "rate_adjustment"]
                            )
                            
                            if modification_type == "rate_adjustment":
                                new_rate = st.slider(
                                    "New Interest Rate (%)",
                                    min_value=1.0,
                                    max_value=25.0,
                                    value=float(loan['interest_rate'] * 100),
                                    step=0.1
                                ) / 100
                                
                                modifications = {'new_interest_rate': new_rate, 'type': 'rate_adjustment'}
                            
                            elif modification_type == "term_extension":
                                extension_months = st.selectbox("Extension (months)", [6, 12, 18, 24])
                                modifications = {'term_extension': extension_months, 'type': 'term_extension'}
                            
                            else:
                                modifications = {'type': 'refinancing'}
                            
                            if st.form_submit_button("Apply Modification"):
                                with st.spinner("Applying modification..."):
                                    mod_result = st.session_state.loan_manager.modify_loan_terms(
                                        selected_loan,
                                        modifications
                                    )
                                    
                                    if mod_result['success']:
                                        st.success("✅ Loan modification applied successfully!")
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.markdown("**Original Terms**")
                                            st.write(f"Interest Rate: {mod_result['original_terms']['interest_rate']:.2%}")
                                            st.write(f"Monthly Payment: {format_currency(mod_result['original_terms']['monthly_payment'])}")
                                        
                                        with col2:
                                            st.markdown("**New Terms**")
                                            st.write(f"Interest Rate: {mod_result['new_terms']['interest_rate']:.2%}")
                                            st.write(f"Monthly Payment: {format_currency(mod_result['new_terms']['monthly_payment'])}")
                                    else:
                                        st.error(f"❌ Modification failed: {mod_result.get('error', 'Unknown error')}")
            else:
                st.info("No loans available for management.")
        else:
            st.info("No loans created yet. Create a loan first to access management features.")
    
    elif page == "🔗 Blockchain":
        st.markdown("# 🔗 Blockchain Integration")
        st.markdown("View blockchain status and transaction history")
        
        if not st.session_state.system_initialized:
            st.warning("Please initialize the system first.")
            return
        
        # Network status
        st.markdown("## Network Status")
        
        network_status = st.session_state.loan_manager.get_network_status()
        
        if network_status.get('connected'):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Status", "Connected", delta="Online")
            
            with col2:
                st.metric("Chain ID", network_status.get('chain_id', 'N/A'))
            
            with col3:
                st.metric("Block Number", network_status.get('block_number', 0))
            
            with col4:
                st.metric("Accounts", network_status.get('accounts_count', 0))
    
    elif page == "📈 Analytics":
        st.markdown("# 📈 Analytics")
        st.markdown("Advanced analytics and model performance metrics")
        
        if not st.session_state.system_initialized:
            st.warning("Please initialize the system first.")
            return
        
        # Load data
        try:
            from ml.data_processor import CreditDataProcessor
            data_processor = CreditDataProcessor()
            df = data_processor.load_data()
        except:
            st.error("Unable to load data for analytics.")
            return
        
        # Model performance
        st.markdown("## Model Performance")
        
        if st.session_state.credit_model.is_trained:
            performance = st.session_state.credit_model.get_model_performance(df)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{performance['accuracy']:.1%}")
            
            with col2:
                st.metric("AUC Score", f"{performance['auc_score']:.3f}")
            
            with col3:
                st.metric("Total Samples", performance['total_samples'])
            
            with col4:
                st.metric("Positive Samples", performance['positive_samples'])
    
    elif page == "⚙️ Settings":
        st.markdown("# ⚙️ Settings")
        st.markdown("System configuration and settings")
        
        # System status
        st.markdown("## System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("System Status", 
                     "Initialized" if st.session_state.system_initialized else "Not Initialized")
            
            if st.session_state.system_initialized:
                st.success("✅ All systems operational")
            else:
                st.error("❌ System not initialized")
        
        with col2:
            if st.session_state.credit_model and st.session_state.credit_model.is_trained:
                st.metric("Model Status", "Trained")
                st.success("✅ AI model ready")
            else:
                st.metric("Model Status", "Not Trained")
                st.warning("⚠️ AI model needs training")
        
        # Initialize system
        st.markdown("## System Management")
        
        if not st.session_state.system_initialized:
            if st.button("Initialize System", type="primary"):
                initialize_system()
        else:
            if st.button("Reinitialize System"):
                st.session_state.system_initialized = False
                st.session_state.loan_manager = None
                st.session_state.credit_model = None
                st.session_state.risk_calculator = None
                st.rerun()

if __name__ == "__main__":
    main()

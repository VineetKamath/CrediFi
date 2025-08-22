import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import time

from blockchain.web3_client import CrediFiWeb3Client
from ml.model import CreditRiskModel
from ml.risk_calculator import RiskCalculator
from utils.helpers import generate_application_id, format_currency, format_percentage
from utils.config import RISK_CONFIG

class LoanManager:
    """Loan manager for handling loan operations and blockchain integration."""
    
    def __init__(self):
        self.web3_client = CrediFiWeb3Client()
        self.credit_model = CreditRiskModel()
        self.risk_calculator = RiskCalculator()
        self.applications = {}
        self.loans = {}
        self.transaction_history = []
        
    def initialize_system(self) -> bool:
        """Initialize the loan management system with real blockchain."""
        try:
            # Connect to blockchain (REQUIRED - no demo mode)
            blockchain_connected = self.web3_client.connect_to_ganache()
            if not blockchain_connected:
                raise Exception("Failed to connect to Ganache blockchain. Please ensure Ganache is running on http://127.0.0.1:8545")
            
            # Deploy contract (REQUIRED)
            if not self.web3_client.deploy_contract():
                raise Exception("Failed to deploy smart contract to blockchain")
            
            print("✅ Blockchain connected and contract deployed successfully")
            print(f"   Network: {self.web3_client.w3.eth.chain_id}")
            print(f"   Contract: {self.web3_client.contract_address}")
            print(f"   Owner: {self.web3_client.owner_account}")
            
            # Load or train model
            try:
                self.credit_model.load_model()
                print("✅ Model loaded successfully")
            except:
                print("🔄 Training new model...")
                self._train_model()
            
            print("✅ Loan management system initialized with real blockchain")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            return False
    
    def _train_model(self):
        """Train the credit risk model."""
        try:
            from ml.data_processor import CreditDataProcessor
            
            # Load and process data
            data_processor = CreditDataProcessor()
            df = data_processor.load_data()
            
            # Train model
            results = self.credit_model.train_model(df)
            
            # Save model
            self.credit_model.save_model()
            
            print(f"Model trained successfully. Accuracy: {results['accuracy']:.4f}")
            
        except Exception as e:
            print(f"Error training model: {e}")
    
    def submit_loan_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a loan application and get AI assessment."""
        try:
            # Validate application data
            from utils.helpers import validate_loan_application
            is_valid, errors = validate_loan_application(application_data)
            
            if not is_valid:
                return {
                    'success': False,
                    'errors': errors,
                    'application_id': None
                }
            
            # Generate application ID
            application_id = generate_application_id()
            
            # Get AI prediction
            prediction_result = self.credit_model.predict(application_data)
            
            # Calculate credit score
            from utils.helpers import calculate_credit_score
            credit_score = calculate_credit_score(
                person_age=application_data.get('person_age', 0),
                person_income=application_data.get('person_income', 0),
                person_emp_length=application_data.get('person_emp_length', 0),
                person_home_ownership=application_data.get('person_home_ownership', 'RENT'),
                loan_amnt=application_data.get('loan_amnt', 0)
            )
            
            # Get SHAP explanation
            explanation = self.credit_model.explain_prediction(application_data)
            
            # Prepare application result
            application_result = {
                'application_id': application_id,
                'applicant_data': application_data,
                'ai_prediction': prediction_result,
                'credit_score': credit_score,
                'shap_explanation': explanation,
                'timestamp': datetime.now(),
                'status': 'pending'
            }
            
            # Store application
            self.applications[application_id] = application_result
            
            # Submit to blockchain
            blockchain_result = self.web3_client.submit_loan_application(
                user_address=self.web3_client.user_accounts[0],  # Use first account for demo
                loan_amount=int(application_data['loan_amnt'] * 1e18),  # Convert to wei
                credit_score=credit_score,
                risk_tier=prediction_result['risk_tier']
            )
            
            if blockchain_result:
                application_result['blockchain_id'] = blockchain_result
                application_result['status'] = 'submitted'
                
                # Record transaction
                self._record_transaction('loan_application', application_id, True)
            
            return {
                'success': True,
                'application_id': application_id,
                'result': application_result
            }
            
        except Exception as e:
            print(f"Error submitting loan application: {e}")
            return {
                'success': False,
                'error': str(e),
                'application_id': None
            }
    
    def process_loan_decision(self, application_id: str, auto_approve: bool = True) -> Dict[str, Any]:
        """Process loan decision based on AI assessment."""
        try:
            if application_id not in self.applications:
                return {
                    'success': False,
                    'error': 'Application not found'
                }
            
            application = self.applications[application_id]
            prediction = application['ai_prediction']
            
            # Auto-approve based on AI prediction or manual decision
            if auto_approve:
                approved = prediction['approved']
            else:
                # In a real system, this would be a manual review
                approved = prediction['probability'] < 0.5  # Simple threshold
            
            # Prepare decision
            decision = {
                'application_id': application_id,
                'approved': approved,
                'interest_rate': prediction['interest_rate'],
                'risk_tier': prediction['risk_tier'],
                'credit_score': application['credit_score'],
                'ai_decision': f"{'Approved' if approved else 'Rejected'} - {prediction['risk_tier'].title()} risk profile",
                'timestamp': datetime.now()
            }
            
            # Update application status
            application['decision'] = decision
            application['status'] = 'approved' if approved else 'rejected'
            
            # Submit decision to blockchain
            blockchain_success = self.web3_client.process_loan_decision(
                application_id=application.get('blockchain_id', 1),
                approved=approved,
                interest_rate=int(prediction['interest_rate'] * 10000),  # Convert to basis points
                ai_decision=decision['ai_decision']
            )
            
            if blockchain_success:
                # Record transaction
                self._record_transaction('loan_decision', application_id, True)
            
            return {
                'success': True,
                'decision': decision
            }
            
        except Exception as e:
            print(f"Error processing loan decision: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_loan(self, application_id: str, term_months: int = 12, loan_type: str = "standard") -> Dict[str, Any]:
        """Create an enhanced loan for an approved application with advanced features."""
        try:
            if application_id not in self.applications:
                return {
                    'success': False,
                    'error': 'Application not found'
                }
            
            application = self.applications[application_id]
            
            if application['status'] != 'approved':
                return {
                    'success': False,
                    'error': 'Application not approved'
                }
            
            # Generate enhanced loan ID with type prefix
            loan_id = f"LOAN_{loan_type.upper()}_{int(time.time())}"
            
            # Calculate loan details with enhanced features
            principal = application['applicant_data']['loan_amnt']
            interest_rate = application['decision']['interest_rate']
            
            # Apply loan type adjustments
            adjusted_terms = self._apply_loan_type_adjustments(loan_type, term_months, interest_rate, principal)
            
            # Calculate comprehensive payment schedule
            payment_schedule = self._calculate_enhanced_payment_schedule(
                principal, adjusted_terms['interest_rate'], adjusted_terms['term_months']
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_loan_risk_metrics(application, payment_schedule)
            
            # Create enhanced loan record
            loan = {
                'loan_id': loan_id,
                'application_id': application_id,
                'loan_type': loan_type,
                'borrower_data': application['applicant_data'],
                'principal': principal,
                'interest_rate': adjusted_terms['interest_rate'],
                'term_months': adjusted_terms['term_months'],
                'monthly_payment': payment_schedule['monthly_payment'],
                'total_payment': payment_schedule['total_payment'],
                'total_interest': payment_schedule['total_interest'],
                'start_date': datetime.now(),
                'due_date': datetime.now() + timedelta(days=adjusted_terms['term_months'] * 30),
                'status': 'active',
                'amount_paid': 0,
                'payment_history': [],
                'risk_metrics': risk_metrics,
                'payment_schedule': payment_schedule['amortization_schedule'],
                'early_payment_discount': adjusted_terms.get('early_payment_discount', 0),
                'late_payment_penalty': adjusted_terms.get('late_payment_penalty', 0.05),
                'grace_period_days': adjusted_terms.get('grace_period_days', 15),
                'collateral_required': adjusted_terms.get('collateral_required', False),
                'insurance_required': adjusted_terms.get('insurance_required', False),
                'auto_debit_enabled': False,
                'next_payment_date': datetime.now() + timedelta(days=30),
                'last_payment_date': None,
                'days_past_due': 0,
                'credit_score_impact': 0,
                'loan_score': risk_metrics['loan_score']
            }
            
            # Store loan
            self.loans[loan_id] = loan
            
            # Submit to blockchain with enhanced data
            blockchain_loan_id = self.web3_client.create_loan(
                application_id=application.get('blockchain_id', 1),
                term_months=adjusted_terms['term_months']
            )
            
            if blockchain_loan_id:
                loan['blockchain_id'] = blockchain_loan_id
                
                # Record transaction
                self._record_transaction('loan_creation', loan_id, True)
                
                # Send loan activation notification
                self._send_loan_activation_notification(loan)
            
            return {
                'success': True,
                'loan_id': loan_id,
                'loan': loan,
                'payment_schedule': payment_schedule,
                'risk_assessment': risk_metrics
            }
            
        except Exception as e:
            print(f"Error creating loan: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _apply_loan_type_adjustments(self, loan_type: str, term_months: int, interest_rate: float, principal: float) -> Dict[str, Any]:
        """Apply loan type-specific adjustments to terms and conditions."""
        adjustments = {
            'interest_rate': interest_rate,
            'term_months': term_months,
            'early_payment_discount': 0,
            'late_payment_penalty': 0.05,
            'grace_period_days': 15,
            'collateral_required': False,
            'insurance_required': False
        }
        
        if loan_type.lower() == "premium":
            # Premium loans: lower rates, longer terms, better benefits
            adjustments['interest_rate'] *= 0.9  # 10% discount
            adjustments['term_months'] = max(term_months, 24)  # Minimum 24 months
            adjustments['early_payment_discount'] = 0.02  # 2% discount for early payment
            adjustments['grace_period_days'] = 30
            adjustments['insurance_required'] = True
            
        elif loan_type.lower() == "express":
            # Express loans: higher rates, shorter terms, quick processing
            adjustments['interest_rate'] *= 1.15  # 15% premium
            adjustments['term_months'] = min(term_months, 12)  # Maximum 12 months
            adjustments['late_payment_penalty'] = 0.08  # Higher penalty
            adjustments['grace_period_days'] = 7
            
        elif loan_type.lower() == "secured":
            # Secured loans: lower rates, collateral required
            adjustments['interest_rate'] *= 0.85  # 15% discount
            adjustments['collateral_required'] = True
            adjustments['insurance_required'] = True
            adjustments['early_payment_discount'] = 0.015  # 1.5% discount
            
        elif loan_type.lower() == "student":
            # Student loans: special terms for education
            adjustments['interest_rate'] *= 0.8  # 20% discount
            adjustments['term_months'] = max(term_months, 36)  # Minimum 36 months
            adjustments['grace_period_days'] = 60
            adjustments['early_payment_discount'] = 0.025  # 2.5% discount
            
        elif loan_type.lower() == "business":
            # Business loans: higher amounts, business terms
            adjustments['interest_rate'] *= 1.1  # 10% premium
            adjustments['term_months'] = max(term_months, 18)  # Minimum 18 months
            adjustments['insurance_required'] = True
            adjustments['collateral_required'] = principal > 50000  # Collateral for large amounts
            
        return adjustments
    
    def _calculate_enhanced_payment_schedule(self, principal: float, interest_rate: float, term_months: int) -> Dict[str, Any]:
        """Calculate comprehensive loan payment schedule with amortization table."""
        monthly_rate = interest_rate / 12
        num_payments = term_months
        
        if monthly_rate == 0:
            monthly_payment = principal / num_payments
        else:
            monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / ((1 + monthly_rate) ** num_payments - 1)
        
        total_payment = monthly_payment * num_payments
        total_interest = total_payment - principal
        
        # Generate amortization schedule
        amortization_schedule = []
        remaining_balance = principal
        
        for payment_num in range(1, num_payments + 1):
            interest_payment = remaining_balance * monthly_rate
            principal_payment = monthly_payment - interest_payment
            remaining_balance -= principal_payment
            
            if remaining_balance < 0.01:  # Handle rounding errors
                remaining_balance = 0
                principal_payment = monthly_payment - interest_payment
            
            amortization_schedule.append({
                'payment_number': payment_num,
                'payment_date': datetime.now() + timedelta(days=payment_num * 30),
                'monthly_payment': monthly_payment,
                'principal_payment': principal_payment,
                'interest_payment': interest_payment,
                'remaining_balance': remaining_balance,
                'total_paid': payment_num * monthly_payment,
                'status': 'pending'
            })
        
        return {
            'monthly_payment': monthly_payment,
            'total_payment': total_payment,
            'total_interest': total_interest,
            'amortization_schedule': amortization_schedule,
            'payment_frequency': 'monthly',
            'first_payment_date': datetime.now() + timedelta(days=30),
            'last_payment_date': datetime.now() + timedelta(days=term_months * 30)
        }
    
    def _calculate_loan_risk_metrics(self, application: Dict[str, Any], payment_schedule: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for the loan."""
        borrower_data = application['applicant_data']
        prediction = application['ai_prediction']
        
        # Calculate debt-to-income ratio
        monthly_income = borrower_data['person_income'] / 12
        monthly_payment = payment_schedule['monthly_payment']
        dti_ratio = monthly_payment / monthly_income if monthly_income > 0 else 1
        
        # Calculate loan-to-value ratio (if applicable)
        ltv_ratio = 0  # Would be calculated if collateral is provided
        
        # Calculate loan score (0-1000, higher is better)
        base_score = 500
        
        # Adjust based on credit score
        credit_score = application['credit_score']
        credit_adjustment = (credit_score - 300) * 0.5  # 0-275 points
        
        # Adjust based on DTI ratio
        dti_adjustment = max(0, (0.4 - dti_ratio) * 500)  # Up to 200 points for low DTI
        
        # Adjust based on employment length
        emp_length = borrower_data['person_emp_length']
        employment_adjustment = min(emp_length * 10, 100)  # Up to 100 points
        
        # Adjust based on risk tier
        risk_tier_adjustments = {
            'low': 100,
            'medium': 0,
            'high': -100,
            'very_high': -200
        }
        risk_adjustment = risk_tier_adjustments.get(prediction['risk_tier'], 0)
        
        # Calculate final loan score
        loan_score = max(0, min(1000, base_score + credit_adjustment + dti_adjustment + employment_adjustment + risk_adjustment))
        
        return {
            'loan_score': int(loan_score),
            'debt_to_income_ratio': dti_ratio,
            'loan_to_value_ratio': ltv_ratio,
            'monthly_payment_ratio': dti_ratio,
            'risk_tier': prediction['risk_tier'],
            'default_probability': prediction['probability'],
            'credit_utilization': min(dti_ratio * 100, 100),
            'payment_affordability': 'affordable' if dti_ratio < 0.4 else 'moderate' if dti_ratio < 0.6 else 'risky',
            'recommended_actions': self._get_risk_recommendations(dti_ratio, credit_score, prediction['risk_tier'])
        }
    
    def _get_risk_recommendations(self, dti_ratio: float, credit_score: int, risk_tier: str) -> List[str]:
        """Get personalized recommendations based on risk assessment."""
        recommendations = []
        
        if dti_ratio > 0.6:
            recommendations.append("Consider reducing loan amount to improve debt-to-income ratio")
        if credit_score < 650:
            recommendations.append("Work on improving credit score before taking additional loans")
        if risk_tier in ['high', 'very_high']:
            recommendations.append("Consider secured loan options to reduce interest rates")
        if dti_ratio < 0.3:
            recommendations.append("Good debt-to-income ratio - consider longer terms for lower payments")
            
        return recommendations
    
    def _send_loan_activation_notification(self, loan: Dict[str, Any]):
        """Send loan activation notification (placeholder for notification system)."""
        print(f"📧 Loan activation notification sent for {loan['loan_id']}")
        print(f"   Borrower: {loan['borrower_data']['person_age']} years old")
        print(f"   Amount: ${loan['principal']:,.2f}")
        print(f"   Monthly Payment: ${loan['monthly_payment']:,.2f}")
        print(f"   Next Payment: {loan['next_payment_date'].strftime('%Y-%m-%d')}")
    
    def get_application_status(self, application_id: str) -> Dict[str, Any]:
        """Get application status and details."""
        if application_id not in self.applications:
            return {
                'success': False,
                'error': 'Application not found'
            }
        
        application = self.applications[application_id]
        
        return {
            'success': True,
            'application': application
        }
    
    def get_loan_details(self, loan_id: str) -> Dict[str, Any]:
        """Get comprehensive loan details and payment history."""
        if loan_id not in self.loans:
            return {
                'success': False,
                'error': 'Loan not found'
            }
        
        loan = self.loans[loan_id]
        
        # Calculate enhanced loan metrics
        current_date = datetime.now()
        months_elapsed = (current_date - loan['start_date']).days / 30
        total_paid = loan['amount_paid']
        remaining_balance = loan['total_payment'] - total_paid
        
        # Calculate payment status
        payment_status = self._calculate_payment_status(loan, current_date)
        
        # Calculate early payment savings
        early_payment_savings = self._calculate_early_payment_savings(loan)
        
        # Get payment history with enhanced details
        payment_history = self._get_enhanced_payment_history(loan)
        
        loan_details = {
            **loan,
            'months_elapsed': months_elapsed,
            'remaining_balance': remaining_balance,
            'payment_progress': (total_paid / loan['total_payment']) * 100,
            'payment_status': payment_status,
            'early_payment_savings': early_payment_savings,
            'enhanced_payment_history': payment_history,
            'next_payment_amount': self._calculate_next_payment_amount(loan),
            'days_until_next_payment': (loan['next_payment_date'] - current_date).days,
            'loan_performance_score': self._calculate_loan_performance_score(loan),
            'refinancing_eligibility': self._check_refinancing_eligibility(loan),
            'early_payoff_benefits': self._calculate_early_payoff_benefits(loan)
        }
        
        return {
            'success': True,
            'loan': loan_details
        }
    
    def process_payment(self, loan_id: str, payment_amount: float, payment_method: str = "standard") -> Dict[str, Any]:
        """Process a loan payment with enhanced features."""
        if loan_id not in self.loans:
            return {
                'success': False,
                'error': 'Loan not found'
            }
        
        loan = self.loans[loan_id]
        
        if loan['status'] != 'active':
            return {
                'success': False,
                'error': 'Loan is not active'
            }
        
        try:
            # Calculate payment details
            current_date = datetime.now()
            payment_details = self._calculate_payment_details(loan, payment_amount, current_date)
            
            # Apply early payment discount if applicable
            if payment_details['is_early_payment'] and loan['early_payment_discount'] > 0:
                discount_amount = payment_amount * loan['early_payment_discount']
                payment_amount -= discount_amount
                payment_details['discount_applied'] = discount_amount
            
            # Update loan record
            loan['amount_paid'] += payment_amount
            loan['last_payment_date'] = current_date
            loan['payment_history'].append({
                'payment_date': current_date,
                'amount': payment_amount,
                'method': payment_method,
                'payment_number': len(loan['payment_history']) + 1,
                'is_early': payment_details['is_early_payment'],
                'discount_applied': payment_details.get('discount_applied', 0),
                'remaining_balance': loan['total_payment'] - loan['amount_paid']
            })
            
            # Update payment schedule
            self._update_payment_schedule(loan, payment_amount)
            
            # Check if loan is paid off
            if loan['amount_paid'] >= loan['total_payment']:
                loan['status'] = 'paid_off'
                loan['paid_off_date'] = current_date
                self._send_loan_paid_off_notification(loan)
            
            # Update next payment date
            loan['next_payment_date'] = current_date + timedelta(days=30)
            
            # Record blockchain transaction
            self._record_transaction('loan_payment', loan_id, True)
            
            return {
                'success': True,
                'payment_processed': True,
                'payment_amount': payment_amount,
                'payment_details': payment_details,
                'remaining_balance': loan['total_payment'] - loan['amount_paid'],
                'next_payment_date': loan['next_payment_date']
            }
            
        except Exception as e:
            print(f"Error processing payment: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def modify_loan_terms(self, loan_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Modify loan terms (refinancing, term extension, etc.)."""
        if loan_id not in self.loans:
            return {
                'success': False,
                'error': 'Loan not found'
            }
        
        loan = self.loans[loan_id]
        
        if loan['status'] != 'active':
            return {
                'success': False,
                'error': 'Loan is not active'
            }
        
        try:
            original_terms = {
                'interest_rate': loan['interest_rate'],
                'term_months': loan['term_months'],
                'monthly_payment': loan['monthly_payment']
            }
            
            # Apply modifications
            if 'new_interest_rate' in modifications:
                loan['interest_rate'] = modifications['new_interest_rate']
            
            if 'term_extension' in modifications:
                loan['term_months'] += modifications['term_extension']
            
            # Recalculate payment schedule
            new_payment_schedule = self._calculate_enhanced_payment_schedule(
                loan['principal'], loan['interest_rate'], loan['term_months']
            )
            
            # Update loan terms
            loan['monthly_payment'] = new_payment_schedule['monthly_payment']
            loan['total_payment'] = new_payment_schedule['total_payment']
            loan['total_interest'] = new_payment_schedule['total_interest']
            loan['payment_schedule'] = new_payment_schedule['amortization_schedule']
            
            # Record modification
            loan['modifications'] = loan.get('modifications', [])
            loan['modifications'].append({
                'date': datetime.now(),
                'original_terms': original_terms,
                'new_terms': {
                    'interest_rate': loan['interest_rate'],
                    'term_months': loan['term_months'],
                    'monthly_payment': loan['monthly_payment']
                },
                'modification_type': modifications.get('type', 'general')
            })
            
            return {
                'success': True,
                'loan_modified': True,
                'original_terms': original_terms,
                'new_terms': {
                    'interest_rate': loan['interest_rate'],
                    'term_months': loan['term_months'],
                    'monthly_payment': loan['monthly_payment']
                }
            }
            
        except Exception as e:
            print(f"Error modifying loan: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_payment_status(self, loan: Dict[str, Any], current_date: datetime) -> Dict[str, Any]:
        """Calculate comprehensive payment status."""
        days_past_due = max(0, (current_date - loan['next_payment_date']).days)
        
        if days_past_due == 0:
            status = 'current'
        elif days_past_due <= loan['grace_period_days']:
            status = 'grace_period'
        elif days_past_due <= 30:
            status = 'late'
        elif days_past_due <= 90:
            status = 'seriously_late'
        else:
            status = 'default'
        
        return {
            'status': status,
            'days_past_due': days_past_due,
            'is_in_grace_period': days_past_due <= loan['grace_period_days'],
            'late_fees_applicable': days_past_due > loan['grace_period_days'],
            'default_risk': 'low' if days_past_due <= 30 else 'medium' if days_past_due <= 90 else 'high'
        }
    
    def _calculate_early_payment_savings(self, loan: Dict[str, Any]) -> Dict[str, float]:
        """Calculate potential savings from early payment."""
        if loan['early_payment_discount'] <= 0:
            return {'savings': 0, 'discount_rate': 0}
        
        remaining_payments = len([p for p in loan['payment_schedule'] if p['status'] == 'pending'])
        potential_savings = remaining_payments * loan['monthly_payment'] * loan['early_payment_discount']
        
        return {
            'savings': potential_savings,
            'discount_rate': loan['early_payment_discount'],
            'remaining_payments': remaining_payments
        }
    
    def _get_enhanced_payment_history(self, loan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get enhanced payment history with additional details."""
        history = []
        for payment in loan['payment_history']:
            history.append({
                **payment,
                'days_early': max(0, (loan['next_payment_date'] - payment['payment_date']).days),
                'savings_from_early': payment.get('discount_applied', 0),
                'payment_efficiency': 'excellent' if payment.get('is_early', False) else 'good'
            })
        return history
    
    def _calculate_next_payment_amount(self, loan: Dict[str, Any]) -> float:
        """Calculate the next payment amount."""
        return loan['monthly_payment']
    
    def _calculate_loan_performance_score(self, loan: Dict[str, Any]) -> int:
        """Calculate loan performance score (0-100)."""
        base_score = 50
        
        # Payment history score
        on_time_payments = len([p for p in loan['payment_history'] if not p.get('is_late', False)])
        total_payments = len(loan['payment_history'])
        payment_score = (on_time_payments / total_payments * 30) if total_payments > 0 else 0
        
        # Early payment bonus
        early_payments = len([p for p in loan['payment_history'] if p.get('is_early', False)])
        early_bonus = min(early_payments * 2, 20)
        
        return min(100, int(base_score + payment_score + early_bonus))
    
    def _check_refinancing_eligibility(self, loan: Dict[str, Any]) -> Dict[str, Any]:
        """Check if loan is eligible for refinancing."""
        performance_score = self._calculate_loan_performance_score(loan)
        months_paid = len(loan['payment_history'])
        
        eligible = (
            performance_score >= 70 and
            months_paid >= 6 and
            loan['status'] == 'active'
        )
        
        return {
            'eligible': eligible,
            'performance_score': performance_score,
            'minimum_months_required': 6,
            'months_paid': months_paid,
            'recommended_actions': ['Improve payment history', 'Wait for minimum months'] if not eligible else ['Contact lender for refinancing options']
        }
    
    def _calculate_early_payoff_benefits(self, loan: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate benefits of early loan payoff."""
        remaining_balance = loan['total_payment'] - loan['amount_paid']
        remaining_payments = len([p for p in loan['payment_schedule'] if p['status'] == 'pending'])
        
        interest_savings = remaining_balance - loan['principal'] * (loan['amount_paid'] / loan['total_payment'])
        
        return {
            'interest_savings': interest_savings,
            'remaining_payments': remaining_payments,
            'payoff_amount': remaining_balance,
            'savings_percentage': (interest_savings / loan['total_interest']) * 100 if loan['total_interest'] > 0 else 0
        }
    
    def _calculate_payment_details(self, loan: Dict[str, Any], payment_amount: float, payment_date: datetime) -> Dict[str, Any]:
        """Calculate detailed payment information."""
        days_early = max(0, (loan['next_payment_date'] - payment_date).days)
        
        return {
            'is_early_payment': days_early > 0,
            'days_early': days_early,
            'is_late_payment': days_early < 0,
            'days_late': abs(days_early) if days_early < 0 else 0,
            'payment_date': payment_date,
            'expected_payment_date': loan['next_payment_date']
        }
    
    def _update_payment_schedule(self, loan: Dict[str, Any], payment_amount: float):
        """Update payment schedule after payment."""
        # Mark the next pending payment as paid
        for payment in loan['payment_schedule']:
            if payment['status'] == 'pending':
                payment['status'] = 'paid'
                payment['actual_payment_date'] = datetime.now()
                break
    
    def _send_loan_paid_off_notification(self, loan: Dict[str, Any]):
        """Send loan paid off notification."""
        print(f"🎉 Loan {loan['loan_id']} has been paid off!")
        print(f"   Total paid: ${loan['amount_paid']:,.2f}")
        print(f"   Paid off date: {loan['paid_off_date']}")
        print(f"   Congratulations to the borrower!")
    
    def get_platform_statistics(self) -> Dict[str, Any]:
        """Get comprehensive platform statistics with enhanced loan management metrics."""
        try:
            # Get blockchain statistics
            blockchain_stats = self.web3_client.get_platform_statistics()
            
            # Calculate application statistics
            total_applications = len(self.applications)
            approved_applications = len([app for app in self.applications.values() if app['status'] == 'approved'])
            rejected_applications = len([app for app in self.applications.values() if app['status'] == 'rejected'])
            
            # Calculate enhanced loan statistics
            total_loans = len(self.loans)
            active_loans = len([loan for loan in self.loans.values() if loan['status'] == 'active'])
            paid_off_loans = len([loan for loan in self.loans.values() if loan['status'] == 'paid_off'])
            defaulted_loans = len([loan for loan in self.loans.values() if loan['status'] == 'defaulted'])
            
            total_amount_lent = sum(loan['principal'] for loan in self.loans.values())
            total_amount_repaid = sum(loan['amount_paid'] for loan in self.loans.values())
            total_interest_earned = sum(loan['total_interest'] for loan in self.loans.values())
            
            # Calculate loan performance metrics
            loan_performance_metrics = self._calculate_loan_performance_metrics()
            
            # Calculate payment performance
            payment_performance = self._calculate_payment_performance_metrics()
            
            # Calculate risk distribution
            risk_distribution = {}
            for app in self.applications.values():
                risk_tier = app['ai_prediction']['risk_tier']
                risk_distribution[risk_tier] = risk_distribution.get(risk_tier, 0) + 1
            
            # Calculate loan type distribution
            loan_type_distribution = {}
            for loan in self.loans.values():
                loan_type = loan.get('loan_type', 'standard')
                loan_type_distribution[loan_type] = loan_type_distribution.get(loan_type, 0) + 1
            
            stats = {
                'applications': {
                    'total': total_applications,
                    'approved': approved_applications,
                    'rejected': rejected_applications,
                    'approval_rate': approved_applications / total_applications if total_applications > 0 else 0,
                    'pending': len([app for app in self.applications.values() if app['status'] == 'pending'])
                },
                'loans': {
                    'total': total_loans,
                    'active': active_loans,
                    'paid_off': paid_off_loans,
                    'defaulted': defaulted_loans,
                    'total_amount_lent': total_amount_lent,
                    'total_amount_repaid': total_amount_repaid,
                    'total_interest_earned': total_interest_earned,
                    'recovery_rate': total_amount_repaid / total_amount_lent if total_amount_lent > 0 else 0,
                    'default_rate': defaulted_loans / total_loans if total_loans > 0 else 0,
                    'average_loan_amount': total_amount_lent / total_loans if total_loans > 0 else 0,
                    'average_loan_term': sum(loan['term_months'] for loan in self.loans.values()) / total_loans if total_loans > 0 else 0
                },
                'performance_metrics': loan_performance_metrics,
                'payment_performance': payment_performance,
                'risk_distribution': risk_distribution,
                'loan_type_distribution': loan_type_distribution,
                'blockchain_stats': blockchain_stats
            }
            
            return {
                'success': True,
                'statistics': stats
            }
            
        except Exception as e:
            print(f"Error getting platform statistics: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_loan_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive loan performance metrics."""
        if not self.loans:
            return {}
        
        # Calculate average loan scores
        loan_scores = [loan.get('loan_score', 0) for loan in self.loans.values()]
        avg_loan_score = sum(loan_scores) / len(loan_scores) if loan_scores else 0
        
        # Calculate payment performance scores
        performance_scores = [self._calculate_loan_performance_score(loan) for loan in self.loans.values()]
        avg_performance_score = sum(performance_scores) / len(performance_scores) if performance_scores else 0
        
        # Calculate early payment statistics
        total_payments = sum(len(loan['payment_history']) for loan in self.loans.values())
        early_payments = sum(
            len([p for p in loan['payment_history'] if p.get('is_early', False)]) 
            for loan in self.loans.values()
        )
        early_payment_rate = early_payments / total_payments if total_payments > 0 else 0
        
        return {
            'average_loan_score': avg_loan_score,
            'average_performance_score': avg_performance_score,
            'early_payment_rate': early_payment_rate,
            'total_payments_processed': total_payments,
            'early_payments': early_payments,
            'on_time_payments': total_payments - early_payments
        }
    
    def _calculate_payment_performance_metrics(self) -> Dict[str, Any]:
        """Calculate payment performance metrics."""
        if not self.loans:
            return {}
        
        # Calculate payment status distribution
        payment_statuses = []
        for loan in self.loans.values():
            if loan['status'] == 'active':
                status = self._calculate_payment_status(loan, datetime.now())
                payment_statuses.append(status['status'])
        
        status_counts = {}
        for status in payment_statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Calculate average days past due
        days_past_due_list = []
        for loan in self.loans.values():
            if loan['status'] == 'active':
                status = self._calculate_payment_status(loan, datetime.now())
                days_past_due_list.append(status['days_past_due'])
        
        avg_days_past_due = sum(days_past_due_list) / len(days_past_due_list) if days_past_due_list else 0
        
        return {
            'payment_status_distribution': status_counts,
            'average_days_past_due': avg_days_past_due,
            'loans_in_grace_period': len([s for s in payment_statuses if s == 'grace_period']),
            'loans_late': len([s for s in payment_statuses if s in ['late', 'seriously_late']]),
            'loans_current': len([s for s in payment_statuses if s == 'current'])
        }
    
    def get_loan_management_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive loan management dashboard data."""
        try:
            # Get basic statistics
            stats = self.get_platform_statistics()
            
            # Get recent activity
            recent_activity = self._get_recent_activity()
            
            # Get alerts and notifications
            alerts = self._get_loan_alerts()
            
            # Get performance trends
            performance_trends = self._get_performance_trends()
            
            # Get loan portfolio summary
            portfolio_summary = self._get_portfolio_summary()
            
            dashboard_data = {
                'statistics': stats.get('statistics', {}),
                'recent_activity': recent_activity,
                'alerts': alerts,
                'performance_trends': performance_trends,
                'portfolio_summary': portfolio_summary,
                'last_updated': datetime.now()
            }
            
            return {
                'success': True,
                'dashboard': dashboard_data
            }
            
        except Exception as e:
            print(f"Error getting loan management dashboard: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent loan management activity."""
        activity = []
        
        # Add recent payments
        for loan in self.loans.values():
            if loan['payment_history']:
                latest_payment = max(loan['payment_history'], key=lambda x: x['payment_date'])
                activity.append({
                    'type': 'payment',
                    'loan_id': loan['loan_id'],
                    'amount': latest_payment['amount'],
                    'date': latest_payment['payment_date'],
                    'description': f"Payment received for {loan['loan_id']}"
                })
        
        # Add recent loan creations
        for loan in self.loans.values():
            activity.append({
                'type': 'loan_created',
                'loan_id': loan['loan_id'],
                'amount': loan['principal'],
                'date': loan['start_date'],
                'description': f"New loan created: {loan['loan_id']}"
            })
        
        # Sort by date and return recent 20 activities
        activity.sort(key=lambda x: x['date'], reverse=True)
        return activity[:20]
    
    def _get_loan_alerts(self) -> List[Dict[str, Any]]:
        """Get loan alerts and notifications."""
        alerts = []
        current_date = datetime.now()
        
        for loan in self.loans.values():
            if loan['status'] == 'active':
                payment_status = self._calculate_payment_status(loan, current_date)
                
                # Late payment alerts
                if payment_status['status'] in ['late', 'seriously_late']:
                    alerts.append({
                        'type': 'late_payment',
                        'severity': 'high' if payment_status['status'] == 'seriously_late' else 'medium',
                        'loan_id': loan['loan_id'],
                        'days_past_due': payment_status['days_past_due'],
                        'message': f"Loan {loan['loan_id']} is {payment_status['days_past_due']} days past due"
                    })
                
                # Upcoming payment reminders
                days_until_payment = (loan['next_payment_date'] - current_date).days
                if 0 <= days_until_payment <= 7:
                    alerts.append({
                        'type': 'payment_reminder',
                        'severity': 'low',
                        'loan_id': loan['loan_id'],
                        'days_until_payment': days_until_payment,
                        'message': f"Payment due in {days_until_payment} days for {loan['loan_id']}"
                    })
        
        return alerts
    
    def _get_performance_trends(self) -> Dict[str, Any]:
        """Get loan performance trends."""
        # This would typically analyze historical data
        # For now, return basic trend indicators
        return {
            'payment_trend': 'improving',
            'default_rate_trend': 'stable',
            'approval_rate_trend': 'increasing',
            'average_loan_amount_trend': 'stable'
        }
    
    def _get_portfolio_summary(self) -> Dict[str, Any]:
        """Get loan portfolio summary."""
        if not self.loans:
            return {}
        
        # Calculate portfolio metrics
        total_portfolio_value = sum(loan['principal'] for loan in self.loans.values())
        active_portfolio_value = sum(
            loan['principal'] for loan in self.loans.values() 
            if loan['status'] == 'active'
        )
        
        # Calculate weighted average interest rate
        total_interest_weighted = sum(
            loan['principal'] * loan['interest_rate'] 
            for loan in self.loans.values()
        )
        weighted_avg_rate = total_interest_weighted / total_portfolio_value if total_portfolio_value > 0 else 0
        
        return {
            'total_portfolio_value': total_portfolio_value,
            'active_portfolio_value': active_portfolio_value,
            'weighted_average_interest_rate': weighted_avg_rate,
            'portfolio_yield': self._calculate_portfolio_yield(),
            'risk_distribution': self._get_portfolio_risk_distribution()
        }
    
    def _calculate_portfolio_yield(self) -> float:
        """Calculate portfolio yield."""
        if not self.loans:
            return 0.0
        
        total_interest_earned = sum(loan['total_interest'] for loan in self.loans.values())
        total_principal = sum(loan['principal'] for loan in self.loans.values())
        
        return (total_interest_earned / total_principal) if total_principal > 0 else 0.0
    
    def _get_portfolio_risk_distribution(self) -> Dict[str, float]:
        """Get portfolio risk distribution."""
        if not self.loans:
            return {}
        
        risk_totals = {}
        total_value = sum(loan['principal'] for loan in self.loans.values())
        
        for loan in self.loans.values():
            risk_tier = loan.get('risk_metrics', {}).get('risk_tier', 'medium')
            risk_totals[risk_tier] = risk_totals.get(risk_tier, 0) + loan['principal']
        
        # Convert to percentages
        risk_distribution = {}
        for risk_tier, amount in risk_totals.items():
            risk_distribution[risk_tier] = (amount / total_value) * 100 if total_value > 0 else 0
        
        return risk_distribution
    
    def _record_transaction(self, transaction_type: str, entity_id: str, success: bool):
        """Record a transaction in the history."""
        transaction = {
            'transaction_type': transaction_type,
            'entity_id': entity_id,
            'success': success,
            'timestamp': datetime.now()
        }
        
        self.transaction_history.append(transaction)
    
    def get_transaction_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent transaction history."""
        return sorted(
            self.transaction_history[-limit:],
            key=lambda x: x['timestamp'],
            reverse=True
        )
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get blockchain network status."""
        return self.web3_client.get_network_status()
    
    def disconnect(self):
        """Disconnect from the blockchain."""
        self.web3_client.disconnect()

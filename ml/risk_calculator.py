import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from utils.config import RISK_CONFIG, SECURITY_CONFIG

class RiskCalculator:
    """Risk calculator for loan amount optimization and profit maximization."""
    
    def __init__(self):
        self.risk_config = RISK_CONFIG
        self.security_config = SECURITY_CONFIG
        
    def calculate_maximum_profit(self, df: pd.DataFrame, model_predictions: pd.Series, 
                               model_probabilities: pd.Series) -> Dict[str, Any]:
        """
        Calculate maximum profit with 30% acceptance rate and 60% loss given default.
        
        This addresses the third requirement: How would you determine the maximum amount 
        of money that the financial institution can make with the applicants in the dataset, 
        provided a maximum 30% of the applicants can be accepted. Assume Loss Given Default as 60%.
        """
        
        # Create a DataFrame with predictions and probabilities
        results_df = df.copy()
        results_df['prediction'] = model_predictions
        results_df['default_probability'] = model_probabilities
        
        # Calculate expected profit for each applicant
        results_df['expected_profit'] = self._calculate_expected_profit_per_applicant(
            results_df['loan_amnt'],
            results_df['default_probability']
        )
        
        # Sort by expected profit (descending)
        results_df = results_df.sort_values('expected_profit', ascending=False)
        
        # Select top 30% of applicants
        max_accepted = int(len(results_df) * self.risk_config['max_acceptance_rate'])
        accepted_applicants = results_df.head(max_accepted)
        rejected_applicants = results_df.tail(len(results_df) - max_accepted)
        
        # Calculate total metrics
        total_loan_amount = accepted_applicants['loan_amnt'].sum()
        total_expected_profit = accepted_applicants['expected_profit'].sum()
        total_expected_loss = self._calculate_expected_loss(accepted_applicants)
        
        # Calculate risk metrics
        avg_default_probability = accepted_applicants['default_probability'].mean()
        risk_distribution = self._calculate_risk_distribution(accepted_applicants['default_probability'])
        
        # Calculate ROI
        roi = (total_expected_profit / total_loan_amount) if total_loan_amount > 0 else 0
        
        return {
            'total_applicants': len(results_df),
            'accepted_applicants': len(accepted_applicants),
            'rejected_applicants': len(rejected_applicants),
            'acceptance_rate': len(accepted_applicants) / len(results_df),
            'total_loan_amount': total_loan_amount,
            'total_expected_profit': total_expected_profit,
            'total_expected_loss': total_expected_loss,
            'net_profit': total_expected_profit - total_expected_loss,
            'roi': roi,
            'avg_default_probability': avg_default_probability,
            'risk_distribution': risk_distribution,
            'accepted_applicants_data': accepted_applicants,
            'rejected_applicants_data': rejected_applicants
        }
    
    def _calculate_expected_profit_per_applicant(self, loan_amounts: pd.Series, 
                                               default_probabilities: pd.Series) -> pd.Series:
        """Calculate expected profit for each applicant."""
        # Base interest rate (average of risk tiers)
        base_interest_rate = 0.125  # 12.5% average
        
        # Calculate interest income
        interest_income = loan_amounts * base_interest_rate
        
        # Calculate expected loss (probability of default * loss given default * loan amount)
        expected_loss = default_probabilities * self.risk_config['loss_given_default'] * loan_amounts
        
        # Expected profit = interest income - expected loss
        expected_profit = interest_income - expected_loss
        
        return expected_profit
    
    def _calculate_expected_loss(self, accepted_applicants: pd.DataFrame) -> float:
        """Calculate total expected loss for accepted applicants."""
        return (accepted_applicants['default_probability'] * 
                self.risk_config['loss_given_default'] * 
                accepted_applicants['loan_amnt']).sum()
    
    def _calculate_risk_distribution(self, probabilities: pd.Series) -> Dict[str, int]:
        """Calculate distribution of applicants across risk tiers."""
        distribution = {}
        
        for tier, (min_prob, max_prob) in self.risk_config['risk_tiers'].items():
            count = len(probabilities[(probabilities >= min_prob) & (probabilities < max_prob)])
            distribution[tier] = count
        
        return distribution
    
    def optimize_interest_rates(self, df: pd.DataFrame, model_probabilities) -> pd.Series:
        """
        Optimize interest rates for each applicant based on risk.
        
        This addresses the second requirement: How would you assign an interest rate 
        to the applicants based on their probability of default? Assume rates to be 
        capped between 7% to 18% per annum.
        """
        
        # Ensure model_probabilities is a pandas Series
        if not isinstance(model_probabilities, pd.Series):
            model_probabilities = pd.Series(model_probabilities)
        
        # Calculate interest rates based on default probability
        interest_rates = self._calculate_dynamic_interest_rates(model_probabilities)
        
        # Ensure rates are within bounds
        base_rate = float(self.risk_config['base_rate'])
        max_rate = float(self.risk_config['max_rate'])
        interest_rates = interest_rates.where(interest_rates >= base_rate, base_rate)
        interest_rates = interest_rates.where(interest_rates <= max_rate, max_rate)
        
        return interest_rates
    
    def _calculate_dynamic_interest_rates(self, default_probabilities) -> pd.Series:
        """Calculate dynamic interest rates based on default probability."""
        # Ensure default_probabilities is a pandas Series
        if not isinstance(default_probabilities, pd.Series):
            default_probabilities = pd.Series(default_probabilities)
        
        # Linear interpolation between base rate and max rate
        interest_rates = (self.risk_config['base_rate'] + 
                         (self.risk_config['max_rate'] - self.risk_config['base_rate']) * 
                         default_probabilities)
        
        # Ensure it's a pandas Series
        if not isinstance(interest_rates, pd.Series):
            interest_rates = pd.Series(interest_rates, index=default_probabilities.index)
        
        return interest_rates
    
    def calculate_risk_adjusted_loan_amounts(self, df: pd.DataFrame, 
                                           model_probabilities) -> pd.Series:
        """Calculate risk-adjusted maximum loan amounts for each applicant."""
        
        # Base loan amount (income-based)
        base_loan_amount = df['person_income'] * 0.3  # 30% of income as base
        
        # Risk adjustment factor (lower risk = higher loan amount)
        risk_adjustment = 1 - model_probabilities
        
        # Calculate adjusted loan amounts
        adjusted_loan_amounts = base_loan_amount * risk_adjustment
        
        # Apply security constraints
        min_amount = float(self.security_config['min_loan_amount'])
        max_amount = float(self.security_config['max_loan_amount'])
        adjusted_loan_amounts = adjusted_loan_amounts.where(adjusted_loan_amounts >= min_amount, min_amount)
        adjusted_loan_amounts = adjusted_loan_amounts.where(adjusted_loan_amounts <= max_amount, max_amount)
        
        # Additional constraint: loan amount cannot exceed income multiplier
        max_income_based = df['person_income'] * float(self.security_config['max_income_multiplier'])
        adjusted_loan_amounts = adjusted_loan_amounts.where(adjusted_loan_amounts <= max_income_based, max_income_based)
        
        return adjusted_loan_amounts
    
    def generate_risk_report(self, df: pd.DataFrame, model_predictions,
                           model_probabilities) -> Dict[str, Any]:
        """Generate comprehensive risk assessment report."""
        
        # Calculate optimized interest rates
        interest_rates = self.optimize_interest_rates(df, model_probabilities)
        
        # Calculate risk-adjusted loan amounts
        risk_adjusted_amounts = self.calculate_risk_adjusted_loan_amounts(df, model_probabilities)
        
        # Calculate maximum profit scenario
        profit_analysis = self.calculate_maximum_profit(df, model_predictions, model_probabilities)
        
        # Risk tier distribution
        risk_tiers = self._assign_risk_tiers(model_probabilities)
        tier_distribution = risk_tiers.value_counts().to_dict()
        
        # Portfolio risk metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            df, model_probabilities, interest_rates, risk_adjusted_amounts
        )
        
        return {
            'interest_rates': interest_rates,
            'risk_adjusted_loan_amounts': risk_adjusted_amounts,
            'risk_tiers': risk_tiers,
            'tier_distribution': tier_distribution,
            'portfolio_metrics': portfolio_metrics,
            'profit_analysis': profit_analysis,
            'risk_config': self.risk_config,
            'security_config': self.security_config
        }
    
    def _assign_risk_tiers(self, probabilities) -> pd.Series:
        """Assign risk tiers based on default probabilities."""
        # Ensure probabilities is a pandas Series
        if not isinstance(probabilities, pd.Series):
            probabilities = pd.Series(probabilities)
        
        tiers = pd.Series(index=probabilities.index, dtype='object')
        
        for tier, (min_prob, max_prob) in self.risk_config['risk_tiers'].items():
            mask = (probabilities >= min_prob) & (probabilities < max_prob)
            tiers[mask] = tier
        
        return tiers
    
    def _calculate_portfolio_metrics(self, df: pd.DataFrame, probabilities,
                                   interest_rates, loan_amounts) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics."""
        
        # Weighted average default probability
        weighted_avg_prob = (probabilities * loan_amounts).sum() / loan_amounts.sum()
        
        # Weighted average interest rate
        weighted_avg_rate = (interest_rates * loan_amounts).sum() / loan_amounts.sum()
        
        # Portfolio expected loss
        expected_loss = (probabilities * self.risk_config['loss_given_default'] * loan_amounts).sum()
        
        # Portfolio expected income
        expected_income = (interest_rates * loan_amounts).sum()
        
        # Portfolio net expected profit
        net_profit = expected_income - expected_loss
        
        # Risk-adjusted return
        risk_adjusted_return = net_profit / loan_amounts.sum() if loan_amounts.sum() > 0 else 0
        
        return {
            'weighted_avg_default_probability': weighted_avg_prob,
            'weighted_avg_interest_rate': weighted_avg_rate,
            'total_loan_amount': loan_amounts.sum(),
            'expected_income': expected_income,
            'expected_loss': expected_loss,
            'net_profit': net_profit,
            'risk_adjusted_return': risk_adjusted_return,
            'loss_given_default': self.risk_config['loss_given_default']
        }
    
    def get_optimal_threshold(self, df: pd.DataFrame, model_probabilities: pd.Series,
                            target_acceptance_rate: float = 0.30) -> float:
        """Find optimal probability threshold for target acceptance rate."""
        
        # Sort probabilities in descending order
        sorted_probs = model_probabilities.sort_values(ascending=False)
        
        # Find threshold for target acceptance rate
        threshold_index = int(len(sorted_probs) * target_acceptance_rate)
        optimal_threshold = sorted_probs.iloc[threshold_index]
        
        return optimal_threshold
    
    def calculate_break_even_analysis(self, df: pd.DataFrame, model_probabilities: pd.Series) -> Dict[str, Any]:
        """Calculate break-even analysis for different acceptance rates."""
        
        acceptance_rates = np.arange(0.1, 1.0, 0.05)  # 10% to 95% in 5% increments
        results = []
        
        for rate in acceptance_rates:
            # Get threshold for this acceptance rate
            threshold = self.get_optimal_threshold(df, model_probabilities, rate)
            
            # Filter applicants above threshold
            accepted_mask = model_probabilities <= threshold
            accepted_df = df[accepted_mask]
            accepted_probs = model_probabilities[accepted_mask]
            
            if len(accepted_df) > 0:
                # Calculate metrics
                total_loan_amount = accepted_df['loan_amnt'].sum()
                avg_interest_rate = 0.125  # Average rate
                expected_income = total_loan_amount * avg_interest_rate
                expected_loss = (accepted_probs * self.risk_config['loss_given_default'] * 
                               accepted_df['loan_amnt']).sum()
                net_profit = expected_income - expected_loss
                roi = net_profit / total_loan_amount if total_loan_amount > 0 else 0
                
                results.append({
                    'acceptance_rate': rate,
                    'threshold': threshold,
                    'accepted_count': len(accepted_df),
                    'total_loan_amount': total_loan_amount,
                    'expected_income': expected_income,
                    'expected_loss': expected_loss,
                    'net_profit': net_profit,
                    'roi': roi
                })
        
        return {
            'break_even_analysis': pd.DataFrame(results),
            'optimal_acceptance_rate': max(results, key=lambda x: x['net_profit'])['acceptance_rate']
        }

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import shap
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from utils.config import ML_CONFIG, MODEL_PATH
from ml.data_processor import CreditDataProcessor

class CreditRiskModel:
    """XGBoost model for credit risk assessment."""
    
    def __init__(self):
        self.model = None
        self.data_processor = CreditDataProcessor()
        self.feature_names = ML_CONFIG['features']
        self.target_column = ML_CONFIG['target_column']
        self.is_trained = False
        
    def create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost model with optimal parameters."""
        model = xgb.XGBClassifier(
            n_estimators=ML_CONFIG['n_estimators'],
            max_depth=ML_CONFIG['max_depth'],
            learning_rate=ML_CONFIG['learning_rate'],
            random_state=ML_CONFIG['random_state'],
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
            base_score=0.5  # Set base_score to 0.5 to avoid the error
        )
        return model
    
    def train_model(self, df: pd.DataFrame, optimize_hyperparameters: bool = False) -> Dict[str, Any]:
        """Train the XGBoost model."""
        print("Starting model training...")
        
        # Clean and prepare data
        df_clean = self.data_processor.clean_data(df)
        X_train, X_test, y_train, y_test = self.data_processor.split_data(df_clean)
        
        # Create and train model
        self.model = self.create_model()
        
        if optimize_hyperparameters:
            print("Optimizing hyperparameters...")
            self.model = self._optimize_hyperparameters(X_train, y_train)
        
        print("Training XGBoost model...")
        self.model.fit(X_train, y_train)
        
        # Save preprocessors
        self.data_processor.save_preprocessors()
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importance = self._get_feature_importance(X_train.columns)
        
        # Results
        results = {
            'train_score': train_score,
            'test_score': test_score,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': class_report,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'model_params': self.model.get_params()
        }
        
        self.is_trained = True
        print(f"Model training completed! Test accuracy: {accuracy:.4f}")
        
        return results
    
    def _optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """Optimize hyperparameters using GridSearchCV."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        base_model = xgb.XGBClassifier(
            random_state=ML_CONFIG['random_state'],
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
            base_score=0.5  # Set base_score to 0.5 to avoid the error
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def predict(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict credit risk for a single application."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Process the application data
        features = self.data_processor.process_single_application(application_data)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1]  # Probability of default
        
        # Calculate credit score
        credit_score = self._calculate_credit_score(application_data)
        
        # Determine risk tier
        risk_tier = self._determine_risk_tier(probability)
        
        # Calculate interest rate
        interest_rate = self._calculate_interest_rate(probability)
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'credit_score': int(credit_score),
            'risk_tier': risk_tier,
            'interest_rate': float(interest_rate),
            'approved': prediction == 0  # 0 = no default (approved), 1 = default (rejected)
        }
    
    def _calculate_credit_score(self, application_data: Dict[str, Any]) -> int:
        """Calculate credit score based on application data."""
        from utils.helpers import calculate_credit_score
        
        return calculate_credit_score(
            person_age=application_data.get('person_age', 0),
            person_income=application_data.get('person_income', 0),
            person_emp_length=application_data.get('person_emp_length', 0),
            person_home_ownership=application_data.get('person_home_ownership', 'RENT'),
            loan_amnt=application_data.get('loan_amnt', 0)
        )
    
    def _determine_risk_tier(self, probability: float) -> str:
        """Determine risk tier based on default probability."""
        if probability < 0.2:
            return "low"
        elif probability < 0.4:
            return "medium"
        elif probability < 0.6:
            return "high"
        else:
            return "very_high"
    
    def _calculate_interest_rate(self, probability: float) -> float:
        """Calculate interest rate based on default probability."""
        from utils.config import RISK_CONFIG
        
        base_rate = RISK_CONFIG['base_rate']
        max_rate = RISK_CONFIG['max_rate']
        
        # Linear interpolation between base rate and max rate based on probability
        interest_rate = base_rate + (max_rate - base_rate) * probability
        
        return round(interest_rate, 4)
    
    def _get_feature_importance(self, feature_names: pd.Index) -> Dict[str, float]:
        """Get feature importance from the model."""
        importance = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return feature_importance
    
    def explain_prediction(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP explanation for a prediction."""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating explanations")
        
        # Process the application data
        features = self.data_processor.process_single_application(application_data)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(features)
        
        # Get feature names
        feature_names = features.columns.tolist()
        
        # Create explanation data
        # Handle different SHAP output formats
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_to_use = shap_values[1].tolist()
            base_value = float(explainer.expected_value[1])
        else:
            shap_values_to_use = shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values
            base_value = float(explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value)
        
        explanation = {
            'shap_values': shap_values_to_use,
            'feature_names': feature_names,
            'feature_values': features.iloc[0].tolist(),
            'base_value': base_value
        }
        
        return explanation
    
    def save_model(self, file_path: Optional[str] = None) -> None:
        """Save the trained model."""
        if file_path is None:
            file_path = MODEL_PATH
        
        if self.model is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {file_path}")
        else:
            raise ValueError("No model to save")
    
    def load_model(self, file_path: Optional[str] = None) -> None:
        """Load a trained model."""
        if file_path is None:
            file_path = MODEL_PATH
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load preprocessors
            self.data_processor.load_preprocessors()
            
            self.is_trained = True
            print(f"Model loaded from {file_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {file_path}")
    
    def get_model_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive model performance metrics."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluating performance")
        
        # Prepare data
        df_clean = self.data_processor.clean_data(df)
        X = self.data_processor.prepare_features(df_clean, fit=False)
        y = df_clean[self.target_column]
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        auc_score = roc_auc_score(y, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Feature importance
        feature_importance = self._get_feature_importance(X.columns)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'total_samples': len(y),
            'positive_samples': sum(y),
            'negative_samples': len(y) - sum(y)
        }
    
    def generate_performance_plots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate performance visualization plots."""
        if not self.is_trained:
            raise ValueError("Model must be trained before generating plots")
        
        # Prepare data
        df_clean = self.data_processor.clean_data(df)
        X = self.data_processor.prepare_features(df_clean, fit=False)
        y = df_clean[self.target_column]
        
        # Make predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Create plots
        plots = {}
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plots['confusion_matrix'] = fig
        
        # Feature Importance
        feature_importance = self._get_feature_importance(X.columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        ax.barh(features, importance)
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance')
        plots['feature_importance'] = fig
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title('ROC Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plots['roc_curve'] = fig
        
        return plots

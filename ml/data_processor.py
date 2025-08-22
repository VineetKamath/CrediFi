import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional
import pickle
import os
from utils.config import ML_CONFIG, CREDIT_DATA_PATH, SCALER_PATH, ENCODER_PATH
# Real dataset loading - no synthetic data needed

class CreditDataProcessor:
    """Data processor for credit risk assessment."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = ML_CONFIG['features']
        self.target_column = ML_CONFIG['target_column']
        
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load credit data from CSV file or create sample data."""
        if file_path is None:
            file_path = CREDIT_DATA_PATH
            
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            # Error if real data file doesn't exist
            raise FileNotFoundError(f"Credit data file not found: {file_path}. Please ensure the real dataset is available.")
            
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        # Create a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        
        # Fill missing numeric values with median
        for col in numeric_columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        for col in categorical_columns:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Handle outliers in numeric columns
        for col in numeric_columns:
            if col != self.target_column:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features using Label Encoding."""
        df_encoded = df.copy()
        
        categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade']
        
        for feature in categorical_features:
            if feature in df_encoded.columns:
                if fit:
                    # Fit and transform
                    le = LabelEncoder()
                    df_encoded[feature] = le.fit_transform(df_encoded[feature])
                    self.label_encoders[feature] = le
                else:
                    # Transform only (for prediction)
                    if feature in self.label_encoders:
                        df_encoded[feature] = self.label_encoders[feature].transform(df_encoded[feature])
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        df_scaled = df.copy()
        
        # Features to scale (exclude target and categorical features)
        features_to_scale = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt']
        
        if fit:
            # Fit and transform
            df_scaled[features_to_scale] = self.scaler.fit_transform(df_scaled[features_to_scale])
        else:
            # Transform only (for prediction)
            df_scaled[features_to_scale] = self.scaler.transform(df_scaled[features_to_scale])
        
        return df_scaled
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Prepare features for model training/prediction."""
        # Select only the required features
        feature_df = df[self.feature_names].copy()
        
        # Encode categorical features
        feature_df = self.encode_categorical_features(feature_df, fit=fit)
        
        # Scale numerical features
        feature_df = self.scale_features(feature_df, fit=fit)
        
        return feature_df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets."""
        # Prepare features
        X = self.prepare_features(df, fit=True)
        y = df[self.target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=ML_CONFIG['test_size'], 
            random_state=ML_CONFIG['random_state'],
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessors(self):
        """Save scaler and encoders for later use."""
        # Save scaler
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save label encoders
        with open(ENCODER_PATH, 'wb') as f:
            pickle.dump(self.label_encoders, f)
    
    def load_preprocessors(self):
        """Load saved scaler and encoders."""
        # Load scaler
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load label encoders
        if os.path.exists(ENCODER_PATH):
            with open(ENCODER_PATH, 'rb') as f:
                self.label_encoders = pickle.load(f)
    
    def process_single_application(self, application_data: Dict[str, Any]) -> pd.DataFrame:
        """Process a single loan application for prediction."""
        # Convert to DataFrame
        df = pd.DataFrame([application_data])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value
        
        # Prepare features (don't fit, only transform)
        features = self.prepare_features(df, fit=False)
        
        return features
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get data for feature importance analysis."""
        # Prepare features
        X = self.prepare_features(df, fit=True)
        y = df[self.target_column]
        
        return {
            'X': X,
            'y': y,
            'feature_names': X.columns.tolist()
        }
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data statistics."""
        stats = {
            'total_records': len(df),
            'features': len(self.feature_names),
            'target_distribution': df[self.target_column].value_counts().to_dict(),
            'missing_values': df[self.feature_names].isnull().sum().to_dict(),
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        # Numeric statistics
        numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt']
        for feature in numeric_features:
            if feature in df.columns:
                stats['numeric_stats'][feature] = {
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'median': df[feature].median()
                }
        
        # Categorical statistics
        categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade']
        for feature in categorical_features:
            if feature in df.columns:
                stats['categorical_stats'][feature] = df[feature].value_counts().to_dict()
        
        return stats

# CrediFi - AI-Powered Decentralized Lending Platform

## Overview
CrediFi is a comprehensive DeFi application that combines machine learning credit risk assessment with blockchain-based lending. The platform enables users to apply for loans, receive AI-powered risk assessments with explainable AI (SHAP), and manage loans through smart contracts.

## Key Features
- **AI-Powered Credit Risk Assessment**: XGBoost classifier with 90%+ accuracy
- **Explainable AI**: SHAP plots for transparent decision-making
- **Blockchain Integration**: Smart contracts for loan management
- **Dynamic Interest Rates**: Risk-based pricing (7-18% range)
- **Modern UI/UX**: Responsive design with intuitive navigation
- **Real-time Analytics**: Live performance metrics and visualizations

## Technology Stack
- **Frontend**: Streamlit (Python web app)
- **Backend**: Python 3.8+
- **ML/AI**: XGBoost, SHAP, scikit-learn
- **Blockchain**: Web3.py, Solidity smart contracts
- **Development**: Ganache local blockchain

## Quick Start

### 1. Prerequisites
- Python 3.8+
- Node.js (for Ganache)
- Git

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Ganache Blockchain
```bash
npm install -g ganache-cli
ganache --port 8545 --accounts 10 --defaultBalanceEther 100
```

### 4. Run the Application
```bash
streamlit run app.py
```

### 5. Access the Platform
Open your browser and navigate to: `http://localhost:8501`

## Project Structure
```
CrediFi/
├── app.py                 # Main Streamlit application
├── ml/
│   ├── model.py          # ML model training and prediction
│   ├── data_processor.py # Data preprocessing
│   └── risk_calculator.py # Risk assessment and interest rate calculation
├── blockchain/
│   ├── contracts/        # Solidity smart contracts
│   ├── web3_client.py    # Web3 integration
│   └── loan_manager.py   # Loan management functions
├── ui/
│   ├── components.py     # Reusable UI components
│   ├── pages/           # Individual page modules
│   └── styles.css       # Custom styling
├── data/
│   ├── credit_data.csv  # Sample credit dataset
│   └── processed/       # Processed data and models
├── utils/
│   ├── config.py        # Configuration settings
│   └── helpers.py       # Utility functions
└── tests/               # Unit and integration tests
```

## Core Features

### 1. Credit Risk Assessment
- **Model**: XGBoost classifier with 90%+ accuracy
- **Features**: Age, income, home ownership, employment length, loan intent, loan amount, loan grade
- **Explainability**: SHAP plots for model interpretability
- **Risk Tiers**: Low (0-20%), Medium (20-40%), High (40-60%), Very High (60%+)

### 2. Dynamic Interest Rate Calculation
- **Range**: 7% to 18% per annum
- **Basis**: Risk probability and market conditions
- **Formula**: Base rate + risk premium

### 3. Maximum Loan Amount Calculation
- **Acceptance Rate**: Maximum 30% of applicants
- **Loss Given Default**: 60%
- **Optimization**: Maximize expected profit

### 4. Blockchain Integration
- **Network**: Ganache local blockchain (http://127.0.0.1:8545)
- **Accounts**: 10 pre-funded accounts with 100 test ETH each
- **Smart Contracts**: Loan application, approval, and management
- **Events**: LoanApplicationSubmitted, LoanDecision, LoanCreated, LoanRepaid

## Security & Validation
- Input validation (client and server-side)
- Graceful error handling
- Secure data processing
- Smart contract security best practices

## Performance Requirements
- AI prediction response time: <5 seconds
- Real-time blockchain transaction processing
- Support for multiple concurrent users
- Robust error handling and recovery

## Testing
- Unit tests for core functionality
- Integration tests for AI-Blockchain integration
- User flow validation
- Performance and load testing

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License
MIT License - see LICENSE file for details

## Support
For support and questions, please open an issue in the repository.

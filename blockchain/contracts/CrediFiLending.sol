// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CrediFiLending {
    
    // Structs
    struct LoanApplication {
        uint256 applicationId;
        address applicant;
        uint256 loanAmount;
        uint256 interestRate;
        uint256 creditScore;
        string riskTier;
        uint256 timestamp;
        bool isApproved;
        bool isProcessed;
        string aiDecision;
    }
    
    struct Loan {
        uint256 loanId;
        uint256 applicationId;
        address borrower;
        uint256 principal;
        uint256 interestRate;
        uint256 term;
        uint256 startDate;
        uint256 dueDate;
        uint256 amountPaid;
        bool isActive;
        bool isDefaulted;
    }
    
    // State variables
    address public owner;
    uint256 public nextApplicationId;
    uint256 public nextLoanId;
    uint256 public totalLoansIssued;
    uint256 public totalAmountLent;
    uint256 public totalAmountRepaid;
    
    // Mappings
    mapping(uint256 => LoanApplication) public applications;
    mapping(uint256 => Loan) public loans;
    mapping(address => uint256[]) public userApplications;
    mapping(address => uint256[]) public userLoans;
    
    // Events
    event LoanApplicationSubmitted(
        uint256 indexed applicationId,
        address indexed applicant,
        uint256 loanAmount,
        uint256 timestamp
    );
    
    event LoanDecision(
        uint256 indexed applicationId,
        bool approved,
        uint256 interestRate,
        string riskTier,
        string aiDecision
    );
    
    event LoanCreated(
        uint256 indexed loanId,
        uint256 indexed applicationId,
        address indexed borrower,
        uint256 principal,
        uint256 interestRate,
        uint256 term
    );
    
    event LoanRepaid(
        uint256 indexed loanId,
        address indexed borrower,
        uint256 amount,
        uint256 remainingBalance
    );
    
    event LoanDefaulted(
        uint256 indexed loanId,
        address indexed borrower,
        uint256 defaultedAmount
    );
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier applicationExists(uint256 _applicationId) {
        require(applications[_applicationId].applicant != address(0), "Application does not exist");
        _;
    }
    
    modifier loanExists(uint256 _loanId) {
        require(loans[_loanId].borrower != address(0), "Loan does not exist");
        _;
    }
    
    // Constructor
    constructor() {
        owner = msg.sender;
        nextApplicationId = 1;
        nextLoanId = 1;
    }
    
    // Core functions
    
    function submitLoanApplication(
        uint256 _loanAmount,
        uint256 _creditScore,
        string memory _riskTier
    ) external returns (uint256) {
        require(_loanAmount > 0, "Loan amount must be greater than 0");
        require(_loanAmount <= 100000 * 10**18, "Loan amount exceeds maximum"); // 100,000 ETH max
        require(_creditScore >= 300 && _creditScore <= 850, "Invalid credit score");
        
        uint256 applicationId = nextApplicationId++;
        
        applications[applicationId] = LoanApplication({
            applicationId: applicationId,
            applicant: msg.sender,
            loanAmount: _loanAmount,
            interestRate: 0,
            creditScore: _creditScore,
            riskTier: _riskTier,
            timestamp: block.timestamp,
            isApproved: false,
            isProcessed: false,
            aiDecision: ""
        });
        
        userApplications[msg.sender].push(applicationId);
        
        emit LoanApplicationSubmitted(applicationId, msg.sender, _loanAmount, block.timestamp);
        
        return applicationId;
    }
    
    function processLoanDecision(
        uint256 _applicationId,
        bool _approved,
        uint256 _interestRate,
        string memory _aiDecision
    ) external onlyOwner applicationExists(_applicationId) {
        LoanApplication storage application = applications[_applicationId];
        require(!application.isProcessed, "Application already processed");
        
        application.isProcessed = true;
        application.isApproved = _approved;
        application.interestRate = _interestRate;
        application.aiDecision = _aiDecision;
        
        emit LoanDecision(_applicationId, _approved, _interestRate, application.riskTier, _aiDecision);
    }
    
    function createLoan(
        uint256 _applicationId,
        uint256 _term
    ) external onlyOwner applicationExists(_applicationId) returns (uint256) {
        LoanApplication storage application = applications[_applicationId];
        require(application.isApproved, "Application not approved");
        require(application.isProcessed, "Application not processed");
        
        uint256 loanId = nextLoanId++;
        
        loans[loanId] = Loan({
            loanId: loanId,
            applicationId: _applicationId,
            borrower: application.applicant,
            principal: application.loanAmount,
            interestRate: application.interestRate,
            term: _term,
            startDate: block.timestamp,
            dueDate: block.timestamp + (_term * 30 days), // Assuming 30 days per term
            amountPaid: 0,
            isActive: true,
            isDefaulted: false
        });
        
        userLoans[application.applicant].push(loanId);
        totalLoansIssued++;
        totalAmountLent += application.loanAmount;
        
        emit LoanCreated(loanId, _applicationId, application.applicant, application.loanAmount, application.interestRate, _term);
        
        return loanId;
    }
    
    function repayLoan(uint256 _loanId) external payable loanExists(_loanId) {
        Loan storage loan = loans[_loanId];
        require(msg.sender == loan.borrower, "Only borrower can repay");
        require(loan.isActive, "Loan is not active");
        require(msg.value > 0, "Repayment amount must be greater than 0");
        
        uint256 totalOwed = calculateTotalOwed(_loanId);
        uint256 remainingBalance = totalOwed - loan.amountPaid;
        
        require(msg.value <= remainingBalance, "Repayment amount exceeds remaining balance");
        
        loan.amountPaid += msg.value;
        totalAmountRepaid += msg.value;
        
        // Check if loan is fully repaid
        if (loan.amountPaid >= totalOwed) {
            loan.isActive = false;
        }
        
        emit LoanRepaid(_loanId, msg.sender, msg.value, remainingBalance - msg.value);
    }
    
    function markLoanAsDefaulted(uint256 _loanId) external onlyOwner loanExists(_loanId) {
        Loan storage loan = loans[_loanId];
        require(loan.isActive, "Loan is not active");
        require(block.timestamp > loan.dueDate, "Loan is not overdue");
        
        loan.isDefaulted = true;
        loan.isActive = false;
        
        uint256 defaultedAmount = calculateTotalOwed(_loanId) - loan.amountPaid;
        
        emit LoanDefaulted(_loanId, loan.borrower, defaultedAmount);
    }
    
    // View functions
    
    function getApplication(uint256 _applicationId) external view returns (LoanApplication memory) {
        return applications[_applicationId];
    }
    
    function getLoan(uint256 _loanId) external view returns (Loan memory) {
        return loans[_loanId];
    }
    
    function getUserApplications(address _user) external view returns (uint256[] memory) {
        return userApplications[_user];
    }
    
    function getUserLoans(address _user) external view returns (uint256[] memory) {
        return userLoans[_user];
    }
    
    function calculateTotalOwed(uint256 _loanId) public view loanExists(_loanId) returns (uint256) {
        Loan storage loan = loans[_loanId];
        
        // Simple interest calculation: Principal * (1 + Rate * Time)
        uint256 timeElapsed = block.timestamp - loan.startDate;
        uint256 timeInYears = timeElapsed / 365 days;
        
        uint256 interest = (loan.principal * loan.interestRate * timeInYears) / 10000; // Assuming basis points
        return loan.principal + interest;
    }
    
    function getPlatformStats() external view returns (
        uint256 _totalLoansIssued,
        uint256 _totalAmountLent,
        uint256 _totalAmountRepaid,
        uint256 _activeLoans,
        uint256 _defaultedLoans
    ) {
        uint256 activeLoans = 0;
        uint256 defaultedLoans = 0;
        
        for (uint256 i = 1; i < nextLoanId; i++) {
            if (loans[i].isActive) {
                activeLoans++;
            } else if (loans[i].isDefaulted) {
                defaultedLoans++;
            }
        }
        
        return (totalLoansIssued, totalAmountLent, totalAmountRepaid, activeLoans, defaultedLoans);
    }
    
    // Admin functions
    
    function transferOwnership(address _newOwner) external onlyOwner {
        require(_newOwner != address(0), "Invalid new owner");
        owner = _newOwner;
    }
    
    function withdrawFunds() external onlyOwner {
        payable(owner).transfer(address(this).balance);
    }
    
    // Emergency functions
    
    function emergencyPause() external onlyOwner {
        // Implementation for emergency pause functionality
    }
    
    function emergencyResume() external onlyOwner {
        // Implementation for emergency resume functionality
    }
}

# Credit Scoring Sample Dataset

## 📊 Dataset Overview

**File**: `credit_data_sample.csv`  
**Records**: 100 customers  
**Features**: 27 columns  
**Target**: `default` (0 = No default, 1 = Default)

---

## 📋 Data Dictionary

### Personal Information
| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `customer_id` | String | Unique customer identifier | C001-C100 |
| `age` | Integer | Customer age in years | 22-60 |
| `gender` | Categorical | Customer gender | Male, Female |
| `marital_status` | Categorical | Marital status | Single, Married, Divorced, Widowed |
| `education` | Categorical | Education level | High School, Undergraduate, Graduate, Postgraduate |
| `dependents` | Integer | Number of dependents | 0-3 |
| `region` | Categorical | Geographic region | Urban, Suburban, Rural |

### Employment Information
| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `employment_status` | Categorical | Current employment status | Employed, Unemployed, Retired, Self-employed |
| `employment_type` | Categorical | Type of employment | Full-time, Part-time, Self-employed, Unemployed, Retired |
| `years_employed` | Integer | Years in current employment | 0-29 |
| `annual_income` | Float | Annual income (USD) | 0-148,000 |

### Financial Information
| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `monthly_debt` | Float | Monthly debt payments (USD) | 550-5,300 |
| `loan_amount` | Float | Requested loan amount (USD) | 23,000-242,000 |
| `loan_purpose` | Categorical | Purpose of loan | Home, Car, Business, Personal, Education, Debt Consolidation |
| `savings_balance` | Float | Savings account balance (USD) | **Has missing values** |
| `checking_balance` | Float | Checking account balance (USD) | **Has missing values** |
| `debt_to_income_ratio` | Float | Monthly debt / Monthly income | **Has missing values** |

### Credit History
| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `credit_history_length` | Integer | Credit history in years | 1-31 |
| `num_credit_cards` | Integer | Number of credit cards | 1-8 |
| `credit_utilization` | Float | Credit utilization ratio | 0.25-0.95 |
| `num_late_payments` | Integer | Number of late payments (12 months) | 0-6 |
| `num_accounts` | Integer | Total number of credit accounts | 2-10 |
| `previous_defaults` | Integer | Previous defaults | 0-1 |
| `credit_score_external` | Integer | External credit score | 495-805 |

### Other Information
| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `home_ownership` | Categorical | Home ownership status | Own, Rent, Mortgage |
| `bank_account_years` | Integer | Years with bank account | 0-29 |

### Target Variable
| Column | Type | Description | Values |
|--------|------|-------------|--------|
| `default` | Binary | Loan default indicator | 0 = No default, 1 = Default |

---

## 🔍 Data Characteristics

### ✅ Real-world Issues Included

#### 1. **Missing Values** (Intentional)
```
savings_balance      : 15 missing values (15%)
checking_balance     : 15 missing values (15%)
debt_to_income_ratio : 15 missing values (15%)
```
**Missing pattern**: Primarily for unemployed customers (realistic scenario)

#### 2. **Imbalanced Classes**
```
Class 0 (No default): ~80 records (80%)
Class 1 (Default):    ~20 records (20%)
```
**Realistic imbalance** requiring techniques like SMOTE, class weights, etc.

#### 3. **Outliers**
- Very high incomes (>$140K)
- Very low credit scores (<500)
- Extreme credit utilization (>0.90)

#### 4. **Correlations**
- Strong positive: `annual_income` ↔ `loan_amount`
- Strong negative: `num_late_payments` ↔ `credit_score_external`
- Moderate: `credit_utilization` ↔ `default`

#### 5. **Categorical Variables**
- Need encoding: `gender`, `marital_status`, `education`, `employment_status`, etc.
- Multiple cardinality levels (binary to 6+ categories)

#### 6. **Feature Engineering Opportunities**
- Create age groups (binning)
- Income brackets
- Debt-to-income ratio (if not missing)
- Credit history score combinations

---

## 📈 Target Distribution

```
No Default (0): 80 customers (80%)
   ├─ Low risk profile
   ├─ Stable employment
   ├─ Good credit history
   └─ Low credit utilization
   
Default (1): 20 customers (20%)
   ├─ High credit utilization (>0.75)
   ├─ Multiple late payments
   ├─ Unemployed or unstable income
   └─ High debt-to-income ratio
```

---

## 🎯 Use Cases

### 1. **Data Exploration**
- Summary statistics
- Distribution analysis
- Missing value patterns
- Correlation analysis

### 2. **Data Preprocessing**
- Handle missing values (imputation/deletion)
- Outlier detection and treatment
- Feature scaling (StandardScaler, MinMaxScaler)
- Encoding categorical variables

### 3. **Feature Engineering**
- Binning continuous variables
- Creating interaction features
- Feature selection (importance, correlation)

### 4. **Model Training**
- Binary classification
- Handle imbalanced data
- Cross-validation
- Hyperparameter tuning

### 5. **Model Evaluation**
- ROC-AUC curve
- Precision-Recall
- Confusion Matrix
- F1-Score

### 6. **Model Interpretation**
- SHAP values
- Feature importance
- Partial dependence plots

---

## 💡 Expected Results

### Good Models Should Achieve:
- **AUC-ROC**: 0.75 - 0.85
- **Accuracy**: 75% - 82%
- **Precision**: 0.65 - 0.75 (for default class)
- **Recall**: 0.70 - 0.80 (for default class)
- **F1-Score**: 0.67 - 0.77

### Key Predictors (Expected):
1. `credit_utilization` - High utilization = Higher risk
2. `num_late_payments` - More late payments = Higher risk
3. `credit_score_external` - Lower score = Higher risk
4. `employment_status` - Unemployed = Higher risk
5. `debt_to_income_ratio` - Higher ratio = Higher risk

---

## 🔧 Usage in Application

### Step 1: Upload
```
Navigate to: ↑ Data Upload & Analysis
Upload: credit_data_sample.csv
```

### Step 2: EDA
- View data summary
- Check missing values (15% in 3 columns)
- Explore distributions
- Analyze correlations

### Step 3: Feature Engineering
- Handle missing values (mean/median imputation)
- Encode categoricals (one-hot/label encoding)
- Scale numerical features
- Select top 15-20 features

### Step 4: Train Model
- Try: XGBoost, LightGBM, Random Forest
- Handle imbalanced data (SMOTE/class weights)
- Cross-validate with 5 folds
- Compare models

### Step 5: Interpret
- Generate SHAP values
- Analyze global importance
- Explain individual predictions

### Step 6: Predict
- Use form to input new customer
- Get credit score (300-850)
- Receive recommendations

---

## 📝 Notes

- **Realistic Data**: Mimics real credit scoring datasets
- **Privacy**: All data is synthetic/randomly generated
- **Purpose**: Educational and testing only
- **Scale**: Small enough for quick testing, large enough to show patterns
- **Balanced Complexity**: Mix of easy and challenging features

---

## 🚀 Quick Test Scenarios

### Scenario 1: High Risk Customer
```
age = 25, unemployed, high credit_utilization (0.85)
num_late_payments = 5, low credit_score (550)
Expected: default = 1
```

### Scenario 2: Low Risk Customer
```
age = 40, employed 15 years, income = $100K
credit_utilization = 0.30, no late payments
Expected: default = 0
```

### Scenario 3: Edge Case
```
age = 35, self-employed, moderate income
Some missing values, average credit history
Expected: Borderline prediction
```

---

**Version**: 1.0  
**Created**: 2025-01-11  
**Records**: 100  
**Use**: Educational/Testing


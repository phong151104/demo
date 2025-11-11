# 🚀 Quick Start with Sample Data

## 📥 Get the Data

### Option 1: Download from App (Recommended)
1. Open Credit Scoring app
2. Go to **"↑ Data Upload & Analysis"**
3. Scroll down to **"View Sample Format"**
4. Click **"📥 Download Professional Sample Data (100 records)"**
5. Save as `credit_data_sample.csv`

### Option 2: Use Existing File
The file is already in: `sample_data/credit_data_sample.csv`

---

## ⚡ Test the Complete Workflow

### Step 1: Upload Data (1 min)
```
Page: ↑ Data Upload & Analysis
1. Click "Browse files"
2. Select: credit_data_sample.csv
3. Wait for upload
✓ Should show: 100 rows, 27 columns
```

**What to check:**
- ✓ Data loaded successfully
- ✓ Summary statistics displayed
- ✓ Missing values detected (3 columns: ~15%)

### Step 2: Explore Data (2 mins)
```
Tab: 📋 Data Sample
- View first 10 rows
- Check column types

Tab: 📊 Descriptive Statistics
- Mean age: ~38 years
- Average income: ~$70K
- Missing values in: savings_balance, checking_balance, debt_to_income_ratio

Tab: 📈 Data Distribution
- Histogram: Check income distribution
- Box Plot: Identify outliers in age/income
- Correlation: See relationship between variables
```

**Expected insights:**
- Age range: 22-60 years
- Income highly correlated with loan_amount
- Credit utilization vs default rate

### Step 3: Feature Engineering (3 mins)
```
Page: ⚡ Feature Engineering

Tab: 🔧 Preprocessing
1. Handle missing values → Select "Mean/Median Imputation"
2. Encode categoricals → Select "One-Hot Encoding"
3. Scale features → Select "Standard Scaler"

Tab: ✅ Select Features
1. Select target: "default"
2. Choose 15-20 features (exclude customer_id)
   Recommended:
   ✓ age
   ✓ annual_income
   ✓ credit_utilization
   ✓ num_late_payments
   ✓ credit_score_external
   ✓ loan_amount
   ✓ years_employed
   ✓ num_credit_cards
   ✓ education
   ✓ employment_status
```

### Step 4: Train Model (2 mins)
```
Page: ◈ Model Training

Tab: ⚙️ Configuration
1. Select model: "XGBoost" or "LightGBM"
2. Test size: 20%
3. Click "🚀 Train Model"

Tab: 📊 Results
- Check AUC-ROC (expect: 0.75-0.85)
- View Confusion Matrix
- Check metrics:
  • Accuracy: ~75-80%
  • Precision: ~65-75%
  • Recall: ~70-80%
```

**Expected performance:**
- ✓ AUC > 0.75 (Good)
- ✓ Precision > 0.65 (Acceptable for default prediction)
- ✓ Model can distinguish defaulters reasonably well

### Step 5: Interpret with SHAP (2 mins)
```
Page: ◐ Model Explanation

1. Click "🔄 Initialize SHAP Explainer"
2. Wait for computation

Tab: 🌍 Global Explanation
- Top important features:
  1. credit_utilization (high = bad)
  2. num_late_payments (high = bad)
  3. credit_score_external (low = bad)
  4. annual_income (low = bad)
  5. employment_status (unemployed = bad)

Tab: 🎯 Local Explanation
- Select a high-risk customer (e.g., index 7, 26, 34)
- View waterfall plot
- Understand why model predicts default
```

### Step 6: Make Predictions (3 mins)
```
Page: ◎ Prediction & Advisory

Tab: 📝 Input Information
Fill form with test cases:

Test Case 1 - High Risk:
  age: 25
  annual_income: 30000
  credit_utilization: 0.85
  num_late_payments: 5
  employment_status: Unemployed
  → Expected: High risk, score ~550

Test Case 2 - Low Risk:
  age: 40
  annual_income: 100000
  credit_utilization: 0.30
  num_late_payments: 0
  employment_status: Employed
  → Expected: Low risk, score ~750

Tab: 🎯 Results
- View credit score (300-850 scale)
- Check risk level
- See contributing factors

Tab: 💡 Recommendations
- Get AI suggestions for improvement
```

---

## 🎯 Quick Verification Checklist

### Data Quality ✓
- [x] 100 records loaded
- [x] 27 features available
- [x] 15% missing values detected
- [x] Target variable: 80% no default, 20% default (imbalanced)

### Processing ✓
- [x] Missing values can be imputed
- [x] Categorical variables identified
- [x] Numerical variables can be scaled
- [x] Features selected successfully

### Modeling ✓
- [x] Model trains without errors
- [x] AUC-ROC > 0.70
- [x] Confusion matrix shows predictions
- [x] Metrics calculated correctly

### Interpretation ✓
- [x] SHAP values computed
- [x] Feature importance visible
- [x] Local explanations work
- [x] Predictions match expectations

---

## 🐛 Common Issues & Solutions

### Issue 1: File Not Found
**Problem**: Can't find sample file  
**Solution**: 
```bash
# Check if file exists
ls sample_data/credit_data_sample.csv

# Or download from app
```

### Issue 2: Import Error
**Problem**: `ModuleNotFoundError`  
**Solution**:
```bash
pip install -r requirements.txt
```

### Issue 3: Memory Error
**Problem**: Large dataset causes memory issues  
**Solution**: 
- Use smaller subset (first 50 rows)
- Close other applications

### Issue 4: Missing Values Warning
**Problem**: Algorithm can't handle missing values  
**Solution**:
- ✓ This is EXPECTED - part of the demo
- Use Feature Engineering page to impute

---

## 📊 Expected Visualizations

You should see these charts:

### Upload & EDA Page:
1. ✓ Histogram of age (bell-shaped, 22-60)
2. ✓ Box plot showing income outliers
3. ✓ Correlation heatmap (income ↔ loan_amount: high)

### Feature Engineering:
1. ✓ Feature importance bar chart
2. ✓ Binning visualization

### Model Training:
1. ✓ ROC curve (should be above diagonal)
2. ✓ Confusion matrix (2x2 grid)

### SHAP:
1. ✓ Global importance (bar chart)
2. ✓ Waterfall plot (individual prediction)

### Prediction:
1. ✓ Credit score gauge (300-850)
2. ✓ Feature impact chart

---

## 🎓 Learning Objectives

After completing this workflow, you should understand:

1. ✓ How to upload and explore credit data
2. ✓ How missing values affect analysis
3. ✓ How to preprocess features
4. ✓ How to train ML models for credit scoring
5. ✓ How to interpret model decisions with SHAP
6. ✓ How to make predictions and provide recommendations

---

## ⏱️ Total Time: ~15 minutes

- Upload & EDA: 3 mins
- Feature Engineering: 3 mins  
- Model Training: 2 mins
- SHAP Explanation: 2 mins
- Prediction: 3 mins
- Exploration: 2 mins

---

## 🎉 Success Criteria

You've successfully completed the demo if:

✓ All pages load without errors  
✓ Data visualizations appear correctly  
✓ Model achieves AUC > 0.70  
✓ SHAP explanations are generated  
✓ Predictions return credit scores  

**Congratulations!** You now have a working credit scoring system! 🚀

---

**Need Help?**
- Check `README.md` for detailed data documentation
- Review error messages in Streamlit
- Ensure all dependencies are installed

**Ready for Production?**
- Follow `DEVELOPER_GUIDE.md` to implement backend
- Add real ML algorithms
- Integrate LLM for analysis
- Deploy to cloud


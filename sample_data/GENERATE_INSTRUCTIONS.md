# 🔧 Generate Sample Data

## ⚡ Quick Start

### Step 1: Generate CSV File

Open terminal/command prompt và chạy:

```bash
# Navigate to project directory
cd d:\demo

# Run generator script
python sample_data/generate_data.py
```

**Output:**
```
✓ Generated 100 records with 27 features
✓ Missing values: 45
✓ Default rate: 20.0%
✓ File saved: sample_data/credit_data_sample.csv
```

### Step 2: Verify File Created

Check file exists:
```bash
# Windows
dir sample_data\credit_data_sample.csv

# macOS/Linux
ls sample_data/credit_data_sample.csv
```

### Step 3: Use in App

1. Refresh Streamlit app (F5)
2. Go to "↑ Data Upload & Analysis"
3. You'll see "📥 Download Professional Sample Data"
4. Download and upload to test

---

## 🐛 If Script Fails

### Error: "No module named 'pandas'"
```bash
pip install pandas numpy
```

### Error: "Permission denied"
```bash
# Windows: Run as Administrator
# macOS/Linux:
chmod +x sample_data/generate_data.py
python3 sample_data/generate_data.py
```

### Error: "Directory not found"
```bash
# Make sure you're in project root
cd d:\demo
# Then run script
python sample_data/generate_data.py
```

---

## 📊 What Gets Generated

### Dataset Specifications:
- **Records**: 100 customers
- **Features**: 27 columns
- **Missing Values**: ~15% in 3 columns
  - savings_balance
  - checking_balance
  - debt_to_income_ratio
- **Target Distribution**: 
  - No default (0): ~80%
  - Default (1): ~20%

### Features Include:
✓ Personal info (age, gender, education, etc.)  
✓ Employment data (status, income, years)  
✓ Financial data (loan, debt, savings)  
✓ Credit history (cards, utilization, late payments)  
✓ Banking info (account years, ownership)  
✓ Target variable (default: 0/1)

---

## 🎯 Why Generate Instead of Static File?

1. **Random Seed**: Reproducible but realistic data
2. **No Parsing Errors**: Pandas handles CSV formatting correctly
3. **Customizable**: Easy to modify parameters in script
4. **Learning**: Shows how to create synthetic data

---

## 🔄 Alternative: Use Basic Sample

If you don't want to run the script:

**The app already has a basic fallback sample** with 10 records that will work immediately. It's simpler but good enough for testing the UI.

To use it:
1. Just open the app
2. Go to Upload & EDA
3. Download "Basic Sample"
4. Upload it

---

## 🚀 Customizing the Data

Edit `sample_data/generate_data.py` to change:

```python
# Line 14: Number of records
n_records = 100  # Change to 200, 500, etc.

# Line 18-26: Adjust age, gender, education distributions
age = np.random.randint(22, 61)  # Change range

# Line 142-150: Adjust default probability logic
if credit_utilization > 0.75:
    default_prob += 0.3  # Change weight
```

Then re-run the script to generate new data.

---

## ✅ Verification Checklist

After running script, verify:

- [ ] File exists: `sample_data/credit_data_sample.csv`
- [ ] File size: ~50-60 KB
- [ ] Open in Excel/Notepad: Should see 101 lines (1 header + 100 data)
- [ ] No parsing errors when uploading to app

---

**Ready to generate? Run:**
```bash
python sample_data/generate_data.py
```


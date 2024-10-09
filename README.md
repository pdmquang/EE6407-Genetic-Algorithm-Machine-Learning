# EE6407-Genetic-Algorithm-Machine-Learning
Master of Science in ​Computer Control & Automation (NTU PostGraduate Program 2024)

## Data Analysis
### Step 1: Process and Understand data
- Load data
- Check missing values
- Check Outliners
    - Boxplot
    - Mean, Median, Mode
- Process Outliners
    - **Removal**: Outliers can be removed from the dataset if they are not numerous.
    - **Imputation**: Outliers can be imputed with the mean, median, mode
    - **Capping**: Outliers can be capped by replacing them with the 5th or 95th percentile values. 

### Step 2: Estimate the parameters (Train Model)
- Bayes Decision rule
    - Dependent, correlated features 
        - Suitable for continuous data normal distributed
    - Parameters: 
        - **Mu**: Mean by features
        - **Sigma**: Standard Deviation by features
- Naive Bayes
    - Assume features are independent and uncorrelated. 
    - Parameters: 
        - **Mu**: Mean
        - **Sigma**: Standard Deviation

### Step 3: Tune Parameter (K-fold)
- K-fold
### Step 4: Compare model
- Accuracy
- Precision (False Positive)
- Recall (False Negative)
- F1-score (Average of Precision & Recall)
    - Useful when Precision and Recall are 2 extreme values
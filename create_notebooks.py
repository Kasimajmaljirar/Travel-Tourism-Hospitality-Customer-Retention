import nbformat as nbf
import os

os.makedirs('notebooks', exist_ok=True)

def create_nb1():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Week 1: Data Acquisition, Cleaning, and Feature Engineering\n\nIn this notebook, we will:\n1. Load the raw Hotel Booking Demand dataset.\n2. Handle missing values in columns like `agent` and `company`.\n3. Treat extreme outliers in `adr` (Average Daily Rate).\n4. Engineer new features, such as `total_duration` (weekend + weekday nights)."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport os\n\n# Ensure plot outputs appear in the notebook\n%matplotlib inline"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("file_path = '../data/hotel_bookings.csv'\nif os.path.exists(file_path):\n    df = pd.read_csv(file_path)\n    print('Dataset shape:', df.shape)\n    display(df.head())\nelse:\n    print('Dataset not found. Please run setup_data.py first.')"),
        nbf.v4.new_markdown_cell("## 2. Handle Missing Values"),
        nbf.v4.new_code_cell("# Check missing values\nmissing_vals = df.isnull().sum()\nprint(missing_vals[missing_vals > 0])\n\n# 'company' has very high missing values, we might drop it or fill with 0 (meaning no company)\ndf['company'] = df['company'].fillna(0)\n\n# 'agent' can also be filled with 0\ndf['agent'] = df['agent'].fillna(0)\n\n# 'country' filled with 'Unknown'\ndf['country'] = df['country'].fillna('Unknown')\n\n# 'children' missing values are small, fill with 0\ndf['children'] = df['children'].fillna(0)\n\nprint('\\nAfter filling:\\n', df[['company', 'agent', 'country', 'children']].isnull().sum())"),
        nbf.v4.new_markdown_cell("## 3. Treat Outliers in Average Daily Rate (ADR)"),
        nbf.v4.new_code_cell("plt.figure(figsize=(10,4))\nsns.boxplot(x=df['adr'])\nplt.title('ADR Distribution (Before Cleaning)')\nplt.show()\n\n# Remove negative ADR and extreme outliers (e.g. > 5000)\ndf = df[(df['adr'] >= 0) & (df['adr'] < 5000)]\n\nplt.figure(figsize=(10,4))\nsns.boxplot(x=df['adr'])\nplt.title('ADR Distribution (After Cleaning)')\nplt.show()"),
        nbf.v4.new_markdown_cell("## 4. Feature Engineering"),
        nbf.v4.new_code_cell("# Calculate total duration of stay\ndf['total_duration'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']\n\n# Remove records with zero duration\ndf = df[df['total_duration'] > 0]\n\nprint('Final dataset shape after cleaning:', df.shape)\ndf.head()"),
        nbf.v4.new_markdown_cell("## 5. Save Cleaned Data"),
        nbf.v4.new_code_cell("df.to_csv('../data/cleaned_hotel_bookings.csv', index=False)\nprint('Cleaned data saved to data/cleaned_hotel_bookings.csv')")
    ]
    with open('notebooks/01_Data_Cleaning_and_Feature_Engineering.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


def create_nb2():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Week 2: Exploratory Data Analysis (EDA) and Statistical Testing\n\nIn this notebook, we will:\n1. Perform Univariate and Bivariate analysis.\n2. Analyze the relationship between Average Daily Rate (ADR) and cancellation rates.\n3. Map correlation matrices to identify drivers of cancellations.\n4. Visualize the booking curve (lead time vs. booking volume)."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n%matplotlib inline\n\n# Set seaborn aesthetic style\nsns.set_style('whitegrid')"),
        nbf.v4.new_markdown_cell("## 1. Load Cleaned Data"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/cleaned_hotel_bookings.csv')\ndf.head()"),
        nbf.v4.new_markdown_cell("## 2. Cancellation Overview"),
        nbf.v4.new_code_cell("plt.figure(figsize=(6,4))\nsns.countplot(x='is_canceled', data=df, palette='Set2')\nplt.title('Booking Status (0 = Not Canceled, 1 = Canceled)')\nplt.show()\n\ncancel_rate = df['is_canceled'].mean() * 100\nprint(f'Overall Cancellation Rate: {cancel_rate:.2f}%')"),
        nbf.v4.new_markdown_cell("## 3. ADR vs Cancellation Rate (Bivariate Analysis)\nWe explore whether higher daily rates correlate with higher cancellation rates."),
        nbf.v4.new_code_cell("plt.figure(figsize=(8,5))\nsns.boxplot(x='is_canceled', y='adr', data=df[df['adr'] < 500], palette='Set1')\nplt.title('ADR Distribution by Cancellation Status')\nplt.show()"),
        nbf.v4.new_markdown_cell("## 4. Booking Curve (Lead Time vs Booking Volume)"),
        nbf.v4.new_code_cell("plt.figure(figsize=(10,6))\nsns.histplot(df['lead_time'], bins=50, kde=True, color='skyblue')\nplt.title('Distribution of Lead Time (Booking Curve)')\nplt.xlabel('Lead Time (Days)')\nplt.ylabel('Number of Bookings')\nplt.show()"),
        nbf.v4.new_markdown_cell("## 5. Correlation Matrix\nLet's isolate numeric features to find strong indicators of cancellation."),
        nbf.v4.new_code_cell("numeric_cols = df.select_dtypes(include=[np.number])\ncorr = numeric_cols.corr()\n\nplt.figure(figsize=(12, 10))\nsns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)\nplt.title('Correlation Matrix of Numeric Features')\nplt.show()\n\nprint('Top numerical correlations with cancellation:')\nprint(corr['is_canceled'].sort_values(ascending=False).head(10))")
    ]
    with open('notebooks/02_Exploratory_Data_Analysis.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


def create_nb3():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Week 3: Baseline Predictive Modeling (Churn Prediction)\n\nIn this notebook, we will:\n1. Build a Customer Segmentation Model (Basic grouping by booking behavior).\n2. Prepare data for machine learning (Encoding categorical variables).\n3. Train a baseline Logistic Regression model to predict `is_canceled`.\n4. Evaluate the model using Accuracy, Precision, Recall, and ROC-AUC."),
        nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, classification_report\n\n%matplotlib inline"),
        nbf.v4.new_markdown_cell("## 1. Load Data"),
        nbf.v4.new_code_cell("df = pd.read_csv('../data/cleaned_hotel_bookings.csv')"),
        nbf.v4.new_markdown_cell("## 2. Customer Segmentation (Simple)\nGrouping users based on Corporate vs Leisure, and early planners vs last-minute."),
        nbf.v4.new_code_cell("df['planner_type'] = np.where(df['lead_time'] > 30, 'Early Planner', 'Last Minute')\n\nplt.figure(figsize=(8,5))\nsns.countplot(x='market_segment', hue='planner_type', data=df)\nplt.title('Customer Segmentation: Market Segment vs Planner Type')\nplt.xticks(rotation=45)\nplt.show()"),
        nbf.v4.new_markdown_cell("## 3. Data Preparation for Machine Learning"),
        nbf.v4.new_code_cell("# Select subset of features for baseline model to keep it simple and interpretable\nfeatures = ['lead_time', 'total_duration', 'previous_cancellations', \n            'booking_changes', 'adr', 'total_of_special_requests', 'market_segment', 'deposit_type']\n\nX = df[features].copy()\ny = df['is_canceled']\n\n# One-hot encode categorical features\nX = pd.get_dummies(X, columns=['market_segment', 'deposit_type'], drop_first=True)\n\n# Train-test split\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\n\n# Scale numerical features\nscaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)"),
        nbf.v4.new_markdown_cell("## 4. Train Logistic Regression Model"),
        nbf.v4.new_code_cell("model = LogisticRegression(max_iter=1000)\nmodel.fit(X_train_scaled, y_train)"),
        nbf.v4.new_markdown_cell("## 5. Model Evaluation"),
        nbf.v4.new_code_cell("y_pred = model.predict(X_test_scaled)\ny_proba = model.predict_proba(X_test_scaled)[:, 1]\n\nprint('Classification Report:')\nprint(classification_report(y_test, y_pred))\n\nprint(f'Accuracy:  {accuracy_score(y_test, y_pred):.4f}')\nprint(f'Precision: {precision_score(y_test, y_pred):.4f}')\nprint(f'Recall:    {recall_score(y_test, y_pred):.4f}')\nprint(f'ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}')"),
        nbf.v4.new_markdown_cell("## 6. ROC Curve"),
        nbf.v4.new_code_cell("fpr, tpr, thresholds = roc_curve(y_test, y_proba)\n\nplt.figure(figsize=(8,6))\nplt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_proba):.2f})')\nplt.plot([0, 1], [0, 1], 'k--')\nplt.xlabel('False Positive Rate')\nplt.ylabel('True Positive Rate')\nplt.title('ROC Curve')\nplt.legend(loc='lower right')\nplt.show()")
    ]
    with open('notebooks/03_Customer_Segmentation_and_Predictive_Modeling.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    create_nb1()
    create_nb2()
    create_nb3()
    print("All notebooks created successfully!")

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, flash

app = Flask(__name__)
app.secret_key = "customer_segment_ai_secret_key"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'customers_1500.xlsx')

try:
    kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    pca = joblib.load(os.path.join(MODEL_DIR, 'pca.pkl'))
except Exception as e:
    print(f"Error loading model files: {e}")
    kmeans, scaler, pca = None, None, None

FEATURES = [
    'Age', 
    'Annual Income (k$)', 
    'Spending Score (1-100)', 
    'Tenure (months)', 
    'Total Orders', 
    'Total Spend ($)', 
    'Avg Order Value ($)', 
    'Days Since Last Purchase'
]

SEGMENT_INFO = {
    0: {"name": "Regular / Moderate Customers", "desc": "Moderate income and spending patterns.", "strategy": "Standard loyalty programs and seasonal offers."},
    1: {"name": "High Value / VIP Customers", "desc": "High annual income combined with high spending score.", "strategy": "VIP access, early drops, dedicated assistance."},
    2: {"name": "Budget / Low Engagement Customers", "desc": "Lower spenders with infrequent purchases.", "strategy": "Re-engage with discount coupons and win-back emails."},
    3: {"name": "At-Risk / Churn Susceptible Customers", "desc": "High past spending but long gap since last purchase.", "strategy": "Urgent retention offers and feedback surveys."}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            age = float(request.form.get('age', 0))
            income = float(request.form.get('income', 0))
            spending_score = float(request.form.get('spending_score', 0))
            tenure = float(request.form.get('tenure', 0))
            orders = float(request.form.get('orders', 0))
            total_spend = float(request.form.get('total_spend', 0))
            avg_order_val = float(request.form.get('avg_order_val', 0))
            recency = float(request.form.get('recency', 0))

            input_data = pd.DataFrame([[
                age, income, spending_score, tenure, 
                orders, total_spend, avg_order_val, recency
            ]], columns=FEATURES)

            if scaler and kmeans:
                scaled_data = scaler.transform(input_data)
                cluster_id = int(kmeans.predict(scaled_data)[0])
                result = SEGMENT_INFO.get(cluster_id, {
                    "name": f"Cluster {cluster_id}",
                    "desc": "Custom segmented behavior group.",
                    "strategy": "Apply tailored promotional strategies."
                })
            else:
                flash("ML Model is not loaded properly.", "danger")
        except Exception as e:
            flash(f"Error processing input data: {str(e)}", "danger")

    return render_template('predict.html', result=result)

@app.route('/dashboard')
def dashboard():
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_excel(DATA_PATH, sheet_name="Customers", engine="openpyxl")

            # Flexibly match columns from dataset to target standard names
            col_map = {}
            for col in df.columns:
                c_lower = str(col).strip().lower()
                if 'age' in c_lower:
                    col_map[col] = 'Age'
                elif 'income' in c_lower:
                    col_map[col] = 'Annual Income (k$)'
                elif 'spending' in c_lower or 'score' in c_lower:
                    col_map[col] = 'Spending Score (1-100)'
                elif 'tenure' in c_lower:
                    col_map[col] = 'Tenure (months)'
                elif 'order' in c_lower and 'total' in c_lower:
                    col_map[col] = 'Total Orders'
                elif 'spend' in c_lower and 'total' in c_lower:
                    col_map[col] = 'Total Spend ($)'
                elif 'avg' in c_lower or 'average' in c_lower:
                    col_map[col] = 'Avg Order Value ($)'
                elif 'days' in c_lower or 'recency' in c_lower or 'last' in c_lower:
                    col_map[col] = 'Days Since Last Purchase'

            df = df.rename(columns=col_map)

            # Assign clusters using model
            if scaler and kmeans and all(f in df.columns for f in FEATURES):
                X_scaled = scaler.transform(df[FEATURES])
                df['Cluster'] = kmeans.predict(X_scaled)
                df['Segment_Name'] = df['Cluster'].map(
                    lambda x: SEGMENT_INFO.get(x, {}).get('name', f"Cluster {x}")
                )
            else:
                df['Segment_Name'] = "Unassigned"

            total_customers = len(df)
            avg_age = round(df['Age'].mean(), 1) if 'Age' in df.columns else 0
            avg_income = round(df['Annual Income (k$)'].mean(), 1) if 'Annual Income (k$)' in df.columns else 0
            avg_spend = round(df['Total Spend ($)'].mean(), 2) if 'Total Spend ($)' in df.columns else 0
            
            cluster_counts = df['Segment_Name'].value_counts().to_dict()

            # Clean records for HTML output
            all_customers = df.to_dict(orient='records')

            return render_template('dashboard.html', 
                                   total_customers=total_customers,
                                   avg_age=avg_age,
                                   avg_income=avg_income,
                                   avg_spend=avg_spend,
                                   cluster_counts=cluster_counts,
                                   all_customers=all_customers)
        except Exception as e:
            print("Dashboard exception:", str(e))
            flash(f"Failed to read dataset: {str(e)}", "warning")
            return render_template('dashboard.html', total_customers=0, cluster_counts={}, all_customers=[])
    else:
        flash("Dataset not found for dashboard statistics.", "warning")
        return render_template('dashboard.html', total_customers=0, cluster_counts={}, all_customers=[])

if __name__ == '__main__':
    app.run(debug=True, port=5000)
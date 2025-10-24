import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import plotly.offline as pyo
import re
import tensorflow as tf
from textblob import TextBlob
from flask import Flask, request, render_template, redirect, url_for, jsonify, Response
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==============================================
# LOAD MODELS & ARTIFACTS
# ==============================================
print("Loading models and artifacts...")
# Only load Wide & Deep model for better performance
wd_model = tf.keras.models.load_model("models/CHAMPION_2_Wide_Deep_tf")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
reference_data = pd.read_csv("models/preprocessed.csv")
print("✓ Wide & Deep model loaded successfully!")
print(f"✓ Expected features: {len(feature_columns)}")

# Define expected column names (canonical form)
CANONICAL_COLUMNS = {
    'channel_name': 'channel_name',
    'category': 'category',
    'sub_category': 'Sub-category',  # Note: keeps original format
    'customer_remarks': 'Customer Remarks',
    'issue_reported_at': 'Issue_reported at',
    'issue_responded': 'issue_responded',
    'survey_response_date': 'Survey_response_Date',
    'agent_name': 'Agent_name',
    'supervisor': 'Supervisor',
    'manager': 'Manager',
    'tenure_bucket': 'Tenure Bucket',
    'agent_shift': 'Agent Shift',
    'item_price': 'Item_price',
    'connected_handling_time': 'Connected_handling_time'
}

# Required columns (in canonical form for display)
REQUIRED_INPUT_COLUMNS = [
    'channel_name',
    'category',
    'Sub-category',
    'Customer Remarks',
    'Issue_reported at',
    'issue_responded',
    'Survey_response_Date',
    'Agent_name',
    'Supervisor',
    'Manager',
    'Tenure Bucket',
    'Agent Shift'
]

OPTIONAL_INPUT_COLUMNS = [
    'Item_price',
    'Connected_handling_time'
]

# ==============================================
# COLUMN NORMALIZATION FUNCTION
# ==============================================
def normalize_column_name(col):
    """
    Normalize column names to handle different formats:
    - Remove leading/trailing spaces
    - Convert to lowercase
    - Replace spaces with underscores
    - Remove special characters except underscores
    """
    if not isinstance(col, str):
        return str(col)
    
    # Remove leading/trailing spaces
    col = col.strip()
    
    # Convert to lowercase
    col = col.lower()
    
    # Replace spaces with underscores
    col = col.replace(' ', '_')
    
    # Replace hyphens with underscores
    col = col.replace('-', '_')
    
    # Remove special characters except underscores
    col = re.sub(r'[^a-z0-9_]', '', col)
    
    return col

def map_to_canonical_columns(df):
    """
    Map input dataframe columns to canonical column names.
    Handles different formats: spaces, underscores, cases, etc.
    """
    column_mapping = {}
    normalized_canonical = {normalize_column_name(k): v for k, v in CANONICAL_COLUMNS.items()}
    
    for col in df.columns:
        normalized = normalize_column_name(col)
        
        # Direct match
        if normalized in normalized_canonical:
            column_mapping[col] = normalized_canonical[normalized]
        # Try to find best match
        else:
            # Check for partial matches
            for norm_key, canonical_value in normalized_canonical.items():
                if normalized in norm_key or norm_key in normalized:
                    column_mapping[col] = canonical_value
                    break
    
    # Rename columns
    df_renamed = df.rename(columns=column_mapping)
    
    return df_renamed, column_mapping

# ==============================================
# INPUT VALIDATION FUNCTION
# ==============================================
def validate_input_data(df):
    """
    Validate that input DataFrame has all required columns after normalization
    Returns: (is_valid, error_message, missing_columns)
    """
    # First, normalize and map columns
    df_mapped, mapping = map_to_canonical_columns(df)
    
    # Check for required columns
    missing_cols = [col for col in REQUIRED_INPUT_COLUMNS if col not in df_mapped.columns]
    
    if missing_cols:
        # Try to provide helpful feedback about what was found
        found_cols = list(df_mapped.columns)
        error_msg = f"Missing required columns: {', '.join(missing_cols)}\n"
        error_msg += f"Found columns: {', '.join(found_cols)}"
        return False, error_msg, missing_cols, df_mapped
    
    # Check for completely empty required columns
    empty_cols = []
    for col in REQUIRED_INPUT_COLUMNS:
        if col in df_mapped.columns and df_mapped[col].isna().all():
            empty_cols.append(col)
    
    if empty_cols:
        warning_msg = f"Warning: These columns are completely empty: {', '.join(empty_cols)}"
        print(warning_msg)
    
    return True, None, [], df_mapped

# ==============================================
# PREPROCESSING FUNCTION (MATCHES TRAINING)
# ==============================================
def preprocess_input(df_raw, validate=True):
    """
    Preprocess input data to match exact training pipeline
    Automatically handles different column name formats
    Args:
        df_raw: Raw input DataFrame
        validate: Whether to validate input columns (default True)
    """
    # Normalize column names first
    if validate:
        is_valid, error_msg, missing_cols, df = validate_input_data(df_raw)
        if not is_valid:
            raise ValueError(error_msg)
    else:
        df, _ = map_to_canonical_columns(df_raw)
    
    df = df.copy()
    
    # 1. Handle Customer Remarks
    df['Customer Remarks_was_missing'] = df['Customer Remarks'].isna().astype(int)
    df['Customer Remarks'] = df['Customer Remarks'].fillna('No Remarks')
    
    # 2. Text features
    df['Customer Remarks_word_count'] = df['Customer Remarks'].apply(lambda x: len(str(x).split()))
    df['sentiment_polarity'] = df['Customer Remarks'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
    )
    df['sentiment_subjectivity'] = df['Customer Remarks'].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity if pd.notna(x) else 0
    )
    df['anger_intensity'] = (1 - df['sentiment_polarity'].abs()) * np.log1p(df['Customer Remarks_word_count'])
    df['is_neutral_brief'] = ((df['sentiment_polarity'].abs() < 0.2) & (df['Customer Remarks_word_count'] < 20)).astype(int)
    
    # 3. Datetime processing
    for col in ['Issue_reported at', 'issue_responded', 'Survey_response_Date']:
        if col in df.columns:
            new_col = col.replace(' ', '_').replace('_at', '').lower()
            df[new_col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
    
    # 4. Response time features
    df['issue_reported_was_missing'] = 0
    df['issue_responded_was_missing'] = 0
    
    if 'issue_reported' in df.columns and 'issue_responded' in df.columns:
        valid_mask = df['issue_reported'].notna() & df['issue_responded'].notna()
        df['response_time_hours'] = np.nan
        df.loc[valid_mask, 'response_time_hours'] = (
            (df.loc[valid_mask, 'issue_responded'] - df.loc[valid_mask, 'issue_reported']).dt.total_seconds() / 3600
        )
        df.loc[df['response_time_hours'] < 0, 'response_time_hours'] = np.nan
        df['response_time_hours'] = df['response_time_hours'].clip(upper=720)
        
        median_response = reference_data['response_time_hours'].median() if 'response_time_hours' in reference_data.columns else 24
        df['response_time_hours'].fillna(median_response, inplace=True)
        
        df['response_very_fast'] = (df['response_time_hours'] <= 0.5).astype(int)
        df['response_immediate'] = (df['response_time_hours'] <= 1).astype(int)
        df['response_fast'] = ((df['response_time_hours'] > 1) & (df['response_time_hours'] <= 24)).astype(int)
        df['response_under_4h'] = (df['response_time_hours'] <= 4).astype(int)
        df['response_under_1day'] = (df['response_time_hours'] <= 24).astype(int)
        df['response_slow'] = ((df['response_time_hours'] > 24) & (df['response_time_hours'] <= 72)).astype(int)
        df['response_very_slow'] = (df['response_time_hours'] > 72).astype(int)
        df['response_time_hours_log'] = np.log1p(df['response_time_hours'])
        df['response_time_squared'] = df['response_time_hours'] ** 2
    
    # 5. Day patterns
    if 'issue_reported' in df.columns:
        df['day_of_week'] = df['issue_reported'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        day_stats = reference_data.groupby('day_of_week')['CSAT Score'].mean() if 'day_of_week' in reference_data.columns and 'CSAT Score' in reference_data.columns else {}
        df['day_avg_csat'] = df['day_of_week'].map(day_stats).fillna(3.0)
    
    # 6. Frequency encoding
    freq_cols = ['Agent_name', 'Supervisor', 'Manager', 'category', 'Sub-category']
    for col in freq_cols:
        if col in df.columns:
            freq_map = reference_data[col].value_counts(normalize=True) if col in reference_data.columns else {}
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(0)
    
    # 7. Agent quality metrics
    if 'Agent_name' in df.columns and 'Agent_name' in reference_data.columns and 'CSAT Score' in reference_data.columns:
        agent_stats = reference_data.groupby('Agent_name')['CSAT Score'].agg(['mean', 'std', 'count'])
        df['agent_avg_csat'] = df['Agent_name'].map(agent_stats['mean']).fillna(3.0)
        df['agent_consistency'] = df['Agent_name'].map(agent_stats['std']).fillna(0)
        df['is_high_performer'] = (df['agent_avg_csat'] >= 3.5).astype(int)
    
    # 8. Category/Channel metrics
    if 'category' in df.columns and 'category' in reference_data.columns and 'CSAT Score' in reference_data.columns:
        category_stats = reference_data.groupby('category')['CSAT Score'].agg(['mean', 'std'])
        df['category_avg_csat'] = df['category'].map(category_stats['mean']).fillna(3.0)
        df['category_variance'] = df['category'].map(category_stats['std']).fillna(0)
    
    if 'channel_name' in df.columns and 'channel_name' in reference_data.columns and 'CSAT Score' in reference_data.columns:
        channel_stats = reference_data.groupby('channel_name')['CSAT Score'].agg(['mean', 'std'])
        df['channel_avg_csat'] = df['channel_name'].map(channel_stats['mean']).fillna(3.0)
        df['channel_std'] = df['channel_name'].map(channel_stats['std']).fillna(0)
    
    # 9. Supervisor/Manager metrics
    if 'Supervisor' in df.columns and 'Supervisor' in reference_data.columns and 'CSAT Score' in reference_data.columns:
        supervisor_stats = reference_data.groupby('Supervisor')['CSAT Score'].mean()
        df['supervisor_avg_csat'] = df['Supervisor'].map(supervisor_stats).fillna(3.0)
    
    if 'Manager' in df.columns and 'Manager' in reference_data.columns and 'CSAT Score' in reference_data.columns:
        manager_stats = reference_data.groupby('Manager')['CSAT Score'].mean()
        df['manager_avg_csat'] = df['Manager'].map(manager_stats).fillna(3.0)
    
    # 10. Shift metrics
    if 'Agent Shift' in df.columns and 'Agent Shift' in reference_data.columns and 'CSAT Score' in reference_data.columns:
        shift_stats = reference_data.groupby('Agent Shift')['CSAT Score'].mean()
        df['shift_avg_csat'] = df['Agent Shift'].map(shift_stats).fillna(3.0)
    
    # 11. Tenure encoding
    if 'Tenure Bucket' in df.columns:
        tenure_map = {'0-2 years': 1, '2-4 years': 2, '4-6 years': 3, '6+ years': 4}
        df['tenure_numeric'] = df['Tenure Bucket'].map(tenure_map).fillna(0)
        df['Tenure Bucket_encoded'] = df['tenure_numeric']
        df['experienced_night_shift'] = ((df['tenure_numeric'] >= 3) & (df['Agent Shift'] == 'Night')).astype(int) if 'Agent Shift' in df.columns else 0
    
    # 12. One-hot encoding
    for col in ['Agent Shift', 'channel_name']:
        if col in df.columns:
            col_prefix = col.replace(' ', '_')
            dummies = pd.get_dummies(df[col], prefix=col_prefix, drop_first=True, dtype=int)
            df = pd.concat([df, dummies], axis=1)
    
    # 13. Add Item_price if missing
    if 'Item_price' not in df.columns:
        df['Item_price'] = reference_data['Item_price'].median() if 'Item_price' in reference_data.columns else 0
    
    # 14. Connected_handling_time
    if 'Connected_handling_time' not in df.columns:
        df['Connected_handling_time'] = reference_data['Connected_handling_time'].median() if 'Connected_handling_time' in reference_data.columns else 0
    
    # 15. Select and align features in EXACT order
    X = pd.DataFrame()
    for col in feature_columns:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0
    
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    # CRITICAL: Ensure columns are in exact order as feature_columns
    X = X[feature_columns]
    
    return X

# ==============================================
# PREDICTION FUNCTION (Wide & Deep Only)
# ==============================================
def verify_feature_alignment(X):
    """Verify features match expected order and names"""
    X_cols = list(X.columns)
    if X_cols != feature_columns:
        print("⚠️ WARNING: Feature mismatch detected!")
        print(f"Expected {len(feature_columns)} features, got {len(X_cols)}")
        
        # Find mismatches
        for i, (expected, actual) in enumerate(zip(feature_columns, X_cols)):
            if expected != actual:
                print(f"  Position {i}: Expected '{expected}', got '{actual}'")
                break
        
        raise ValueError(
            f"Feature columns don't match! "
            f"Expected {len(feature_columns)} features in specific order. "
            f"First mismatch at position {i}: expected '{expected}', got '{actual}'"
        )
    return True

def predict_csat(X):
    """Predict CSAT using only Wide & Deep model for faster performance"""
    verify_feature_alignment(X)
    
    # Convert to numpy array to avoid feature name mismatch
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    
    # Scale features
    X_scaled = scaler.transform(X_array)
    
    # Predict using Wide & Deep model only
    pred = wd_model.predict(X_scaled, verbose=0)
    
    # Get predicted classes and confidence
    predicted_classes = np.argmax(pred, axis=1)
    predicted_csat = predicted_classes + 1
    confidence = np.max(pred, axis=1)
    
    return predicted_csat, confidence

# ==============================================
# FLASK ROUTES
# ==============================================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/get_required_columns')
def get_required_columns():
    """API endpoint to get required column information"""
    return jsonify({
        'required': REQUIRED_INPUT_COLUMNS,
        'optional': OPTIONAL_INPUT_COLUMNS,
        'total_required': len(REQUIRED_INPUT_COLUMNS),
        'note': 'Column names are flexible - the system will automatically normalize different formats (spaces, underscores, cases)'
    })

@app.route('/single', methods=['GET','POST'])
def single_prediction():
    prediction_result = None
    error_message = None
    
    if request.method == 'POST':
        try:
            # Collect data in exact order of REQUIRED_INPUT_COLUMNS
            data = {}
            for col in REQUIRED_INPUT_COLUMNS:
                data[col] = [request.form.get(col, '')]
            
            # Add optional columns if provided
            for col in OPTIONAL_INPUT_COLUMNS:
                value = request.form.get(col, None)
                if value:
                    data[col] = [value]
            
            df_raw = pd.DataFrame(data)
            
            # Validate and preprocess
            X = preprocess_input(df_raw, validate=True)
            predicted_csat, confidence = predict_csat(X)
            
            prediction_result = {
                'csat': int(predicted_csat[0]),
                'confidence': float(confidence[0]) * 100  # Convert to percentage
            }
        except ValueError as ve:
            error_message = f"Validation Error: {str(ve)}"
        except Exception as e:
            error_message = f"Prediction Error: {str(e)}"
    
    return render_template('single.html', 
                         prediction=prediction_result, 
                         error=error_message,
                         required_columns=REQUIRED_INPUT_COLUMNS)

@app.route('/bulk', methods=['GET','POST'])
def bulk_prediction():
    results = None
    error_message = None
    validation_info = None
    
    if request.method == 'POST' and 'file' in request.files:
        try:
            file = request.files['file']
            df_raw = pd.read_csv(file)
            
            print(f"✓ Uploaded file columns: {list(df_raw.columns)}")
            
            # Validate input columns (with automatic normalization)
            is_valid, error_msg, missing_cols, df_normalized = validate_input_data(df_raw)
            
            if not is_valid:
                validation_info = {
                    'status': 'error',
                    'missing_columns': missing_cols,
                    'found_columns': list(df_raw.columns),
                    'required_columns': REQUIRED_INPUT_COLUMNS
                }
                error_message = error_msg
            else:
                # Preprocess and predict
                X = preprocess_input(df_raw, validate=True)
                predicted_csat, confidence = predict_csat(X)
                
                results = []
                for i in range(len(predicted_csat)):
                    results.append({
                        'id': i+1,
                        'csat': int(predicted_csat[i]),
                        'confidence': float(confidence[i]) * 100  # Convert to percentage
                    })
                
                validation_info = {
                    'status': 'success',
                    'total_records': len(df_raw),
                    'columns_found': len(df_raw.columns)
                }
                print(f"✓ Successfully processed {len(results)} records")
                
        except ValueError as ve:
            error_message = f"Validation Error: {str(ve)}"
        except Exception as e:
            error_message = f"Processing Error: {str(e)}"
            import traceback
            print(traceback.format_exc())
    
    return render_template('bulk.html', 
                         results=results, 
                         error=error_message,
                         validation_info=validation_info,
                         required_columns=REQUIRED_INPUT_COLUMNS)

@app.route('/download_template')
def download_template():
    """Generate and download a CSV template with required columns in correct order"""
    template_df = pd.DataFrame(columns=REQUIRED_INPUT_COLUMNS + OPTIONAL_INPUT_COLUMNS)
    
    # Add example row with proper order matching REQUIRED_INPUT_COLUMNS
    example_row = {
        'channel_name': 'Email',
        'category': 'Technical',
        'Sub-category': 'Software',
        'Customer Remarks': 'Product not working as expected, need immediate assistance',
        'Issue_reported at': '2024-01-01 10:00:00',
        'issue_responded': '2024-01-01 11:00:00',
        'Survey_response_Date': '2024-01-02 10:00:00',
        'Agent_name': 'Agent_001',
        'Supervisor': 'Supervisor_A',
        'Manager': 'Manager_X',
        'Tenure Bucket': '2-4 years',
        'Agent Shift': 'Day',
        'Item_price': '100',
        'Connected_handling_time': '15'
    }

    template_df = pd.concat([template_df, pd.DataFrame([example_row])], ignore_index=True)
    
    csv_string = template_df.to_csv(index=False)
    
    return Response(
        csv_string,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=csat_input_template.csv"}
    )

@app.route('/about')
def about():
    model_info = {
        'name': 'Wide & Deep Neural Network',
        'accuracy': '71.91%',
        'within1_accuracy': '84.83%',
        'f1_macro': '0.2504',
        'f1_weighted': '0.6327',
        'features': len(feature_columns),
        'architecture': 'Hybrid deep learning combining wide (memorization) and deep (generalization) components for accurate CSAT prediction',
        'flexibility': 'Accepts any column order and automatically normalizes column names (spaces, underscores, cases)',
        'nlp_features': 'Uses TextBlob for sentiment analysis, polarity, subjectivity, and anger intensity detection'
    }
    return render_template('about.html', model_info=model_info)

# ==============================================
# RUN APP
# ==============================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
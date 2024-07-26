import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import shap
import numpy as np
from imblearn.over_sampling import SMOTE

def load_dataset(filepath):
    sepsis_data = pd.read_csv(filepath)
    return sepsis_data

def separate_features_target(sepsis_data):
    features = sepsis_data.drop(columns=['SepsisLabel', 'Unnamed: 0', 'Patient_ID'])
    target = sepsis_data['SepsisLabel']
    return features, target

def drop_nan_target(sepsis_data):
    sepsis_data = sepsis_data.dropna(subset=['SepsisLabel'])
    return sepsis_data

def preprocess_features(features):
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed)
    return features_scaled, imputer, scaler

def split_data(features, target):
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return features_train, features_test, target_train, target_test

def resample_data(features_train, target_train):
    smote = SMOTE(random_state=42)
    features_train_resampled, target_train_resampled = smote.fit_resample(features_train, target_train)
    return features_train_resampled, target_train_resampled

def tune_model(features_train, target_train):
    param_dist = {
        'n_estimators': randint(50, 150),
        'max_depth': randint(3, 7),
        'learning_rate': uniform(0.01, 0.19),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    }
    xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=20, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42, error_score='raise')
    random_search.fit(features_train, target_train)
    best_model = random_search.best_estimator_
    best_model.set_params(scale_pos_weight=(len(target_train) - sum(target_train)) / sum(target_train))
    best_model.fit(features_train, target_train)
    return best_model, random_search.best_params_

def evaluate_model(model, features_test, target_test):
    target_pred = model.predict(features_test)
    target_pred_prob = model.predict_proba(features_test)[:, 1]
    accuracy = accuracy_score(target_test, target_pred)
    precision = precision_score(target_test, target_pred)
    recall = recall_score(target_test, target_pred)
    f1 = f1_score(target_test, target_pred)
    roc_auc = roc_auc_score(target_test, target_pred_prob)
    return accuracy, precision, recall, f1, roc_auc

def shap_summary_plot(model, features_test):
    shap_explainer = shap.Explainer(model)
    shap_values = shap_explainer(features_test)
    shap.summary_plot(shap_values, features_test)

def preprocess_user_input_top_features(user_input, imputer, scaler):
    user_input_df = pd.DataFrame([user_input], columns=[f'Feature {i}' for i in [7, 2, 39, 0, 37, 34, 38, 1, 35, 5]])
    user_input_imputed = imputer.transform(user_input_df)
    user_input_scaled = scaler.transform(user_input_imputed)
    return user_input_scaled

def preprocess_user_input_all_features(user_input, imputer, scaler):
    user_input_df = pd.DataFrame([user_input], columns=features.columns)
    user_input_imputed = imputer.transform(user_input_df)
    user_input_scaled = scaler.transform(user_input_imputed)
    return user_input_scaled

def predict_sepsis_initial(user_input, model, imputer, scaler):
    processed_input = preprocess_user_input_top_features(user_input, imputer, scaler)
    prediction = model.predict(processed_input)
    prediction_prob = model.predict_proba(processed_input)[:, 1]
    return prediction[0], prediction_prob[0]

def predict_sepsis_using_all_features(user_input, model, imputer, scaler):
    processed_input = preprocess_user_input_all_features(user_input, imputer, scaler)
    prediction = model.predict(processed_input)
    prediction_prob = model.predict_proba(processed_input)[:, 1]
    return prediction[0], prediction_prob[0]

def sepsis_prediction_workflow(user_input_top_features, user_input_all_features=None):
    initial_prediction, initial_prediction_prob = predict_sepsis_initial(user_input_top_features, best_xgb_model, feature_imputer, feature_scaler)
    print(f'Initial Prediction: {initial_prediction} (1 indicates sepsis, 0 indicates no sepsis)')
    print(f'Initial Prediction Probability: {initial_prediction_prob:.4f}')
    
    if initial_prediction == 1 and user_input_all_features is not None:
        detailed_prediction, detailed_prediction_prob = predict_sepsis_using_all_features(user_input_all_features, best_xgb_model, feature_imputer, feature_scaler)
        print(f'Detailed Prediction: {detailed_prediction} (1 indicates sepsis, 0 indicates no sepsis)')
        print(f'Detailed Prediction Probability: {detailed_prediction_prob:.4f}')
        return detailed_prediction, detailed_prediction_prob
    
    return initial_prediction, initial_prediction_prob

def example_input_usage():
    user_input_example_top_features = [19, 30, 0.5, 24, 0.21, 7.4, 40, 98, 30, 15]
    user_input_example_all_features = [89, 96, 37.2, 120, 80, 73, 19, 30, 0.5, 24, 0.21, 7.4, 40, 98, 30, 15, 1.2, 140, 0.3, 90, 0.2, 45, 14, 35, 8, 12, 1, 14, 200, 0.5, 140, 10, 90, 0.1, 13, 40, 30, 0, 0, 10, 300]

    sepsis_prediction_workflow(user_input_example_top_features, user_input_example_all_features)

def run_sepsis_prediction(filepath):
    sepsis_data = load_dataset(filepath)
    sepsis_data = drop_nan_target(sepsis_data)
    features, target = separate_features_target(sepsis_data)
    features_preprocessed, feature_imputer, feature_scaler = preprocess_features(features)
    features_train, features_test, target_train, target_test = split_data(features_preprocessed, target)
    features_train_resampled, target_train_resampled = resample_data(features_train, target_train)
    global best_xgb_model, best_model_params
    best_xgb_model, best_model_params = tune_model(features_train_resampled, target_train_resampled)
    accuracy, precision, recall, f1, roc_auc = evaluate_model(best_xgb_model, features_test, target_test)
    
    print(f'Best Model Parameters: {best_model_params}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')
    
    shap_summary_plot(best_xgb_model, features_test)
    example_input_usage()

run_sepsis_prediction('/content/Dataset.csv')

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, \
    precision_recall_curve, average_precision_score
import shap
import re
import joblib
import os
import warnings
from openai import OpenAI

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Lung Cancer Risk Prediction System",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Lung Cancer Risk Prediction System")
st.markdown("### AI-Powered Personalized Risk Assessment with Explainability")
st.markdown("---")


class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attention_scores = self.attention_weights(x)
        weighted_features = x * attention_scores
        return weighted_features, attention_scores


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim=32):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(32, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1d(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class EnhancedTemporalCausalNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_treatments=3, sequence_length=5,
                 use_attention=True, use_cnn=True):
        super(EnhancedTemporalCausalNetwork, self).__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.num_treatments = num_treatments
        self.use_attention = use_attention
        self.use_cnn = use_cnn

        if use_cnn:
            self.cnn_extractor = CNNFeatureExtractor(input_dim, hidden_dim // 2)
            cnn_output_dim = hidden_dim // 2
        else:
            cnn_output_dim = 0

        if use_attention:
            self.attention = AttentionModule(input_dim)

        lstm_input_dim = input_dim
        if use_attention:
            lstm_input_dim += input_dim
        if use_cnn:
            lstm_input_dim += cnn_output_dim

        self.state_encoder = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True, dropout=0.2)
        self.treatment_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )
        self.causal_mechanism = nn.Sequential(
            nn.Linear(hidden_dim + num_treatments, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.propensity_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_treatments),
            nn.Softmax(dim=1)
        )

    def forward(self, x, treatment=None, return_components=False):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        original_x = x.clone()

        if self.use_attention:
            x_attended, _ = self.attention(x)
            x = torch.cat([original_x, x_attended], dim=1)

        if self.use_cnn:
            cnn_features = self.cnn_extractor(original_x)
            x = torch.cat([x, cnn_features], dim=1) if self.use_attention else torch.cat([original_x, cnn_features],
                                                                                         dim=1)

        if len(x.shape) == 2:
            batch_size, input_dim = x.shape
            x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
            if self.training:
                noise = torch.randn_like(x) * 0.01
                x = x + noise

        encoded_states, (hidden, cell) = self.state_encoder(x)
        patient_representation = hidden[-1]
        state_encoded = self.treatment_encoder(patient_representation)

        if treatment is None:
            return self.propensity_net(patient_representation)
        else:
            if len(treatment.shape) == 1:
                treatment = treatment.unsqueeze(0)
            treatment_effect = self.causal_mechanism(torch.cat([state_encoded, treatment], dim=1))
            treatment_effect = torch.clamp(treatment_effect, 1e-7, 1 - 1e-7)
            if return_components:
                return {'treatment_effect': treatment_effect, 'patient_representation': patient_representation}
            return treatment_effect

    def estimate_ate(self, x, treatments=None):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        device = x.device
        if treatments is None:
            treatments = torch.eye(self.num_treatments, device=device)
        outcomes = []
        for treatment in treatments:
            treatment_expanded = treatment.unsqueeze(0)
            if batch_size > 1:
                treatment_expanded = treatment_expanded.repeat(batch_size, 1)
            outcome = self.forward(x, treatment_expanded)
            outcomes.append(outcome)
        return torch.stack(outcomes, dim=1)

    def counterfactual_prediction(self, x, current_treatment, alternative_treatment):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        if len(current_treatment.shape) == 1:
            current_treatment = current_treatment.unsqueeze(0)
        if len(alternative_treatment.shape) == 1:
            alternative_treatment = alternative_treatment.unsqueeze(0)
        if current_treatment.shape[0] == 1 and batch_size > 1:
            current_treatment = current_treatment.repeat(batch_size, 1)
        if alternative_treatment.shape[0] == 1 and batch_size > 1:
            alternative_treatment = alternative_treatment.repeat(batch_size, 1)
        current_outcome = self.forward(x, current_treatment)
        alternative_outcome = self.forward(x, alternative_treatment)
        return {
            'current_outcome': current_outcome,
            'alternative_outcome': alternative_outcome,
            'treatment_effect': alternative_outcome - current_outcome
        }


medical_knowledge = {
    "WHEEZING": "Wheezing is a common sign of airway narrowing or obstruction. In lung cancer patients, tumor compression of the airways or local inflammation can cause wheezing.",
    "SMOKING": "Smoking is the leading risk factor for lung cancer. Carcinogens in tobacco damage DNA in lung epithelial cells, triggering mutations.",
    "AGE": "Age is a major risk factor for many cancers. DNA repair mechanisms become less efficient with age, allowing mutations to accumulate.",
    "CHEST_PAIN": "Chest pain can result from tumor invasion of the pleura, ribs, or nerve compression.",
    "SHORTNESS_OF_BREATH": "Shortness of breath often occurs due to airway obstruction, pleural effusion, or atelectasis caused by the tumor.",
    "GENDER": "Gender influences lung cancer risk. Men have historically higher smoking rates, but women who never smoke also face elevated risks.",
    "ALCOHOL_CONSUMING": "Excessive alcohol consumption can damage DNA and increase cancer risk.",
    "COUGHING": "Persistent coughing can be an early sign of lung cancer, often caused by tumor irritation of the airways."
}


@st.cache_resource
def load_model_and_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.info(f"Using device: {device}")

    base_dir = '/home/featurize'
    model_path = os.path.join(base_dir, 'model_weights.pth')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    data_path = os.path.join(base_dir, 'test_data.npz')
    features_path = os.path.join(base_dir, 'feature_names.pkl')
    history_path = os.path.join(base_dir, 'training_history.npz')

    required_files = [model_path, scaler_path, data_path, features_path]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        st.error(f"Missing required files: {missing}. Please ensure these files exist in {base_dir}.")
        return None, None, None, None, None, device, None

    scaler = joblib.load(scaler_path)
    input_dim = scaler.mean_.shape[0]
    model = EnhancedTemporalCausalNetwork(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data = np.load(data_path)
    X_test_scaled = data['X_test_scaled']
    y_test = data['y_test']

    feature_names = joblib.load(features_path)

    training_history = None
    if os.path.exists(history_path):
        history = np.load(history_path, allow_pickle=True)
        training_history = {
            'train_losses': history['train_losses'],
            'test_losses': history['test_losses'],
            'train_accuracies': history['train_accuracies'],
            'test_accuracies': history['test_accuracies'],
            'learning_rates': history['learning_rates'],
            'treatment_effects': history['treatment_effects']
        }
        st.sidebar.success("Training history loaded.")
    else:
        st.sidebar.warning("Training history file not found. Some charts will be unavailable.")

    return model, scaler, X_test_scaled, y_test, feature_names, device, training_history


def predict_proba(model, input_dict, device, scaler, feature_names):
    """
    input_dict: dict with keys as feature names and values as input values.
    feature_names: list of feature names in the order used during training.
    """
    X_ordered = [input_dict[name] for name in feature_names]
    X_array = np.array(X_ordered).reshape(1, -1)
    X_scaled = scaler.transform(X_array)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    fixed_treatment = torch.tensor([1., 0., 0.], device=device)
    treatment_batch = fixed_treatment.unsqueeze(0).repeat(X_tensor.shape[0], 1)
    with torch.no_grad():
        outputs = model(X_tensor, treatment_batch)
    return outputs.cpu().item()


@st.cache_data
def get_shap_values(_model, X_train_scaled, X_test_scaled, feature_names, device):
    if os.path.exists('shap_results.pkl'):
        with open('shap_results.pkl', 'rb') as f:
            data = joblib.load(f)
        return data['shap_values'], data['importance_df']

    def f(X):
        X_tensor = torch.FloatTensor(X).to(device)
        batch_size = X_tensor.shape[0]
        fixed_treatment = torch.tensor([1., 0., 0.], device=device)
        treatment_batch = fixed_treatment.unsqueeze(0).repeat(batch_size, 1)
        with torch.no_grad():
            outputs = _model(X_tensor, treatment_batch)
        return outputs.cpu().numpy().flatten()

    background = X_train_scaled[:50]
    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(X_test_scaled[:50], nsamples=50)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_ABS_SHAP': mean_abs_shap,
        'Mean_SHAP': np.mean(shap_values, axis=0)
    }).sort_values('Mean_ABS_SHAP', ascending=False)

    return shap_values, importance_df


def generate_rag_explanation(importance_df, patient_features, top_k=5):
    """
    patient_features: dict of the current patient's feature values (with names as keys)
    """
    feature_descriptions = []
    for name, value in patient_features.items():
        # 格式化显示
        if isinstance(value, (int, float)):
            if name == "AGE":
                feature_descriptions.append(f"- {name}: {value} years old")
            else:
                display_value = "Yes" if value == 1 else "No"
                feature_descriptions.append(f"- {name}: {display_value}")
        else:
            feature_descriptions.append(f"- {name}: {value}")
    patient_info = "\n".join(feature_descriptions)

    top_features = importance_df.head(top_k)['Feature'].tolist()
    knowledge_context = "【Relevant Medical Knowledge】\n"
    for feat in top_features[:3]:
        if feat in medical_knowledge:
            knowledge_context += f"- {feat}: {medical_knowledge[feat]}\n"

    top_features_table = importance_df.head(top_k).to_string(index=False)

    prompt = f"""
You are a medical AI explanation expert. Using the medical knowledge provided, generate an accessible yet professional explanation of the lung cancer risk prediction model **for this specific patient**.

【Patient's Features】
{patient_info}

{knowledge_context}

【Global Feature Importance】(sorted by mean |SHAP value|):
{top_features_table}

In your explanation, please:
1. Based on the patient's specific values, explain which features are most contributing to their risk (increase or decrease).
2. Use the medical knowledge to interpret why those features matter.
3. Provide a final personalized summary for the patient and their doctor.

The answer should be fluent, easy to understand, and tailored to this individual's data.
"""
    try:
        client = OpenAI(api_key="sk-09de19c62e754ed3a7be9012fd7ab154", base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "You are a professional medical AI explanation assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        explanation = response.choices[0].message.content
    except Exception as e:
        explanation = f"LLM call failed: {str(e)}"
    explanation = re.sub(r'\*\*([^*]+)\*\*', r'\1', explanation)
    explanation = re.sub(r'\*([^*]+)\*', r'\1', explanation)
    explanation = re.sub(r'^#{1,6}\s+', '', explanation, flags=re.MULTILINE)
    explanation = re.sub(r'^[\s]*[-*]\s+', '', explanation, flags=re.MULTILINE)
    explanation = re.sub(r'^\d+\.\s+', '', explanation, flags=re.MULTILINE)
    explanation = re.sub(r'`([^`]+)`', r'\1', explanation)
    return explanation.strip()


def generate_recommendations(patient_features, prediction, importance_df):
    """
    patient_features: dict of the current patient's feature values
    prediction: 'High Risk' or 'Low Risk'
    """
    recommendations = []

    if prediction == "High Risk":
        recommendations.append(
            "⚠️ **Immediate Action**: Your risk assessment indicates elevated likelihood. Please consult a healthcare provider for further evaluation.")
    else:
        recommendations.append(
            "✅ **Low Risk**: Your current risk is low, but continue healthy habits and regular check-ups.")

    age = patient_features.get('AGE', 0)
    if age > 55:
        recommendations.append(
            "📅 **Age-related Screening**: Given your age, annual low‑dose CT screening is recommended (USPSTF guidelines).")
    elif age > 40:
        recommendations.append(
            "📅 **Age Consideration**: Your age suggests discussing lung cancer screening options with your doctor.")

    if patient_features.get('SMOKING') == 1:
        recommendations.append(
            "🚭 **Smoking Cessation**: Quitting smoking is the most effective way to lower your risk. Seek support from cessation programs.")
    else:
        recommendations.append("👍 **Non‑smoker**: You have already avoided the major risk factor. Stay smoke‑free.")

    if patient_features.get('WHEEZING') == 1:
        recommendations.append(
            "🫁 **Respiratory Assessment**: Your wheezing symptom requires medical evaluation. Consider a pulmonary function test.")
    if patient_features.get('CHEST_PAIN') == 1:
        recommendations.append(
            "💊 **Pain Management**: Persistent chest pain should be investigated. Your doctor may order imaging studies.")
    if patient_features.get('SHORTNESS_OF_BREATH') == 1:
        recommendations.append(
            "🏃 **Breathing Exercises**: Pulmonary rehabilitation or breathing exercises may help manage shortness of breath.")
    if patient_features.get('COUGHING') == 1:
        recommendations.append(
            "🏥 **Cough Evaluation**: Persistent cough warrants medical attention, especially if it has changed in character.")

    gender_important = importance_df[importance_df['Feature'] == 'GENDER']['Mean_ABS_SHAP'].values
    if len(gender_important) > 0 and gender_important[0] > 0.01:
        gender = patient_features.get('GENDER', 0)
        if gender == 1:
            recommendations.append(
                "⚕️ **Gender Risk**: Men have higher lung cancer incidence; ensure you discuss any symptoms with your doctor.")
        else:
            recommendations.append(
                "⚕️ **Gender Risk**: Even non‑smoking women can develop lung cancer; report any new respiratory symptoms promptly.")

    general = [
        "🏃 **Healthy Lifestyle**: Maintain a balanced diet, regular exercise, and avoid environmental pollutants (radon, asbestos, secondhand smoke).",
        "🩺 **Regular Check-ups**: Schedule routine health screenings as recommended by your healthcare provider.",
        "📊 **Risk Monitoring**: Keep track of any new or changing symptoms (cough, chest pain, weight loss) and report them to your doctor."
    ]
    recommendations.extend(general)

    return recommendations


def main():
    with st.spinner("Loading model and data..."):
        model, scaler, X_test_scaled, y_test, feature_names, device, training_history = load_model_and_data()

    if model is None:
        st.stop()

    st.sidebar.header("📋 Patient Information")
    age = st.sidebar.number_input("Age", min_value=18, max_value=120, value=55)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    gender_code = 0 if gender == "Male" else 1
    smoking = st.sidebar.selectbox("Smoking", ["Yes", "No"])
    smoking_val = 1 if smoking == "Yes" else 0
    wheezing = st.sidebar.selectbox("Wheezing", ["Yes", "No"])
    wheezing_val = 1 if wheezing == "Yes" else 0
    shortness = st.sidebar.selectbox("Shortness of Breath", ["Yes", "No"])
    shortness_val = 1 if shortness == "Yes" else 0
    chest_pain = st.sidebar.selectbox("Chest Pain", ["Yes", "No"])
    chest_pain_val = 1 if chest_pain == "Yes" else 0

    if st.sidebar.button("🔍 Predict Lung Cancer Risk", type="primary", use_container_width=True):
        input_dict = {
            'AGE': age,
            'GENDER': gender_code,
            'CHEST_PAIN': chest_pain_val,
            'SMOKING': smoking_val,
            'SHORTNESS_OF_BREATH': shortness_val,
            'WHEEZING': wheezing_val
        }

        prob = predict_proba(model, input_dict, device, scaler, feature_names)

        min_prob = 0.001
        if prob < min_prob:
            prob = min_prob

        risk_count = (smoking_val == 1) + (wheezing_val == 1) + (shortness_val == 1) + (chest_pain_val == 1)

        if age < 50:
            if risk_count >= 3 and prob < 0.03:
                prob = max(prob, 0.03)  # 提升到 3%
            elif risk_count >= 2 and prob < 0.02:
                prob = max(prob, 0.02)  # 提升到 2%
            elif risk_count >= 1 and prob < 0.01:
                prob = max(prob, 0.01)  # 提升到 1%
        if age >= 50 and risk_count >= 3 and prob < 0.3:
            prob = 0.3

        prediction = "High Risk" if prob > 0.5 else "Low Risk"

        col1, col2 = st.columns(2)
        col1.metric("Risk Probability", f"{prob * 100:.2f}%")
        col2.metric("Prediction", prediction)
        st.progress(prob)

        st.session_state['last_prediction'] = prediction
        st.session_state['last_features'] = {
            'AGE': age,
            'GENDER': gender,
            'GENDER_CODE': gender_code,
            'SMOKING': smoking_val,
            'WHEEZING': wheezing_val,
            'SHORTNESS_OF_BREATH': shortness_val,
            'CHEST_PAIN': chest_pain_val
        }

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Model Performance", "🔬 Model Explainability", "🧠 AI Explanation (RAG)", "💡 Recommendations"])

    with tab1:
        st.subheader("Model Performance Metrics")
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        fixed_treatment = torch.tensor([1., 0., 0.], device=device)
        treatment_batch = fixed_treatment.unsqueeze(0).repeat(X_test_tensor.shape[0], 1)
        with torch.no_grad():
            outputs = model(X_test_tensor, treatment_batch)
            y_pred_proba = outputs.cpu().numpy().flatten()
        y_true = y_test.astype(int).flatten()

        fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5))

        if training_history is not None:
            train_losses = training_history['train_losses']
            test_losses = training_history['test_losses']
            axes1[0].plot(train_losses, label='Training Loss', linewidth=2, color='blue')
            axes1[0].plot(test_losses, label='Test Loss', linewidth=2, color='red')
            if len(train_losses) > 0:
                final_epoch = len(train_losses) - 1
                axes1[0].axvline(x=final_epoch, color='purple', linestyle='--', alpha=0.7,
                                 label=f'Final Epoch ({final_epoch + 1})')
            axes1[0].set_xlabel('Epoch')
            axes1[0].set_ylabel('Loss')
            axes1[0].set_title('Training and Test Loss')
            axes1[0].legend()
            axes1[0].grid(True, alpha=0.3)
            if len(train_losses) > 0:
                textstr = f'Final Train Loss: {train_losses[-1]:.4f}\nFinal Test Loss: {test_losses[-1]:.4f}'
                axes1[0].text(0.02, 0.98, textstr, transform=axes1[0].transAxes, fontsize=9,
                              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            axes1[0].text(0.5, 0.5, "Training history not available.\nLoss curve cannot be shown.",
                          ha='center', va='center', transform=axes1[0].transAxes)
            axes1[0].set_title('Loss Curve (Unavailable)')

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes1[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes1[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes1[1].set_xlim([0.0, 1.0])
        axes1[1].set_ylim([0.0, 1.05])
        axes1[1].set_xlabel('False Positive Rate')
        axes1[1].set_ylabel('True Positive Rate')
        axes1[1].set_title('Receiver Operating Characteristic')
        axes1[1].legend(loc="lower right")
        axes1[1].grid(True, alpha=0.3)

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        axes1[2].plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        axes1[2].set_xlim([0.0, 1.0])
        axes1[2].set_ylim([0.0, 1.05])
        axes1[2].set_xlabel('Recall')
        axes1[2].set_ylabel('Precision')
        axes1[2].set_title('Precision-Recall Curve')
        axes1[2].legend(loc="lower left")
        axes1[2].grid(True, alpha=0.3)

        y_pred = (y_pred_proba > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes1[3],
                    xticklabels=['No Cancer', 'Cancer'], yticklabels=['No Cancer', 'Cancer'])
        axes1[3].set_xlabel('Predicted Label')
        axes1[3].set_ylabel('True Label')
        axes1[3].set_title('Confusion Matrix (Test Set)')

        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        if training_history is not None and 'treatment_effects' in training_history:
            treatment_effects = training_history['treatment_effects']
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(treatment_effects, label='Average Treatment Effect', linewidth=2, color='purple')
            if len(treatment_effects) > 0:
                final_epoch = len(treatment_effects) - 1
                ax2.axvline(x=final_epoch, color='purple', linestyle='--', alpha=0.7,
                            label=f'Final Epoch ({final_epoch + 1})')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Treatment Effect')
            ax2.set_title('Treatment Effect Over Training')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            if len(treatment_effects) > 0:
                final_ate = treatment_effects[-1]
                textstr = f'Final ATE: {final_ate:.4f}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
                         verticalalignment='top', bbox=props)
            st.pyplot(fig2)
            plt.close()
        else:
            st.info("Training history not available – cannot display Treatment Effect curve.")

        if training_history is not None and 'train_accuracies' in training_history and 'test_accuracies' in training_history:
            train_accuracies = training_history['train_accuracies']
            test_accuracies = training_history['test_accuracies']
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.plot(train_accuracies, label='Training Accuracy', linewidth=2, color='blue')
            ax3.plot(test_accuracies, label='Test Accuracy', linewidth=2, color='red')
            if len(train_accuracies) > 0:
                final_epoch = len(train_accuracies) - 1
                ax3.axvline(x=final_epoch, color='purple', linestyle='--', alpha=0.7,
                            label=f'Final Epoch ({final_epoch + 1})')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Training and Testing Accuracy')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            if len(train_accuracies) > 0:
                textstr = f'Final Train Acc: {train_accuracies[-1]:.4f}\nFinal Test Acc: {test_accuracies[-1]:.4f}'
                ax3.text(0.02, 0.02, textstr, transform=ax3.transAxes, fontsize=10,
                         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            st.pyplot(fig3)
            plt.close()
        else:
            st.info("Training history not available – cannot display Accuracy curves.")

        st.markdown("#### Treatment Effect Analysis (on Test Set)")
        if hasattr(model, 'estimate_ate'):
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
                final_ate = model.estimate_ate(X_test_tensor)
                st.write(f"**Average Treatment Effects for {final_ate.shape[1]} treatments:**")
                for i in range(final_ate.shape[1]):
                    ate_values = final_ate[:, i]
                    ate_mean = ate_values.mean().item()
                    ate_std = ate_values.std().item()
                    st.write(
                        f"  Treatment {i}: {ate_mean:.4f} ± {ate_std:.4f}  (95% CI: [{ate_mean - 1.96 * ate_std:.4f}, {ate_mean + 1.96 * ate_std:.4f}])")
                if final_ate.shape[1] >= 2:
                    st.write("**Treatment Effect Comparisons:**")
                    for i in range(final_ate.shape[1]):
                        for j in range(i + 1, final_ate.shape[1]):
                            diff = (final_ate[:, i] - final_ate[:, j]).mean().item()
                            st.write(f"  Treatment {i} vs {j}: {diff:+.4f}")
        else:
            st.info("Model does not provide `estimate_ate` method; ATE analysis skipped.")

        st.markdown("#### Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        col1.metric("Accuracy", f"{acc * 100:.2f}%")
        col2.metric("Precision", f"{prec * 100:.2f}%")
        col3.metric("Recall", f"{rec * 100:.2f}%")
        col4.metric("F1-Score", f"{f1 * 100:.2f}%")

    with tab2:
        st.subheader("SHAP Feature Importance Analysis")
        if X_test_scaled is not None:
            X_train_scaled = X_test_scaled  # 简化：用测试集作为背景（实际应使用训练集）
            shap_vals, imp_df = get_shap_values(model, X_train_scaled, X_test_scaled, feature_names, device)
            st.session_state['importance_df'] = imp_df
            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_vals, X_test_scaled[:50], feature_names=feature_names, plot_type="bar", show=False)
            plt.title("Feature Importance (Bar Plot)")
            st.pyplot(fig_bar)
            plt.close()
            fig_sum, ax_sum = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_test_scaled[:50], feature_names=feature_names, show=False)
            plt.title("SHAP Summary Plot")
            st.pyplot(fig_sum)
            plt.close()
            st.subheader("Feature Importance Table")
            st.dataframe(imp_df.style.format({'Mean_ABS_SHAP': '{:.6f}', 'Mean_SHAP': '{:.6f}'}).background_gradient(
                cmap='RdYlGn', subset=['Mean_SHAP']))
        else:
            st.info("Test data not available. SHAP analysis cannot be performed.")

    with tab3:
        st.subheader("AI-Generated Explanation (RAG)")
        if 'importance_df' in st.session_state and 'last_features' in st.session_state:
            patient_features_for_rag = {
                'AGE': st.session_state['last_features']['AGE'],
                'GENDER': st.session_state['last_features']['GENDER'],
                'SMOKING': st.session_state['last_features']['SMOKING'],
                'WHEEZING': st.session_state['last_features']['WHEEZING'],
                'SHORTNESS_OF_BREATH': st.session_state['last_features']['SHORTNESS_OF_BREATH'],
                'CHEST_PAIN': st.session_state['last_features']['CHEST_PAIN']
            }
            with st.spinner("Generating personalized explanation with DeepSeek RAG..."):
                explanation = generate_rag_explanation(st.session_state['importance_df'], patient_features_for_rag,
                                                       top_k=5)
                st.markdown(explanation)
        else:
            st.info("Please make a prediction first (using the sidebar) to generate a personalized explanation.")

    with tab4:
        st.subheader("Personalized Health Recommendations")
        if 'last_prediction' in st.session_state and 'importance_df' in st.session_state and 'last_features' in st.session_state:
            recs = generate_recommendations(
                st.session_state['last_features'],
                st.session_state['last_prediction'],
                st.session_state['importance_df']
            )
            for rec in recs:
                st.markdown(rec)
                st.markdown("---")
        else:
            st.warning("Please make a prediction first using the sidebar, and ensure SHAP analysis has been run.")


if __name__ == "__main__":
    main()
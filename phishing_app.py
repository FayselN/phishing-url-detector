# phishing_app.py

import streamlit as st
import joblib
import pandas as pd
import os
import tldextract
from features.feature_extractor import extract_url_features

# Set page configuration
st.set_page_config(
    page_title="GuardianEye",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Glassmorphism & Cyber-security theme)
st.markdown("""
<style>
    /* Main container and background */
    .stApp {
        background: linear-gradient(135deg, #0e121e 0%, #151a30 100%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Title styling */
    h1 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #6366f1, #3b82f6, #10b981) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
    }
    
    /* Glassmorphic card base style */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 24px;
        margin-bottom: 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    }
    
    /* Border glows based on risk status */
    .glass-card-safe {
        background: rgba(16, 185, 129, 0.04);
        border: 1px solid rgba(16, 185, 129, 0.25);
        box-shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.08);
    }
    .glass-card-safe:hover {
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 12px 40px 0 rgba(16, 185, 129, 0.15);
    }
    
    .glass-card-danger {
        background: rgba(239, 68, 68, 0.04);
        border: 1px solid rgba(239, 68, 68, 0.25);
        box-shadow: 0 8px 32px 0 rgba(239, 68, 68, 0.08);
    }
    .glass-card-danger:hover {
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 12px 40px 0 rgba(239, 68, 68, 0.15);
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    
    /* Security checks styling */
    .check-item {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.3);
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .check-icon {
        font-size: 1.25rem;
        margin-right: 14px;
    }
    .check-text {
        flex-grow: 1;
    }
    .check-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #f1f5f9;
    }
    .check-desc {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    
    /* Progress bar custom styling */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #ef4444, #f59e0b, #10b981) !important;
    }
    
    /* Sidebar premium styling */
    [data-testid="stSidebar"] {
        background: #0b0d19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# 25 model features expected in exact order
model_features = [
    'url_length', 'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes', 'num_digits', 
    'num_special_chars', 'has_ip_in_url', 'has_https', 'has_at_symbol', 'has_double_slash', 
    'subdomain_count', 'domain_length', 'path_length', 'query_length', 'entropy', 
    'ratio_digits_to_letters', 'has_keyword_login', 'has_keyword_secure', 'has_keyword_update', 
    'has_keyword_verify', 'has_keyword_account', 'has_keyword_bank', 'has_random_subdomain', 
    'has_embedded_legit_domain'
]

# Assets & Model paths
MODEL_DIR = "/home/faysel/04_projects/01_mlprojects/phishing-url-detector/model"
ASSET_DIR = "/home/faysel/04_projects/01_mlprojects/phishing-url-detector/assets"

LGB_MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_phishing_model.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_phishing_model.joblib")
TLD_RATE_PATH = os.path.join(ASSET_DIR, "tld_phishing_rate.pkl")
GLOBAL_MEAN_PATH = os.path.join(ASSET_DIR, "global_mean.pkl")

# Load models and assets with caching
@st.cache_resource
def load_all_assets():
    lgb_model = joblib.load(LGB_MODEL_PATH) if os.path.exists(LGB_MODEL_PATH) else None
    xgb_model = joblib.load(XGB_MODEL_PATH) if os.path.exists(XGB_MODEL_PATH) else None
    tld_rate = joblib.load(TLD_RATE_PATH) if os.path.exists(TLD_RATE_PATH) else {}
    global_mean = joblib.load(GLOBAL_MEAN_PATH) if os.path.exists(GLOBAL_MEAN_PATH) else 0.470158
    
    # Cast numpy float to standard float if needed
    if hasattr(global_mean, 'item'):
        global_mean = float(global_mean.item())
        
    return lgb_model, xgb_model, tld_rate, global_mean

lgb_model, xgb_model, tld_rate, global_mean = load_all_assets()

# Sidebar Layout
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #6366f1; font-family: Outfit;'>🛡️ GuardianEye</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: -10px;'>ML-Powered URL Threat Analytics</p>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### ⚙️ Diagnostic Model")
    selected_model_name = st.selectbox(
        "Select primary classifier:",
        ["XGBoost (Primary)", "LightGBM (Alternative)"]
    )

    # Select active model — XGBoost is primary (balanced 50/50 training)
    if selected_model_name.startswith("XGBoost") and xgb_model is not None:
        active_model = xgb_model
        model_type_label = "XGBoost Classifier"
    elif lgb_model is not None:
        active_model = lgb_model
        model_type_label = "LightGBM Classifier"
    else:
        active_model = xgb_model or lgb_model
        model_type_label = "XGBoost Classifier" if xgb_model else "LightGBM Classifier"
        
    st.divider()
    
    # Sidebar stats or descriptions
    st.markdown("### 📊 Database Statistics")
    st.markdown(f"""
    - **Monitored TLD Suffixes:** `{len(tld_rate)}`
    - **Global Phishing Base Rate:** `{global_mean:.4%}`
    - **Feature Dimension Count:** `{len(model_features)}`
    """)
    st.divider()
    st.info("GuardianEye parses only syntactic features of the URL instantly. No slow webpage scraping or network requests are executed.")

# Main Layout
st.markdown("<h1>GuardianEye</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #94a3b8;'>Protect yourself from modern phishing attacks. Paste any URL below for an instant, high-fidelity security diagnostic scan.</p>", unsafe_allow_html=True)

# URL input
url_input = st.text_input(
    "Enter URL for diagnostic scanning:",
    placeholder="https://secure-login-paypal.com/signin",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 5])
with col1:
    scan_clicked = st.button("🔍 Scan URL", type="primary", use_container_width=True)

if scan_clicked or url_input:
    if not url_input.strip():
        st.warning("Please enter a valid URL.")
    else:
        # Extract features
        raw_features = extract_url_features(url_input, tld_rate, global_mean)
        
        if raw_features is None:
            st.error("Failed to extract features from the provided URL. Please verify the format.")
        else:
            # Build inference array
            ordered_features = [raw_features[feat] for feat in model_features]
            X_new = pd.DataFrame([ordered_features], columns=model_features)
            
            # Run model predictions
            pred = active_model.predict(X_new)[0]
            probs = active_model.predict_proba(X_new)[0]
            
            # Mapping: 1 -> Legitimate, 0 -> Phishing
            # Confidence is probability of the predicted class
            pred_class_label = "LEGITIMATE" if pred == 1 else "PHISHING"
            safety_score = probs[1] * 100  # Probability of legitimate
            risk_score = probs[0] * 100    # Probability of phishing
            
            st.markdown("### 🔍 Security Analysis Report")

            # ── Determine verdict tier from raw risk_score ──
            if risk_score < 20:
                verdict_icon  = "🟢"
                verdict_label = "SAFE"
                verdict_sub   = "Structural analysis found no significant phishing markers."
                gauge_color   = "#10b981"
                card_class    = "glass-card-safe"
            elif risk_score < 50:
                verdict_icon  = "🟡"
                verdict_label = "LOW RISK"
                verdict_sub   = "Some mild anomalies detected. Exercise caution before entering credentials."
                gauge_color   = "#f59e0b"
                card_class    = "glass-card"
            elif risk_score < 75:
                verdict_icon  = "🟠"
                verdict_label = "SUSPICIOUS"
                verdict_sub   = "Multiple phishing indicators detected. Do not submit personal information."
                gauge_color   = "#f97316"
                card_class    = "glass-card-danger"
            else:
                verdict_icon  = "🔴"
                verdict_label = "HIGH RISK — PHISHING"
                verdict_sub   = "Critical threat markers detected. This URL likely targets credential theft."
                gauge_color   = "#ef4444"
                card_class    = "glass-card-danger"

            # ── Main gauge banner ──
            st.markdown(f"""
            <div class="glass-card {card_class}" style="padding: 28px 32px;">
              <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
                <!-- Left: Icon + labels -->
                <div style="display: flex; align-items: center; gap: 18px;">
                  <div style="font-size: 3rem; line-height: 1;">{verdict_icon}</div>
                  <div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: {gauge_color};
                                font-family: Outfit, sans-serif; letter-spacing: 0.02em;">
                      {verdict_label}
                    </div>
                    <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 4px; max-width: 420px;">
                      {verdict_sub}<br>
                      <span style="color: #64748b; font-size: 0.78rem;">Analysed by {model_type_label}</span>
                    </div>
                  </div>
                </div>
                <!-- Right: Big percentage -->
                <div style="text-align: right; min-width: 130px;">
                  <div style="font-size: 3.6rem; font-weight: 900; line-height: 1;
                              color: {gauge_color}; font-family: Outfit, sans-serif;">
                    {risk_score:.1f}%
                  </div>
                  <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
                              color: #64748b; margin-top: 2px;">Phishing Risk</div>
                </div>
              </div>
              <!-- Progress bar -->
              <div style="margin-top: 20px; background: rgba(255,255,255,0.06); border-radius: 99px; height: 10px; overflow: hidden;">
                <div style="height: 100%; width: {risk_score:.1f}%;
                            background: linear-gradient(to right, #10b981, #f59e0b, #ef4444);
                            border-radius: 99px; transition: width 0.6s ease;"></div>
              </div>
              <div style="display: flex; justify-content: space-between; margin-top: 6px;
                          font-size: 0.72rem; color: #475569;">
                <span>0% — Safe</span><span>50% — Caution</span><span>100% — Phishing</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Key Metrics Grid (Columns)
            m_col1, m_col2, m_col3 = st.columns(3)

            with m_col1:
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 18px;">
                    <div class="metric-value" style="color: {gauge_color};">
                        {risk_score:.1f}%
                    </div>
                    <div class="metric-label">Phishing Risk Score</div>
                    <div style="font-size: 0.78rem; color: #64748b; margin-top: 6px;">({safety_score:.1f}% legitimate confidence)</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col2:
                # Get TLD data
                ext_tld = tldextract.extract(url_input)
                tld_name = (ext_tld.suffix or "unknown").upper()
                tld_phish_pct = raw_features.get('tld_phishing_rate', global_mean) * 100
                risk_tier = raw_features.get('tld_risk_tier', 0)
                
                tier_colors = ["#10b981", "#3b82f6", "#f59e0b", "#f97316", "#ef4444"]
                tier_names = ["Low", "Elevated", "Moderate", "High", "Critical"]
                
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 18px;">
                    <div class="metric-value" style="color: {tier_colors[risk_tier]};">
                        .{tld_name}
                    </div>
                    <div class="metric-label">TLD Suffix ({tier_names[risk_tier]} Risk)</div>
                </div>
                """, unsafe_allow_html=True)
                
            with m_col3:
                # Entropy complexity
                entropy_val = raw_features.get('entropy', 0.0)
                entropy_status = "Low" if entropy_val < 3.2 else "Moderate" if entropy_val < 4.2 else "High"
                entropy_color = "#10b981" if entropy_val < 3.2 else "#f59e0b" if entropy_val < 4.2 else "#ef4444"
                
                st.markdown(f"""
                <div class="glass-card" style="text-align: center; padding: 18px;">
                    <div class="metric-value" style="color: {entropy_color};">
                        {entropy_val:.3f}
                    </div>
                    <div class="metric-label">Entropy Complexity ({entropy_status})</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Details Layout: Security Checks (Left) & Risk Profile Details (Right)
            d_col1, d_col2 = st.columns([3, 2])
            
            with d_col1:
                st.markdown("#### 🛡️ Deep Safety Checks")
                
                # Check 1: HTTPS Encryption
                has_https = raw_features.get('has_https', 0)
                h_icon, h_title, h_desc, h_color = (
                    ("✅", "Secure HTTPS Connection", "Data transmitted to this URL is encrypted using standard HTTPS/TLS.", "#10b981") if has_https 
                    else ("⚠️", "Unsecured HTTP Connection", "The URL uses unencrypted HTTP. Sent credentials are susceptible to interception.", "#f59e0b")
                )
                
                # Check 2: IP Address Masking
                has_ip = raw_features.get('has_ip_in_url', 0)
                ip_icon, ip_title, ip_desc, ip_color = (
                    ("🚨", "IP Address Hostname Masking", "This URL bypasses normal DNS naming by resolving directly to an IP address.", "#ef4444") if has_ip
                    else ("✅", "Standard DNS Hostname", "Standard domain name resolution is in place; no raw IP spoofing detected.", "#10b981")
                )
                
                # Check 3: Keyword Anomalies
                keyword_flags = [k for k in ['login', 'secure', 'update', 'verify', 'account', 'bank'] if raw_features.get(f'has_keyword_{k}', 0) == 1]
                if keyword_flags:
                    kw_icon, kw_title, kw_desc, kw_color = ("⚠️", f"Suspicious Keywords Found", f"Contains key security terms: {', '.join(keyword_flags)} (often used to bait users).", "#f59e0b")
                else:
                    kw_icon, kw_title, kw_desc, kw_color = ("✅", "No Baiting Keywords", "No standard high-frequency phishing keywords detected in URL string.", "#10b981")
                    
                # Check 4: Brand Spoofing / Impersonation
                has_brand_spoof = raw_features.get('has_embedded_legit_domain', 0)
                brand_icon, brand_title, brand_desc, brand_color = (
                    ("🚨", "Brand Impersonation Detected", "Trusted brand name found in subdomains or paths, but doesn't match primary registered domain.", "#ef4444") if has_brand_spoof
                    else ("✅", "Consistent Brand Registration", "No known brands embedded deceptively within secondary subdomains or directories.", "#10b981")
                )
                
                # Check 5: Special Characters Complexity
                spec_chars = raw_features.get('num_special_chars', 0)
                slashes = raw_features.get('num_slashes', 0)
                dots = raw_features.get('num_dots', 0)
                
                if spec_chars > 8 or slashes > 5 or dots > 3:
                    spec_icon, spec_title, spec_desc, spec_color = ("⚠️", "High Structural Complexity", f"High quantity of structural noise ({spec_chars} special chars, {slashes} slashes, {dots} dots).", "#f59e0b")
                else:
                    spec_icon, spec_title, spec_desc, spec_color = ("✅", "Clean URL Structure", f"Simple structural configuration ({spec_chars} special chars, {slashes} slashes, {dots} dots).", "#10b981")

                # Print check list
                for icon, title, desc, col in [(h_icon, h_title, h_desc, h_color), 
                                               (ip_icon, ip_title, ip_desc, ip_color), 
                                               (kw_icon, kw_title, kw_desc, kw_color), 
                                               (brand_icon, brand_title, brand_desc, brand_color),
                                               (spec_icon, spec_title, spec_desc, spec_color)]:
                    st.markdown(f"""
                    <div class="check-item">
                        <div class="check-icon">{icon}</div>
                        <div class="check-text">
                            <div class="check-title" style="color: {col};">{title}</div>
                            <div class="check-desc">{desc}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with d_col2:
                st.markdown("#### 📋 Domain Risk Profile")
                
                # Custom breakdown card
                st.markdown(f"""
                <div class="glass-card" style="padding: 20px;">
                    <h5 style="margin-top: 0; color: #a21caf; font-family: Outfit; font-weight: 600;">TLD Suffix Intelligence</h5>
                    <div style="margin-top: 10px; display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8;">TLD Suffix:</span>
                        <strong style="color: #f1f5f9;">.{tld_name}</strong>
                    </div>
                    <div style="margin-top: 6px; display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8;">Phishing Rate:</span>
                        <strong style="color: #f1f5f9;">{tld_phish_pct:.3f}%</strong>
                    </div>
                    <div style="margin-top: 6px; display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8;">Risk Tier:</span>
                        <strong style="color: {tier_colors[risk_tier]};">{risk_tier}/4 ({tier_names[risk_tier]})</strong>
                    </div>
                    <div style="margin-top: 6px; display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8;">Trusted TLD Status:</span>
                        <strong style="color: {'#10b981' if raw_features.get('tld_is_trusted', 0) else '#ef4444'};">
                            {'✅ Trusted' if raw_features.get('tld_is_trusted', 0) else '❌ Untrusted/Risk'}
                        </strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Another meta properties card
                st.markdown(f"""
                <div class="glass-card" style="padding: 20px; margin-top: -10px;">
                    <h5 style="margin-top: 0; color: #3b82f6; font-family: Outfit; font-weight: 600;">URL Dimensions</h5>
                    <div style="margin-top: 10px; display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8;">URL Length:</span>
                        <strong style="color: #f1f5f9;">{raw_features.get('url_length', 0)} chars</strong>
                    </div>
                    <div style="margin-top: 6px; display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8;">Domain Length:</span>
                        <strong style="color: #f1f5f9;">{raw_features.get('domain_length', 0)} chars</strong>
                    </div>
                    <div style="margin-top: 6px; display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8;">Subdomain Count:</span>
                        <strong style="color: #f1f5f9;">{raw_features.get('subdomain_count', 0)}</strong>
                    </div>
                    <div style="margin-top: 6px; display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8;">Shortened URL:</span>
                        <strong style="color: {'#e2e8f0' if not raw_features.get('is_shortened_url', 0) else '#ef4444'};">
                            {'❌ No' if not raw_features.get('is_shortened_url', 0) else '⚠️ Yes (Redirect Risk)'}
                        </strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Raw features interactive table
            st.divider()
            with st.expander("🛠️ Advanced Diagnostic Audit (Model Feature Inspector)"):
                st.markdown("The 25 numeric features extracted from the URL and fed directly into the model are listed below:")
                
                # Convert features to a clean df for display
                disp_data = []
                for feat in model_features:
                    val = raw_features.get(feat, 0.0)
                    disp_data.append({
                        "Feature Name": feat,
                        "Value": val,
                        "Type": "Boolean Flag (0/1)" if feat.startswith(('has_', 'is_')) else "Numeric Count" if feat.startswith('num_') or 'count' in feat or 'length' in feat else "Continuous Metric"
                    })
                
                features_df = pd.DataFrame(disp_data)
                st.dataframe(features_df, use_container_width=True, height=450)
                
                # Print raw array format for ease of inspection
                st.markdown("**Raw Inference Feature Vector (ordered float64):**")
                st.code(str(ordered_features))

# Footer styling
st.markdown("<p style='text-align: center; color: #475569; font-size: 0.8rem; margin-top: 50px;'>GuardianEye URL Threat Shield. Powered by LightGBM and XGBoost Classifiers. For research and security demonstration purposes only.</p>", unsafe_allow_html=True)

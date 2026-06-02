import re
import math
import tldextract
from urllib.parse import urlparse
from collections import Counter

def extract_url_features(url, tld_phishing_rate=None, global_mean=0.5):
    """
    Extract all URL features for phishing detection.
    
    Args:
        url (str): URL to extract features from
        tld_phishing_rate (dict): Mapping of TLD → phishing rate (computed from training)
        global_mean (float): Default phishing rate for unknown TLDs (0.0-1.0)
    
    Returns:
        dict: All numeric features (no strings), or None if error
    """

    if tld_phishing_rate is None:
        tld_phishing_rate = {}

    trusted_brands = [
        'paypal', 'paypai', 'paypa1',
        'amazon', 'amazan', 'amazom', 'amazaon', 'arnazon',
        'google', 'gooogle', 'googie',
        'apple', 'appie', 'app1e',
        'microsoft', 'mlcrosoft',
        'facebook', 'faceb00k', 'facebok',
        'netflix', 'netfl1x',
        'ebay', 'ebav',
        'bankofamerica', 'wellsfargo', 'chase'
    ]

    suspicious_keywords = [
        'login', 'secure', 'update', 'verify', 'account', 'bank'
    ]

    shorteners = [
        'bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 
        'short.link', 'buff.ly', 'adf.ly'
    ]

    features = {}

    try:
        # Handle URLs without scheme
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        parsed = urlparse(url)
        extracted = tldextract.extract(url)

        subdomain = extracted.subdomain.lower()
        domain = extracted.domain.lower()
        suffix = extracted.suffix.lower()
        url_lower = url.lower()

        # ─────────────────────────────────
        # Basic counts
        # ─────────────────────────────────

        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_special_chars'] = sum(
            not c.isalnum() and not c.isspace() for c in url
        )

        # ─────────────────────────────────
        # Boolean flags
        # ─────────────────────────────────

        # IP address detection
        ip_pattern_v4 = r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
        ip_pattern_v6 = r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'

        features['has_ip_in_url'] = int(
            bool(re.search(ip_pattern_v4, url) or re.search(ip_pattern_v6, url))
        )

        features['has_https'] = int(url.startswith('https://'))
        features['has_at_symbol'] = int('@' in url)
        features['has_double_slash'] = int('//' in parsed.path)
        features['has_punycode'] = int('xn--' in url_lower)

        # ─────────────────────────────────
        # Domain info
        # ─────────────────────────────────

        features['subdomain_count'] = (
            len(subdomain.split('.')) if subdomain else 0
        )
        features['domain_length'] = len(extracted.domain + '.' + extracted.suffix)

        # ─────────────────────────────────
        # TLD Risk Features (NO RAW STRING)
        # ─────────────────────────────────

        tld_rate = tld_phishing_rate.get(suffix, global_mean)
        features['tld_phishing_rate'] = tld_rate

        # Binned risk tier (0-4)
        if tld_rate >= 0.99:
            features['tld_risk_tier'] = 4
        elif tld_rate >= 0.80:
            features['tld_risk_tier'] = 3
        elif tld_rate >= 0.40:
            features['tld_risk_tier'] = 2
        elif tld_rate >= 0.10:
            features['tld_risk_tier'] = 1
        else:
            features['tld_risk_tier'] = 0

        features['tld_is_trusted'] = int(tld_rate == 0.0)
        features['tld_is_high_risk'] = int(tld_rate >= 0.99)

        # ─────────────────────────────────
        # URL part lengths
        # ─────────────────────────────────

        features['path_length'] = len(parsed.path)
        features['query_length'] = len(parsed.query)

        # ─────────────────────────────────
        # Entropy (randomness score)
        # ─────────────────────────────────

        counts = Counter(url)
        probabilities = [count / len(url) for count in counts.values()]
        features['entropy'] = (
            -sum(p * math.log2(p) for p in probabilities) if url else 0
        )

        # ─────────────────────────────────
        # Digit-to-letter ratio
        # ─────────────────────────────────

        digits = sum(c.isdigit() for c in url)
        letters = sum(c.isalpha() for c in url)
        features['ratio_digits_to_letters'] = (
            digits / letters if letters > 0 else 0
        )

        # ─────────────────────────────────
        # Suspicious keywords
        # ─────────────────────────────────

        for keyword in suspicious_keywords:
            features[f'has_keyword_{keyword}'] = int(keyword in url_lower)

        # ─────────────────────────────────
        # Random domain/subdomain
        # ─────────────────────────────────

        features['has_random_subdomain'] = 0
        parts = subdomain.split('.') if subdomain else []
        parts.append(domain)

        for part in parts:
            if len(part) >= 5:
                vowels = sum(c in 'aeiou' for c in part)
                if vowels / len(part) < 0.2:  # < 20% vowels = random
                    features['has_random_subdomain'] = 1
                    break

        # ─────────────────────────────────
        # Embedded legitimate brand
        # ─────────────────────────────────

        features['has_embedded_legit_domain'] = 0
        check_string = subdomain + parsed.path.lower()

        for brand in trusted_brands:
            if brand in check_string and brand != domain:
                features['has_embedded_legit_domain'] = 1
                break

        # ─────────────────────────────────
        # URL shortener detection
        # ─────────────────────────────────

        features['is_shortened_url'] = int(
            any(shortener in url_lower for shortener in shorteners)
        )

    except Exception as e:
        # Return None on any error — caller must handle
        return None

    return features
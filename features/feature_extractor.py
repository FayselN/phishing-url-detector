# feature_extractor.py

import pandas as pd
import re
from urllib.parse import urlparse
from collections import Counter
import math

selected_features = [
    "URLSimilarityIndex",
    "LetterRatioInURL",
    "IsHTTPS",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "TLDLegitimateProb",
    "url_entropy",
    "CharContinuationRate",
    "DegitRatioInURL",
    "NoOfDegitsInURL",
    "URLLength",
    "NoOfLettersInURL",
    "DomainEntropy",
    "URLCharProb",
    "path_depth",
    "DomainLength",
    "NoOfSubDomain",
    "Pay",
    "TLDLength",
    "TLD_reduced_app",
    "NoOfQMarkInURL",
    "Bank",
    "TLD_reduced_com",
    "NoOfEqualsInURL",
    "TLD_reduced_org",
    "has_login_keyword",
    "TLD_reduced_co",
    "has_bank_or_pay_keyword",
    "TLD_reduced_io",
    "TLD_reduced_uk"
]


def domain_entropy(domain):
    if not domain or len(domain) == 0:
        return 0.0
    chars = [c.lower() for c in domain if c.isalnum()]
    if len(chars) == 0:
        return 0.0
    freq = Counter(chars)
    probs = [freq[c] / len(chars) for c in freq]
    return -sum(p * math.log2(p) for p in probs)

def extract_url_features(url):
    if not isinstance(url, str) or not url.strip():
        # Return zeros for ALL selected features if input is invalid
        return [0.0] * len(selected_features)

    # Parse
    parsed = urlparse(url.lower())
    domain = parsed.netloc
    path = parsed.path

    # Compute basic counts
    url_len = len(url)
    path_len = len(path)
    query_len = len(parsed.query)

    n_dots = url.count('.')
    n_slash = url.count('/')
    n_at = url.count('@')
    n_question = url.count('?')
    n_ampersand = url.count('&')
    n_equals = url.count('=')
    n_digits = sum(c.isdigit() for c in url)

    # Special chars
    n_special = url.count('@') + url.count('#') + url.count('&') + url.count('=') + url.count('%')

    # Ratios
    total_chars = len(url)
    letter_ratio = sum(c.isalpha() for c in url) / max(1, total_chars)
    digit_ratio = n_digits / max(1, total_chars)
    special_ratio = n_special / max(1, total_chars)

    # HTTPS, IP, keywords
    uses_https = 1 if url.startswith('https://') else 0
    has_ip = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0

    # Shorteners (optional; you didn’t train with this, so we skip encoding it)
    shorteners = ["bit.ly", "tinyurl", "goo.gl", "t.co", "cutt.ly"]
    is_shortened = 1 if any(tok in url for tok in shorteners) else 0

    # Keywords (same as in training)
    login_words = ["login", "signin", "sign-in", "log-in", "auth"]
    has_login_keyword = 1 if any(w in url for w in login_words) else 0

    bank_pay_words = ["bank", "paypal", "payment", "pay", "bill", "invoice", "verify", "account", "secure", "update"]
    has_bank_or_pay_keyword = 1 if any(w in url for w in bank_pay_words) else 0

    # TLD (dummy; you should extract domain and then map TLD)
    # For this example, assume we get a TLD like "com", "org", etc.
    # In real world: use tldextract or similar
    # Here we’ll just use a simple suffix
    if "." in domain:
        tld = domain.split(".")[-1]
    else:
        tld = "unknown"

    # Map TLD to your TLD_reduced values (as in training)
    top_tlds = {"com", "org", "uk", "io", "app", "co"}
    tld_reduced = tld if tld in top_tlds else "other_tld"

    # Create mapping to your ONE‑HOT / reduced columns (you must match training exactly)
    # Example: you had dummy columns like TLD_reduced_app, TLD_reduced_com, etc.
    tld_cols = [
        "TLD_reduced_app",
        "TLD_reduced_com",
        "TLD_reduced_co",
        "TLD_reduced_io",
        "TLD_reduced_uk"
    ]
    tld_flags = [1 if tld == t.replace("TLD_reduced_", "") else 0 for t in tld_cols]

    # Build dictionary with the SAME exact keys as selected_features
    # This is the heart of your pipeline.
    features = {
        "URLSimilarityIndex": 0.5,  # dummy; you defined this in training; fill with your computation
        "LetterRatioInURL": letter_ratio,
        "IsHTTPS": uses_https,
        "NoOfOtherSpecialCharsInURL": n_special,
        "SpacialCharRatioInURL": special_ratio,
        "TLDLegitimateProb": 0.8,  # dummy; plug in your TLD‑based prob logic
        "url_entropy": domain_entropy(url),
        "CharContinuationRate": 0.5,  # dummy; you defined this in training
        "DegitRatioInURL": digit_ratio,
        "NoOfDegitsInURL": n_digits,
        "URLLength": url_len,
        "NoOfLettersInURL": int(letter_ratio * len(url)),
        "DomainEntropy": domain_entropy(domain),
        "URLCharProb": 0.5,  # dummy
        "path_depth": len([p for p in path.strip("/").split("/") if p]),
        "DomainLength": len(domain),
        "NoOfSubDomain": max(0, domain.count(".") - 1),
        "Pay": 1 if "pay" in url else 0,  # example; match your training logic
        "TLDLength": len(tld),
        "TLD_reduced_app": int(tld == "app"),
        "NoOfQMarkInURL": n_question,
        "Bank": 1 if "bank" in url else 0,
        "TLD_reduced_com": int(tld == "com"),
        "NoOfEqualsInURL": n_equals,
        "TLD_reduced_org": int(tld == "org"),
        "has_login_keyword": has_login_keyword,
        "TLD_reduced_co": int(tld == "co"),
        "has_bank_or_pay_keyword": has_bank_or_pay_keyword,
        "TLD_reduced_io": int(tld == "io"),
        "TLD_reduced_uk": int(tld == "uk")
    }

    # Make sure output order matches selected_features exactly
    # selected_features is defined in phishing_app.py
    return [features[col] for col in selected_features]

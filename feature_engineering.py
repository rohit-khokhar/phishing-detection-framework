import re
import math
from collections import Counter
from urllib.parse import urlparse
import numpy as np

# =====================================================
# Keywords
# =====================================================

SUSPICIOUS_KEYWORDS = [
    "login", "secure", "account", "verify",
    "update", "confirm", "bank", "signin",
    "security", "paypal", "ebay", "webscr"
]

URGENCY_KEYWORDS = [
    "urgent",
    "immediately",
    "verify",
    "suspend",
    "click here",
    "act now",
    "limited time",
    "confirm",
    "password",
    "security alert",
    "account will be",
    "update your",
    "winner",
    "congratulations",
    "free",
    "click below"
]

# =====================================================
# URL Entropy
# =====================================================

def url_entropy(url):

    if not url:
        return 0

    counts = Counter(url)

    length = len(url)

    entropy = 0

    for c in counts.values():

        p = c / length

        entropy -= p * math.log2(p)

    return entropy


# =====================================================
# Clean email text for TF-IDF
# =====================================================

def clean_for_tfidf(text):

    text = str(text).lower()

    text = re.sub(r'https?://\S+|www\.\S+', ' URLTOKEN ', text)

    text = re.sub(r'<[^>]+>', ' ', text)

    text = re.sub(r'[^a-z\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# =====================================================
# URL Feature Extraction
# =====================================================

def extract_url_features(url):

    url = str(url).strip()

    parsed = urlparse(url)

    domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]

    tld = domain.split('.')[-1] if '.' in domain else ''

    url_length = len(url)

    domain_length = len(domain)

    tld_length = len(tld)

    no_subdomains = max(0, len(domain.split('.')) - 2)

    is_domain_ip = int(
        re.fullmatch(r'(\d{1,3}\.){3}\d{1,3}', domain) is not None
    )

    has_obfuscation = int('%' in url)

    no_obfuscated = url.count('%')

    obfuscation_ratio = no_obfuscated / (url_length + 1)

    letters = sum(c.isalpha() for c in url)

    digits = sum(c.isdigit() for c in url)

    equals = url.count('=')

    qmarks = url.count('?')

    ampersands = url.count('&')

    special = sum(
        not c.isalnum()
        for c in url
    )

    return {

        "URLLength": url_length,

        "DomainLength": domain_length,

        "IsDomainIP": is_domain_ip,

        "TLDLength": tld_length,

        "NoOfSubDomain": no_subdomains,

        "HasObfuscation": has_obfuscation,

        "NoOfObfuscatedChar": no_obfuscated,

        "ObfuscationRatio": obfuscation_ratio,

        "NoOfLettersInURL": letters,

        "LetterRatioInURL": letters / (url_length + 1),

        "NoOfDegitsInURL": digits,

        "DegitRatioInURL": digits / (url_length + 1),

        "NoOfEqualsInURL": equals,

        "NoOfQMarkInURL": qmarks,

        "NoOfAmpersandInURL": ampersands,

        "NoOfOtherSpecialCharsInURL": special,

        "SpacialCharRatioInURL": special / (url_length + 1),

        "URLEntropy": url_entropy(url),

        "HasSuspiciousKeyword": int(
            any(k in url.lower() for k in SUSPICIOUS_KEYWORDS)
        ),

        "DigitToLetterRatio": digits / (letters + 1)
    }

# =====================================================
# Email Feature Extraction
# =====================================================

def count_links(text):
    return len(re.findall(r'https?://\S+|www\.\S+', str(text)))


def count_html_tags(text):
    return len(re.findall(r'<[^>]+>', str(text)))


def count_keywords(text, keywords):
    text = str(text).lower()
    return sum(text.count(k) for k in keywords)


def uppercase_word_ratio(text):
    words = str(text).split()

    if len(words) == 0:
        return 0

    upper = sum(
        1 for w in words
        if w.isupper() and len(w) > 1
    )

    return upper / len(words)


def extract_email_features(text):

    text = str(text)

    text_length = len(text)

    words = text.split()

    word_count = len(words)

    num_links = count_links(text)

    num_exclamations = text.count("!")

    num_digits = sum(c.isdigit() for c in text)

    urgency_keyword_count = count_keywords(
        text,
        URGENCY_KEYWORDS
    )

    num_html_tags = count_html_tags(text)

    upper_ratio = uppercase_word_ratio(text)

    avg_word_length = text_length / (word_count + 1)

    url_density = num_links / (word_count + 1)

    return {

        "text_length": text_length,

        "word_count": word_count,

        "num_links": num_links,

        "num_exclamations": num_exclamations,

        "num_digits": num_digits,

        "urgency_keyword_count": urgency_keyword_count,

        "num_html_tags": num_html_tags,

        "uppercase_word_ratio": upper_ratio,

        "avg_word_length": avg_word_length,

        "url_density": url_density
    }
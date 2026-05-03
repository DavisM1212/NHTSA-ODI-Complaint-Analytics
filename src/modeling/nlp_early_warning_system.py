import argparse
import contextlib
import html
import re
import unicodedata
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import spacy
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from src.config import settings
from src.config.contracts import (
    COMPONENT_MULTILABEL_CASES_STEM,
    COMPONENT_TEXT_SIDECAR_STEM,
    NLP_EARLY_WARNING_COMPLAINT_TOPICS,
    NLP_EARLY_WARNING_OFFICIAL_MANIFEST,
    NLP_EARLY_WARNING_RECURRING_SIGNALS,
    NLP_EARLY_WARNING_RISK_MONITOR,
    NLP_EARLY_WARNING_TERMS,
    NLP_EARLY_WARNING_TOPIC_LIBRARY,
    NLP_EARLY_WARNING_TOPIC_SCAN,
    NLP_EARLY_WARNING_WATCHLIST,
    NLP_EARLY_WARNING_WATCHLIST_SUMMARY,
    NLP_PREPPED_STEM,
)
from src.config.paths import OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_project_directories
from src.data.io_utils import load_frame, write_json
from src.modeling.common.helpers import DATE_COL, ID_COL, log_line
from src.modeling.common.text_fusion import merge_text_sidecar

# -----------------------------------------------------------------------------
# Locked official defaults
# -----------------------------------------------------------------------------
SPACY_MODEL_NAME = 'en_core_web_sm'
DEVELOPMENT_SPLIT_CUTOFF = pd.Timestamp('2025-01-01')
MIN_NLP_WORD_COUNT = 5
SPARSE_RECALL_SURVIVOR_MAX = 3
LEMMA_BATCH_SIZE = 1000
LEMMA_N_PROCESS = 1

TFIDF_MIN_DF = 20
TFIDF_MAX_DF = 0.75
TFIDF_MAX_FEATURES = 50000
CLUE_TFIDF_MAX_FEATURES = 2000
CLUE_TOP_N = 5

TOPIC_K_CANDIDATES = [12, 16, 20, 24]
TOPIC_K_LOCK = 20
TOPIC_STRONG_THRESHOLD = 0.20

NMF_INIT = 'nndsvda'
NMF_SOLVER = 'cd'
NMF_MAX_ITER = 500

COHORT_GROWTH_THRESHOLD = 2.0
COHORT_MIN_COMPLAINTS = 3
RISK_MIN_SEVERITY = 0.35
RISK_MIN_CRITICAL_OP = 0.25
RISK_MIN_HIGH_SPEED = 0.35

TOPIC_LABELS = {
    0: "Mixed driving incident / battery issues",
    1: "Electric power steering assist loss",
    2: "Check engine light / engine fault complaints",
    3: "Airbag warning / passenger airbag sensor issue",
    4: "Brake pedal / braking performance issue",
    5: "Recall notice / VIN eligibility / remedy process",
    6: "Transmission shifting / gear engagement issue",
    7: "Door latch / lock / unintended opening issue",
    8: "Windshield / glass cracking issue",
    9: "Fuel pump / fuel tank / leak or odor issue",
    10: "Steering wheel binding / difficult turning issue",
    11: "Airbag nondeployment / crash restraint issue",
    12: "Cruise / traction / electronic braking control issue",
    13: "Seat belt / passenger restraint issue",
    14: "Diagnosis / mechanic / repair follow-up cluster",
    15: "Oil consumption / low oil issue",
    16: "Coolant leak / cylinder intrusion / overheating issue",
    17: "Vehicle shutoff / stall while driving issue",
    18: "Headlight / low beam visibility issue",
    19: "Tire / suspension / frame rust mixed issue"
}

TOPIC_CATEGORIES = {
    0: 'defect_mixed',
    1: 'defect',
    2: 'defect',
    3: 'defect',
    4: 'defect',
    5: 'recall_process',
    6: 'defect',
    7: 'defect',
    8: 'defect',
    9: 'defect',
    10: 'defect',
    11: 'defect',
    12: 'defect',
    13: 'defect',
    14: 'defect_admin_mixed',
    15: 'defect',
    16: 'defect',
    17: 'defect',
    18: 'defect',
    19: 'defect_mixed'
}

DEFECT_WATCHLIST_TOPICS = [
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    18,
    19
]

CAUTION_TOPICS = [0, 14]

PROCESS_TOPICS = [5]

PRESERVE_STOP_WORDS = {
    'no',
    'not',
    'never',
    'none',
    'without',
    'while',
    'after',
    'before',
    'during',
    'against',
    'off',
    'over',
    'under',
    'down',
    'up'
}

NLP_EXTRA_STOP_WORDS = {
    'vehicle',
    'vehicles',
    'car',
    'contact',
    'owner',
    'consumer',
    'complainant',
    'driver',
    'stated',
    'informed',
    'told',
    'dealer',
    'dealership',
    'manufacturer',
    'repair',
    'repaired',
    'replace',
    'replacement',
    'reported',
    'noticed',
    'referenced',
    'notification',
    'notified',
    'states',
    'campaign',
    'unknown',
    'na',
    'nhtsa',
    'issue',
    'problem',
    'warranty',
    'service',
    'fix',
    'said',
    'known',
    'thank',
    'patience'
}

SIGNAL_TIER_ORDER = {
    'Monitor': 0,
    'Early signal': 1,
    'Moderate signal': 2,
    'High-confidence signal': 3
}

MULTI_REQUIRED_COLS = [
    ID_COL,
    'mfr_name',
    'maketxt',
    'modeltxt',
    'yeartxt',
    'state',
    DATE_COL,
    'severity_primary_flag',
    'severity_broad_flag',
    'component_groups'
]

FOIA_START_RE = re.compile(
    r"""
    (
        parts?\s+of\s+this\s+document\s+have\s+been\s+redacted
        |
        redacted\s+to\s+protect\s+personally\s+identifiable\s+information
        |
        information\s+redacted\s+pursuant\s+to\s+the\s+freedom\s+of\s+information\s+act
        |
        pursuant\s+to\s+the\s+free
        |
        freedom\s+of\s+information\s+act\s*\(?foia\)?
        |
        \bfoia\b\s*,?\s*5\s*u\.?s\.?c
    )
    """,
    flags=re.I | re.X
)

LEADING_MARKER_RE = re.compile(r'(^|(?<=[\s.;:]))[A-Z]{1,4}\*\s*:?', flags=re.I)
TRAILING_MARKER_RE = re.compile(r'\*[A-Z]{1,4}\b', flags=re.I)
UPDATED_RE = re.compile(r'\bUPDATED\s+\d{1,2}/\d{1,2}/\d{2,4}\.?', flags=re.I)

PERSONAL_ARTIFACT_PATTERNS = [
    (re.compile(r'\[?X{2,}\]?', flags=re.I), ' '),
    (re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b'), ' '),
    (re.compile(r'https?://\S+|www\.\S+', flags=re.I), ' '),
    (re.compile(r'\b\d{3}[-.)\s]*\d{3}[-.\s]*\d{4}\b'), ' '),
    (re.compile(r'\b[A-HJ-NPR-Z0-9]{17}\b', flags=re.I), ' ')
]

ADMIN_PHRASES_RE = re.compile(
    r"""
    \b(
        consumer\s+writes\s+in\s+regards\s+to |
        (consumer|contact|owner|driver|complainant)\s+(?:also\s+)?(states|stated|has|had|owns|owned|leased)(?:\s+that)? |
        (consumer|contact|owner|driver|complainant)\s+(received|called|indicated|referenced|related|took|sent) |
        (?:please\s+)?see\s+(?:the\s+)?attached(?:\s+(?:document|documents|complaint|file|files|statement|letter|pdf))?(?:\s+(?:for|with)\s+complaint)? |
        vehicle\s+(was|is|had) |
        manufacturer\s+was\s+notified |
        dealer\s+was\s+notified |
        (?:what\s+)?component\s+or\s+system\s+failed\s+or\s+malfunctioned\s |
        (?:is\s+it\s+)?available\s+for\s+inspection\s+upon\s+request |
        how\s+was\s+your\s+safety\s+or\s+the\s+safety\s+of\s+others\s+put\s+at\s+risk |
        has\s+the\s+problem\s+been\s+reproduced(?:\s+or\s+confirmed(?:\s+by\s+(?:an?\s+)?(?:independent\s+)?dealer(?:\s+or\s+(?:independent\s+)?(?:service\s+center|shop))?)?)? |
        (?:has\s+the\s+)?(?:vehicle\s+or\s+)?component\s+been\s+inspected(?:\s+by\s+the\s+manufacturer(?:,\s+police,\s+insurance\s+representatives(?:,)?\s+or\s+others)?)? |
        were\s+there\s+any\s+warning\s+(lamps|lights) |
        (?:warning\s+)?messages\s+or\s+(?:other\s+)?symptoms |
        problem\s+prior\s+to\s+the\s+failure(?:,\s+and\s+when\s+did\s+the\s+first\s+appear) |
        failure\s+mileage |
        approximate\s+failure\s+mileage |
        vehicle\s+was\s+not\s+diagnosed |
        vehicle\s+was\s+not\s+repaired |
        not\s+diagnosed\s+or\s+repaired |
        not\s+diagnosed\s+nor\s+repaired |
        manufacturer\s+was\s+not\s+notified |
        dealer\s+was\s+not\s+contacted |
        local\s+dealer\s+was\s+not\s+contacted |
        warning\s+light(?:s)?\s+(?:was\s+)?illuminated |
        while\s+driving\s+at\s+an\s+undisclosed\s+speed |
        while\s+driving\s+at\s+an\s+unknown\s+speed |
        at\s+an\s+undisclosed\s+speed |
        at\s+an\s+unknown\s+speed |
        the\s+vehicle\s+was\s+taken\s+to\s+the\s+local\s+dealer |
        the\s+vehicle\s+was\s+taken\s+to\s+an\s+independent\s+mechanic |
        the\s+vehicle\s+was\s+taken\s+to\s+the\s+dealer |
        where\s+it\s+was\s+diagnosed\s+that |
        where\s+it\s+was\s+determined\s+that |
        needed\s+to\s+be\s+replaced |
        was\s+diagnosed\s+that |
        was\s+determined\s+that |
        campaign\s+number
    )\b[\s,:;-]*
    """,
    flags=re.I | re.X
)

ADMIN_SENTENCE_ARTIFACTS_RE = re.compile(
    r"""
    (
        (?:^|(?<=[.!?]\s))
        (?:TL\*\s*)?
        the\s+contact\s+own(?:s|ed)\s+an?\s+[^.!?]{2,140}[.!?] |
        \b(?:the\s+)?(?:approximate\s+)?failure\s+mileage\s+(?:was\s+)?(?:approximately\s+)?(?:unknown|unavailable|not\s+available|[\d,]+)[.!?]? |
        \bthe\s+approximate\s+failure\s+mileage\s+was\s+unknown[.!?]? |
        \b(?:the\s+)?contact\s+received\s+notifications?\s+of\s+(?:the\s+)?nhtsa\s+campaign\s+numbers?[:\s]+[0-9A-Z,\sand]+[^.!?]{0,260}? |
        \b(?:the\s+)?contact\s+stated\s+that\s+the\s+manufacturer\s+exceeded\s+a\s+reasonable\s+amount\s+of\s+time\s+for\s+the\s+repair[.!?]? |
        \bvin\s+tool\s+confirms\s+parts?\s+(?:are\s+)?not\s+available[.!?]? |
        \bparts?\s+distribution\s+disconnect[.!?]? |
        \b(?:the\s+)?contact\s+had\s+not\s+experienced\s+(?:a\s+)?failure[.!?]? |
        \b(?:the\s+)?contact\s+had\s+not\s+experienced\s+the\s+failure[.!?]? |
        \b(?:the\s+)?(?:local\s+|unknown\s+|undisclosed\s+)?dealer\s+(?:was\s+)?(?:contacted|made\s+aware|notified)[^.!?]{0,160}?confirmed\s+that\s+(?:the\s+)?parts?\s+(?:was|were)\s+(?:not\s+yet\s+)?available[^.!?]*[.!?]? |
        \b(?:the\s+)?manufacturer\s+(?:was\s+)?(?:contacted|made\s+aware|notified)[^.!?]{0,160}?confirmed\s+that\s+(?:the\s+)?parts?\s+(?:was|were)\s+(?:not\s+yet\s+)?available[^.!?]*[.!?]? |
        \b(?:the\s+)?(?:local\s+|unknown\s+|undisclosed\s+)?dealer\s+(?:stated|confirmed|informed\s+the\s+contact)\s+that\s+(?:the\s+)?parts?\s+(?:was|were)\s+(?:not\s+yet\s+)?(?:available|unavailable|on\s+back\s*order|on\s+backorder)[^.!?]*[.!?]? |
        \b(?:the\s+)?part\s+was\s+(?:not\s+yet\s+)?available[.!?]? |
        \b(?:the\s+)?parts?\s+(?:was|were)\s+(?:not\s+yet\s+)?(?:available|unavailable|on\s+back\s*order|on\s+backorder)[.!?]? |
        \b(?:the\s+)?vin\s+was\s+(?:not\s+available|unavailable|invalid)[.!?]? |
        \binvalid\s+vin[.!?]? |
        \b(?:after\s+investigating\s+the\s+failure,\s+)?the\s+contact\s+(?:related|referenced|associated|linked|was\s+relating)\s+the\s+failures?\s+(?:to|with)\s+(?:nhtsa\s+campaign\s+number|customer\s+satisfaction\s+program|tsb)[:\s]*[^.!?]*[.!?]? |
        \bupon\s+investigation,\s+the\s+contact\s+(?:associated|linked|discovered)\s+[^.!?]{0,80}?(?:nhtsa\s+campaign\s+number|customer\s+satisfaction\s+program|tsb)[^.!?]*[.!?]? |
        \b(?:the\s+)?contact\s+researched\s+online\s+and\s+related\s+the\s+failures?\s+to\s+(?:nhtsa\s+campaign\s+number|customer\s+satisfaction\s+program|tsb)[^.!?]*[.!?]? |
        \b(?:the\s+)?vehicle\s+(?:was|had\s+been|had\s+not\s+been|had\s+yet\s+to\s+be|was\s+not\s+yet)\s+(?:not\s+)?(?:diagnosed|repaired|diagnosed\s+(?:or|nor)\s+repaired|taken\s+to\s+be\s+diagnosed\s+(?:or|nor)\s+repaired)[^.!?;]*[.!?]? |
        \b(?:the\s+)?vehicle\s+was\s+not\s+taken\s+to\s+(?:a\s+)?(?:local\s+)?dealer\s+or\s+(?:an\s+)?independent\s+mechanic[^.!?]*[.!?]? |
        \b(?:the\s+)?vehicle\s+was\s+not\s+taken\s+to\s+(?:a\s+)?dealer[^.!?]*[.!?]? |
        \b(?:the\s+)?vehicle\s+was\s+taken\s+to\s+(?:the\s+)?(?:local\s+)?dealer\s+to\s+be\s+diagnosed[.!?]? |
        \b(?:the\s+)?vehicle\s+was\s+taken\s+to\s+(?:an\s+)?independent\s+mechanic\s+to\s+be\s+diagnosed[.!?]? |
        \b(?:the\s+)?vehicle\s+was\s+towed\s+to\s+(?:the\s+)?(?:local\s+)?(?:dealer|residence|contact'?s\s+residence|tow\s+yard|tow\s+lot)[^.!?]*[.!?]? |
        \b(?:the\s+)?vehicle\s+remained\s+at\s+the\s+dealer\s+unrepaired[.!?]? |
        \b(?:the\s+)?vehicle\s+was\s+not\s+repaired\s+due\s+to\s+the\s+cost[.!?]? |
        \b(?:the\s+)?vehicle\s+was\s+not\s+repaired\s+and\s+remained\s+(?:at|in\s+the\s+possession\s+of)\s+the\s+dealer [.!?]? |
        \b(?:the\s+)?manufacturer\s+(?:was|had\s+been|had\s+not\s+been|had\s+yet\s+to\s+be|was\s+not\s+yet)\s+(?:not\s+)?(?:made\s+aware|notified|informed|contacted)(?:\s+of\s+(?:the\s+)?(?:failure|issue|recall))?[^.!?]*[.!?]? |
        \b(?:the\s+)?manufacturer\s+(?:was\s+also\s+)?notified\s+of\s+(?:the\s+)?(?:failure|issue)[^.!?]*[.!?]? |
        \b(?:the\s+)?manufacturer\s+was\s+contacted[^.!?]*[.!?]? |
        \b(?:the\s+)?(?:local\s+|unknown\s+|undisclosed\s+)?dealer\s+(?:was|had\s+been|was\s+not)\s+(?:not\s+)?(?:contacted|notified|made\s+aware|informed)(?:\s+of\s+(?:the\s+)?(?:failure|issue))?[^.!?]*[.!?]? |
        \b(?:the\s+)?(?:local\s+)?dealer\s+and\s+(?:the\s+)?manufacturer\s+were\s+(?:not\s+)?(?:contacted|notified|made\s+aware|informed)(?:\s+of\s+(?:the\s+)?(?:failure|issue))?[^.!?]*[.!?]? |
        \bneither\s+the\s+dealer\s+nor\s+the\s+manufacturer\s+were\s+notified\s+of\s+(?:the\s+)?(?:failure|issue)[^.!?]*[.!?]? |
        \b(?:no\s+additional\s+|no\s+further\s+)?assistance\s+was\s+(?:provided|offered)[.!?]? |
        \b(?:the\s+)?manufacturer\s+(?:provided|offered)\s+no\s+assistance[.!?]? |
        \b(?:the\s+)?manufacturer\s+was\s+notified\s+of\s+the\s+failure,\s+but\s+no\s+assistance\s+was\s+(?:provided|offered)[.!?]? |
        \b(?:the\s+)?manufacturer\s+was\s+notified\s+of\s+the\s+failure\s+but\s+(?:provided|offered)\s+no\s+assistance[.!?]? |
        \b(?:a\s+)?case\s+was\s+(?:opened|filed)[.!?]? |
        \b(?:the\s+)?manufacturer\s+(?:opened|filed)\s+a\s+case[.!?]? |
        \b(?:the\s+)?manufacturer\s+was\s+notified\s+of\s+the\s+failure\s+and\s+(?:a\s+)?case\s+was\s+(?:opened|filed)[.!?]? |
        \b(?:the\s+)?contact\s+was\s+(?:referred|advised)\s+to\s+(?:contact\s+)?the\s+nhtsa\s+hotline[^.!?]*[.!?]? |
        \b(?:the\s+)?manufacturer\s+referred\s+the\s+contact\s+to\s+the\s+nhtsa\s+hotline[^.!?]*[.!?]? |
        \b(?:the\s+)?manufacturer\s+was\s+made\s+aware\s+of\s+the\s+failure\s+and\s+referred\s+the\s+contact\s+to\s+(?:the\s+)?nhtsa\s+hotline[^.!?]*[.!?]? |
        \b(?:a|no)\s+police\s+report\s+was\s+(?:not\s+)?filed[.!?]? |
        \ba\s+police\s+report\s+was\s+taken\s+at\s+the\s+scene[.!?]? |
        \b(?:a\s+)?fire\s+(?:department\s+)?report\s+was\s+filed[.!?]? |
        \b(?:there\s+were\s+)?no\s+injur(?:y|ies)\s+(?:were\s+)?(?:sustained|reported)[.!?]? |
        \bno\s+medical\s+attention\s+was\s+required[.!?]? |
        \b(?:there\s+(?:was|were)\s+)?no\s+warning\s+(?:light|lights|lamp|lamps)\s+(?:was|were\s+)?illuminated[.!?]? |
        \bno\s+warning\s+lights?\s+illuminated[.!?]? |
        \bseveral\s+unknown\s+warning\s+(?:lights|lamps)\s+(?:were\s+)?illuminated[.!?]? |
        \ban\s+unknown\s+warning\s+light\s+was\s+illuminated[.!?]? |
        \b(?:the\s+)?cause\s+of\s+the\s+failure\s+was\s+not(?:\s+yet)?\s+determined[.!?]? |
        \bno\s+further\s+information\s+was\s+available[.!?]?
    )
    """,
    flags=re.I | re.X
)

ADMIN_LEADIN_ARTIFACTS_RE = re.compile(
    r"""
    \b(
        at\s+an?\s+(?:undisclosed|unknown)\s+speed |
        at\s+various\s+speeds |
        while\s+driving\s+at\s+an?\s+(?:undisclosed|unknown)\s+speed |
        (?:was\s+)?taken\s+to\s+(?:an?\s+)?(?:independent\s+)?(?:mechanic|dealer|local\s+dealer)(?:\s+who)?\s+(?:diagnosed|determined|informed\s+the\s+contact)\s+(?:that\s+)? |
        (?:an?\s+)?(?:independent\s+)?mechanic\s+(?:diagnosed|determined|informed\s+the\s+contact)\s+(?:that\s+)? |
        (?:the\s+)?dealer\s+(?:diagnosed|determined|informed\s+the\s+contact)\s+(?:that\s+)? |
        (?:the\s+)?contact\s+was\s+informed\s+(?:that\s+)? |
        (?:the\s+)?contact\s+was\s+advised\s+(?:that\s+)? |
        (?:the\s+)?contact\s+stated\s+(?:that\s+)? |
        (?:the\s+)?contact\s+also\s+stated\s+(?:that\s+)? |
        (?:the\s+)?contact\s+indicated\s+(?:that\s+)? |
        (?:the\s+)?contact\s+reported\s+(?:that\s+)? |
        the\s+failure\s+recurred |
        the\s+failure\s+reoccurred |
        the\s+failure\s+persisted |
        the\s+failure\s+was\s+intermittent |
        the\s+failure\s+had\s+been\s+recurring |
        the\s+failure\s+had\s+been\s+reoccurring |
        failure\s+became\s+a\s+regular\s+occurrence
    )\b[\s,:;-]*
    """,
    flags=re.I | re.X
)

DEALER_ADDRESS_RE = re.compile(
    r"""
    \(
        [^)]*(?:\b[A-Z]{2}\s*\d{5}(?:-\d{4})? |
            \bphone\s+number\b |
            \b\d{3}[-.)\s]*\d{3}[-.\s]*\d{4}\b |
            \b\d{3,5}\s+[A-Z0-9][A-Z0-9\s]{2,40}(?:rd|road|st|street|ave|avenue|blvd|boulevard|hwy|highway|dr|drive|ln|lane|pkwy|parkway)\b)
        [^)]*
    \)
    """,
    flags=re.I | re.X
)

RECALL_ARTIFACT_RE = re.compile(
    r"""
    (
        exceeded\s+a\s+reasonable\s+amount\s+of\s+time\s+for\s+the\s+recall\s+repair |
        vin\s+tool\s+confirms\s+parts?\s+not\s+available |
        parts?\s+distribution\s+disconnect |
        parts?\s+(?:was|were)\s+(?:not\s+yet\s+)?(?:available|unavailable|on\s+back\s*order|on\s+backorder) |
        vin\s+was\s+not\s+included\s+in\s+(?:the\s+recall|nhtsa\s+campaign) |
        nhtsa\s+campaign\s+number |
        recall\s+status\s+recall\s+incomplete\s+remedy\s+not\s+yet\s+available |
        \bhowever,\s+the\s+(?:part|parts)\s+to\s+do\s+the\s+recall\s+repairs?\s+(?:was|were)\s+(?:not\s+yet\s+)?(?:available|unavailable)[.!?]? |
        \b(?:the\s+)?contact\s+stated\s+that\s+the\s+manufacturer\s+(?:had\s+)?exceeded\s+a\s+reasonable\s+amount\s+of\s+time\s+for\s+the\s+recall\s+repairs?[.!?]? |
        \b(?:the\s+)?contact\s+was\s+informed\s+that\s+(?:the\s+)?(?:vehicle|vin)\s+was\s+not\s+included\s+in\s+(?:nhtsa\s+campaign\s+number[:\s]+[0-9A-Z]+[^.!?]*|the\s+recall)[.!?]? |
        \b(?:the\s+)?(?:vehicle|vin)\s+was\s+not\s+included\s+in\s+(?:nhtsa\s+campaign\s+number[:\s]+[0-9A-Z]+[^.!?]*|the\s+recall)[.!?]? |
        \b(?:the\s+)?contact\s+was\s+informed\s+that\s+(?:the\s+)?vin\s+was\s+not\s+under\s+recall[.!?]? |
        \bparts\s+not\s+(?:yet\s+)?available\b |
        \bconfirmed\s+that\s+parts\b.{0,30}\bnot\s+(?:yet\s+)?available\b |
        \brecall\b.{0,120}\b(?:part|parts)\b.{0,40}\bnot\s+(?:yet\s+)?available\b |
        \brecall\s+repair\b.{0,40}\bnot\s+(?:yet\s+)?available\b |
        \bpart\s+to\s+do\s+the\s+recall\s+repair\b.{0,40}\bnot\s+(?:yet\s+)?available\b |
        \bparts\s+distribution\s+disconnect\b |
        \bdistribution\s+disconnect\b |
        \bexceeded\s+a\s+reasonable\s+amount\s+of\s+time\b |
        \bexceeded\s+reasonable\s+time\b |
        \brecall\b.{0,160}\breasonable\s+amount\s+of\s+time\b |
        (?:part|parts)\s+to\s+do\s+the\s+recall\s+repairs?\s+(?:was|were)\s+(?:not\s+yet\s+)?(?:available|unavailable)[^.!?]*[.!?]?
    )
    """,
    flags=re.I | re.X
)

IN_OPERATION_RE = re.compile(
    r"""
    \b(
        while\s+driving |
        when\s+driving |
        while\s+operating |
        during\s+operation |
        while\s+the\s+vehicle\s+was\s+in\s+motion |
        while\s+in\s+motion |
        in\s+motion |
        while\s+traveling |
        while\s+driving\s+on |
        driving\s+on\s+the |
        driving\s+approximately |
        at\s+approximately\s+\d{1,3}\s*mph |
        approximately\s+\d{1,3}\s*mph |
        at\s+\d{1,3}\s*mph
    )\b
    """,
    flags=re.I | re.X
)

HIGHWAY_CONTEXT_RE = re.compile(
    r"""
    \b(
        highway |
        interstate |
        freeway |
        expressway |
        highway\s+speed |
        freeway\s+speed |
        interstate\s+speed
    )\b
    """,
    flags=re.I | re.X
)

SPEED_MPH_RE = re.compile(
    r'\b(?:approximately|approx\.?|about|around|at|going|traveling|driving)?\s*(\d{1,3})\s*mph\b',
    flags=re.I
)

CRITICAL_EVENT_RE = re.compile(
    r"""
    \b(
        stall(?:ed|ing)? |
        shut\s+off |
        turned\s+off |
        lost\s+(?:all\s+)?power |
        loss\s+of\s+(?:motive\s+)?power |
        lost\s+motive\s+power |
        would\s+not\s+accelerate |
        could\s+not\s+accelerate |
        failed\s+to\s+accelerate |
        brake(?:s)?\s+(?:failed|failure) |
        brake\s+pedal\s+(?:went\s+to\s+the\s+floor|soft|hard) |
        could\s+not\s+stop |
        unable\s+to\s+stop |
        could\s+not\s+steer |
        unable\s+to\s+steer |
        steering\s+(?:locked|seized) |
        caught\s+fire |
        fire |
        smoke |
        air\s*bags?\s+did\s+not\s+deploy |
        air\s*bags?\s+failed\s+to\s+deploy
    )\b
    """,
    flags=re.I | re.X
)

CONTRACTION_MAP = {
    "can't": "cannot",
    "can not": "cannot",
    "won't": "will not",
    "shan't": "shall not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "'s": "",
}

_SPACY_NLP = None


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the official NLP early-warning pipeline from the locked lemma-based notebook path'
    )
    parser.add_argument('--multi-input-path', default=None)
    parser.add_argument('--text-sidecar-path', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--processed-dir', default=None)
    parser.add_argument('--random-seed', type=int, default=settings.RANDOM_SEED)
    parser.add_argument('--publish-status', default='official')
    parser.add_argument('--skip-cache-rebuild', action='store_true')
    return parser.parse_args()


def write_named_frame(df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == '.parquet':
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    return output_path


def require_columns(df, required_cols, frame_name):
    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'{frame_name} is missing required columns: {missing_text}')


def coerce_bool_like(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    truthy = {'1', 'true', 't', 'yes', 'y'}
    falsy = {'0', 'false', 'f', 'no', 'n', ''}
    normalized = series.astype('string').fillna('').str.strip().str.lower()
    bool_series = normalized.isin(truthy)
    unknown_mask = ~normalized.isin(truthy | falsy)
    if unknown_mask.any():
        bad_values = sorted(normalized.loc[unknown_mask].unique().tolist())[:5]
        raise ValueError(f'Unexpected boolean-like values: {bad_values}')
    return bool_series.astype(bool)


def build_custom_stop_words():
    return sorted((set(ENGLISH_STOP_WORDS) - PRESERVE_STOP_WORDS) | NLP_EXTRA_STOP_WORDS)


def ensure_topic_contracts():
    expected_ids = set(range(TOPIC_K_LOCK))
    if set(TOPIC_LABELS) != expected_ids:
        raise ValueError('TOPIC_LABELS does not match TOPIC_K_LOCK')
    if set(TOPIC_CATEGORIES) != expected_ids:
        raise ValueError('TOPIC_CATEGORIES does not match TOPIC_K_LOCK')

    defect_ids = set(DEFECT_WATCHLIST_TOPICS)
    caution_ids = set(CAUTION_TOPICS)
    process_ids = set(PROCESS_TOPICS)
    assigned_ids = defect_ids | caution_ids | process_ids

    if defect_ids & caution_ids or defect_ids & process_ids or caution_ids & process_ids:
        raise ValueError('Topic watchlist groups overlap')
    if assigned_ids != expected_ids:
        raise ValueError('Topic watchlist groups do not cover every locked topic id')


def load_spacy_model():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load(SPACY_MODEL_NAME, disable=['ner', 'parser'])
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is required for the official NLP early-warning pipeline"
            ) from exc
    return _SPACY_NLP


def lemmatize_series(text_series, batch_size=LEMMA_BATCH_SIZE, n_process=LEMMA_N_PROCESS):
    nlp = load_spacy_model()
    output = []
    for doc in nlp.pipe(
        text_series.fillna('').astype(str),
        batch_size=batch_size,
        n_process=n_process
    ):
        lemmas = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            lemma = token.lemma_.lower().strip()
            if lemma == '-pron-':
                lemma = token.text.lower()
            lemmas.append(lemma)
        output.append(' '.join(lemmas))
    return output


def extract_max_mph(text):
    speeds = []
    for match in SPEED_MPH_RE.finditer(str(text)):
        with contextlib.suppress(ValueError):
            speeds.append(int(match.group(1)))
    if speeds:
        return max(speeds)
    return np.nan


def proximity_flag(text, pattern_a, pattern_b, window=160):
    text = str(text)
    a_spans = [match.span() for match in pattern_a.finditer(text)]
    b_spans = [match.span() for match in pattern_b.finditer(text)]
    for a_start, a_end in a_spans:
        for b_start, b_end in b_spans:
            if abs(a_start - b_start) <= window or abs(a_end - b_end) <= window:
                return True
    return False


def cut_foia_boilerplate(text):
    match = FOIA_START_RE.search(text)
    if match:
        return text[:match.start()]
    return text


def remove_nhtsa_markers(text):
    text = UPDATED_RE.sub(' ', text)
    text = LEADING_MARKER_RE.sub(' ', text)
    return TRAILING_MARKER_RE.sub(' ', text)


def remove_personal_artifacts(text):
    for pattern, replacement in PERSONAL_ARTIFACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def remove_all_admin_artifacts(text):
    for _ in range(2):
        text = DEALER_ADDRESS_RE.sub(' ', text)
        text = ADMIN_SENTENCE_ARTIFACTS_RE.sub(' ', text)
        text = ADMIN_PHRASES_RE.sub(' ', text)
        text = RECALL_ARTIFACT_RE.sub(' ', text)
        text = ADMIN_LEADIN_ARTIFACTS_RE.sub(' ', text)
        text = IN_OPERATION_RE.sub(' ', text)
        text = HIGHWAY_CONTEXT_RE.sub(' ', text)
        text = SPEED_MPH_RE.sub(' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_text(text):
    text = html.unescape(str(text))
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\x00', ' ')
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def expand_contractions(text):
    text = str(text)
    text = text.replace("â€™", "'").replace("â€˜", "'").replace("`", "'")
    for contraction, expanded in CONTRACTION_MAP.items():
        if contraction.startswith("'") or contraction == "n't":
            text = re.sub(re.escape(contraction), expanded, text, flags=re.I)
        else:
            text = re.sub(rf"\b{re.escape(contraction)}\b", expanded, text, flags=re.I)
    return text


def clean_for_topic_modeling(text):
    text = normalize_text(text)
    text = expand_contractions(text)
    text = cut_foia_boilerplate(text)
    text = remove_nhtsa_markers(text)
    text = remove_personal_artifacts(text)
    text = remove_all_admin_artifacts(text)
    text = text.lower()
    text = re.sub(r'(\d+)(mph|mpg|mi|miles|km|rpm)', r'\1 \2', text)
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def remove_vehicle_identifiers(row):
    text = row['nlp_text']
    values_to_remove = [
        str(row.get('maketxt', '')),
        str(row.get('modeltxt', '')),
        str(row.get('yeartxt', ''))
    ]
    for value in values_to_remove:
        value = value.lower().strip()
        if value and value != 'nan':
            text = re.sub(rf'\b{re.escape(value)}\b', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def format_year_series(series):
    numeric = pd.to_numeric(series, errors='coerce')
    year_text = numeric.astype('Int64').astype('string')
    fallback = series.astype('string')
    return year_text.where(year_text.notna(), fallback).fillna('Unknown')


def combine_unique_text(values):
    unique_values = sorted(
        {
            value.strip()
            for value in pd.Series(values).dropna().astype(str).tolist()
            if value and value.strip()
        }
    )
    if not unique_values:
        return ''
    if len(unique_values) == 1:
        return unique_values[0]
    return ' | '.join(unique_values)


def normalize_topic_weights(topic_weights):
    row_sums = topic_weights.sum(axis=1, keepdims=True)
    return np.divide(
        topic_weights,
        row_sums,
        out=np.zeros_like(topic_weights),
        where=row_sums > 0
    )


def get_topic_top_terms(model, feature_names, top_n=10):
    topic_term_map = {}
    for topic_id, weights in enumerate(model.components_):
        top_idx = np.argsort(weights)[::-1][:top_n]
        topic_term_map[topic_id] = feature_names[top_idx].tolist()
    return topic_term_map


# -----------------------------------------------------------------------------
# Corpus prep
# -----------------------------------------------------------------------------
def build_nlp_cache(multi_df, sidecar_df):
    require_columns(multi_df, MULTI_REQUIRED_COLS, 'component multilabel cases')

    work = multi_df[MULTI_REQUIRED_COLS].copy()
    work[ID_COL] = work[ID_COL].astype('string').astype(str)
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors='coerce')
    work = merge_text_sidecar(work, sidecar_df)

    work['severity_primary_flag'] = coerce_bool_like(work['severity_primary_flag'])
    work['severity_broad_flag'] = coerce_bool_like(work['severity_broad_flag'])
    work['cdescr_missing_flag'] = coerce_bool_like(work['cdescr_missing_flag'])
    work['cdescr_placeholder_flag'] = coerce_bool_like(work['cdescr_placeholder_flag'])

    attached_mask = work['cdescr_model_text'].fillna('').str[:25].str.contains(
        r'\battached\s+(?:document|documents)\b',
        case=False,
        regex=True,
        na=False
    )
    work.loc[attached_mask, 'cdescr_placeholder_flag'] = True

    work['nlp_raw'] = work['cdescr'].fillna('').astype(str)
    work['nlp_base'] = work['cdescr_model_text'].fillna('').astype(str)
    work['raw_word_count'] = pd.to_numeric(work['cdescr_word_count'], errors='coerce').fillna(0).astype('Int64')
    work['month'] = work[DATE_COL].dt.to_period('M').dt.to_timestamp()

    work = work.loc[
        ~work['cdescr_missing_flag']
        & ~work['cdescr_placeholder_flag']
        & work['nlp_base'].str.strip().ne('')
    ].copy()

    work['in_operation_flag'] = work['nlp_base'].str.contains(IN_OPERATION_RE, regex=True, na=False)
    work['highway_flag'] = work['nlp_base'].str.contains(HIGHWAY_CONTEXT_RE, regex=True, na=False)
    work['max_reported_mph'] = work['nlp_base'].map(extract_max_mph)
    work['high_speed_flag'] = work['highway_flag'] | work['max_reported_mph'].ge(45).fillna(False)
    work['recall_artifact_flag'] = work['nlp_base'].str.contains(RECALL_ARTIFACT_RE, regex=True, na=False)
    work['critical_event_near_operation_flag'] = work['nlp_base'].map(
        lambda text: proximity_flag(text, CRITICAL_EVENT_RE, IN_OPERATION_RE, window=160)
    )

    work['nlp_text'] = work['nlp_base'].map(clean_for_topic_modeling)
    work['nlp_text'] = work.apply(remove_vehicle_identifiers, axis=1)
    work['nlp_word_count'] = work['nlp_text'].str.split().str.len()
    work = work.loc[work['nlp_word_count'].ge(MIN_NLP_WORD_COUNT)].copy()

    log_line(f'[nlp-ew] lemmatizing {len(work):,} complaint texts')
    work['nlp_text_lemma'] = lemmatize_series(work['nlp_text'])

    stop_words = build_custom_stop_words()
    survivor_count_vectorizer = TfidfVectorizer(
        lowercase=False,
        stop_words=list(stop_words),
        ngram_range=(1, 1),
        min_df=1
    )
    survivor_count_analyzer = survivor_count_vectorizer.build_analyzer()

    work['topic_survivor_unigram_count'] = (
        work['nlp_text_lemma']
        .fillna('')
        .map(survivor_count_analyzer)
        .map(len)
    )
    work['recall_artifact_sparse_flag'] = (
        work['recall_artifact_flag'].fillna(False)
        & work['topic_survivor_unigram_count'].le(SPARSE_RECALL_SURVIVOR_MAX)
    )
    work['topic_model_exclude_flag'] = work['recall_artifact_sparse_flag']

    output_cols = [
        ID_COL,
        'mfr_name',
        'maketxt',
        'modeltxt',
        'yeartxt',
        DATE_COL,
        'state',
        'month',
        'severity_primary_flag',
        'severity_broad_flag',
        'component_groups',
        'raw_word_count',
        'nlp_raw',
        'nlp_word_count',
        'nlp_text',
        'nlp_text_lemma',
        'max_reported_mph',
        'in_operation_flag',
        'highway_flag',
        'high_speed_flag',
        'recall_artifact_flag',
        'critical_event_near_operation_flag',
        'topic_survivor_unigram_count',
        'recall_artifact_sparse_flag',
        'topic_model_exclude_flag'
    ]
    return work[output_cols].copy().reset_index(drop=True)


# -----------------------------------------------------------------------------
# Topic modeling
# -----------------------------------------------------------------------------
def prepare_topic_model_inputs(nlp_df):
    stop_words = build_custom_stop_words()

    train_df = nlp_df.loc[
        (nlp_df[DATE_COL] < DEVELOPMENT_SPLIT_CUTOFF) & ~nlp_df['topic_model_exclude_flag']
    ].copy()
    forward_df = nlp_df.loc[
        (nlp_df[DATE_COL] >= DEVELOPMENT_SPLIT_CUTOFF) & ~nlp_df['topic_model_exclude_flag']
    ].copy()

    if train_df.empty:
        raise ValueError('No development rows available for topic fitting')

    vectorizer = TfidfVectorizer(
        lowercase=False,
        stop_words=list(stop_words),
        ngram_range=(1, 2),
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(train_df['nlp_text_lemma'])
    X_forward = vectorizer.transform(forward_df['nlp_text_lemma'])
    feature_names = np.array(vectorizer.get_feature_names_out())

    return {
        'stop_words': stop_words,
        'train_df': train_df,
        'forward_df': forward_df,
        'vectorizer': vectorizer,
        'X_train': X_train,
        'X_forward': X_forward,
        'feature_names': feature_names
    }


def scan_topic_counts(train_df, X_train, feature_names, random_seed):
    scan_rows = []
    for topic_k in TOPIC_K_CANDIDATES:
        log_line(f'[nlp-ew] scanning topic_k={topic_k}')
        topic_model = NMF(
            n_components=topic_k,
            init=NMF_INIT,
            solver=NMF_SOLVER,
            random_state=random_seed,
            max_iter=NMF_MAX_ITER
        )
        fit_weights = normalize_topic_weights(topic_model.fit_transform(X_train))
        dominant_topic = fit_weights.argmax(axis=1)
        dominant_share = (
            pd.Series(dominant_topic)
            .value_counts(normalize=True)
            .reindex(range(topic_k), fill_value=0.0)
        )

        top_terms = get_topic_top_terms(topic_model, feature_names, top_n=10)
        overlaps = []
        for left_topic in range(topic_k):
            left_terms = set(top_terms[left_topic])
            for right_topic in range(left_topic + 1, topic_k):
                right_terms = set(top_terms[right_topic])
                union = left_terms | right_terms
                overlap = len(left_terms & right_terms) / len(union) if union else 0.0
                overlaps.append(overlap)

        scan_rows.append(
            {
                'topic_k': topic_k,
                'fit_rows': len(train_df),
                'share_strong_dominant': float((fit_weights.max(axis=1) >= TOPIC_STRONG_THRESHOLD).mean()),
                'mean_max_topic_weight': float(fit_weights.max(axis=1).mean()),
                'max_topic_share': float(dominant_share.max()),
                'topics_under_1pct': int((dominant_share < 0.01).sum()),
                'mean_top10_jaccard_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
                'median_top10_jaccard_overlap': float(np.median(overlaps)) if overlaps else 0.0
            }
        )

    topic_scan_df = pd.DataFrame(scan_rows)
    topic_scan_df['topic_scan_score'] = (
        topic_scan_df['share_strong_dominant']
        - topic_scan_df['mean_top10_jaccard_overlap']
        - topic_scan_df['max_topic_share']
        - 0.02 * topic_scan_df['topics_under_1pct']
    )

    eligible = topic_scan_df.loc[
        (topic_scan_df['share_strong_dominant'] >= 0.75)
        & (topic_scan_df['topics_under_1pct'] <= 2)
        & (topic_scan_df['max_topic_share'] <= 0.35)
        & (topic_scan_df['mean_top10_jaccard_overlap'] < 0.35)
    ].copy()
    rank_df = eligible if len(eligible) else topic_scan_df
    recommended_topic_k = int(
        rank_df.sort_values(['topic_scan_score', 'topic_k'], ascending=[False, False]).iloc[0]['topic_k']
    )
    return topic_scan_df, recommended_topic_k


def fit_final_topic_model(nlp_df, prepared, random_seed):
    ensure_topic_contracts()

    train_df = prepared['train_df'].copy()
    forward_df = prepared['forward_df'].copy()
    X_train = prepared['X_train']
    X_forward = prepared['X_forward']
    feature_names = prepared['feature_names']

    topic_model = NMF(
        n_components=TOPIC_K_LOCK,
        init=NMF_INIT,
        solver=NMF_SOLVER,
        random_state=random_seed,
        max_iter=NMF_MAX_ITER
    )
    fit_weights = normalize_topic_weights(topic_model.fit_transform(X_train))
    forward_weights = normalize_topic_weights(topic_model.transform(X_forward))
    top_terms_map = get_topic_top_terms(topic_model, feature_names, top_n=10)

    fit_topic_id = fit_weights.argmax(axis=1)
    forward_topic_id = forward_weights.argmax(axis=1)
    fit_topic_weight_max = fit_weights.max(axis=1)
    forward_topic_weight_max = forward_weights.max(axis=1)

    train_df['topic_id'] = fit_topic_id
    train_df['topic_strength'] = fit_topic_weight_max
    train_df['topic_strength_flag'] = np.where(
        fit_topic_weight_max >= TOPIC_STRONG_THRESHOLD,
        'strong',
        'weak'
    )
    train_df['topic_fit_split'] = 'development_2020_2024'

    forward_df['topic_id'] = forward_topic_id
    forward_df['topic_strength'] = forward_topic_weight_max
    forward_df['topic_strength_flag'] = np.where(
        forward_topic_weight_max >= TOPIC_STRONG_THRESHOLD,
        'strong',
        'weak'
    )
    forward_df['topic_fit_split'] = 'forward_2025_2026'

    modeled_topic_df = pd.concat([train_df, forward_df], ignore_index=True)
    modeled_topic_df['topic_category'] = modeled_topic_df['topic_id'].map(TOPIC_CATEGORIES)
    modeled_topic_df['topic_label'] = modeled_topic_df['topic_id'].map(TOPIC_LABELS)
    modeled_topic_df['watchlist_group'] = 'caution'
    modeled_topic_df.loc[
        modeled_topic_df['topic_id'].isin(DEFECT_WATCHLIST_TOPICS),
        'watchlist_group'
    ] = 'defect_watchlist'
    modeled_topic_df.loc[
        modeled_topic_df['topic_id'].isin(PROCESS_TOPICS),
        'watchlist_group'
    ] = 'process'

    excluded_df = nlp_df.loc[nlp_df['topic_model_exclude_flag']].copy()
    if not excluded_df.empty:
        excluded_df['topic_id'] = pd.NA
        excluded_df['topic_strength'] = np.nan
        excluded_df['topic_strength_flag'] = 'excluded'
        excluded_df['topic_fit_split'] = np.where(
            excluded_df[DATE_COL] < DEVELOPMENT_SPLIT_CUTOFF,
            'development_2020_2024',
            'forward_2025_2026'
        )
        excluded_df['topic_category'] = pd.NA
        excluded_df['topic_label'] = pd.NA
        excluded_df['watchlist_group'] = 'excluded'

    topic_df = pd.concat([modeled_topic_df, excluded_df], ignore_index=True, sort=False)

    topic_library_rows = []
    for topic_id in range(TOPIC_K_LOCK):
        topic_terms = top_terms_map[topic_id]
        topic_examples = (
            train_df
            .assign(topic_weight=fit_weights[:, topic_id])
            .sort_values('topic_weight', ascending=False)
            [['nlp_raw']]
            .drop_duplicates()
            .head(3)['nlp_raw']
            .fillna('')
            .astype(str)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
            .str.slice(0, 280)
            .tolist()
        )

        topic_library_rows.append(
            {
                'topic_id': topic_id,
                'topic_label': TOPIC_LABELS[topic_id],
                'topic_category': TOPIC_CATEGORIES[topic_id],
                'watchlist_group': (
                    'defect_watchlist'
                    if topic_id in DEFECT_WATCHLIST_TOPICS
                    else 'process'
                    if topic_id in PROCESS_TOPICS
                    else 'caution'
                ),
                'topic_top_terms': ' | '.join(topic_terms),
                'development_share': float((fit_topic_id == topic_id).mean()),
                'forward_share': float((forward_topic_id == topic_id).mean()) if len(forward_df) else 0.0,
                'median_topic_weight': float(
                    np.median(
                        modeled_topic_df.loc[
                            modeled_topic_df['topic_id'].eq(topic_id),
                            'topic_strength'
                        ].dropna()
                    )
                ),
                'share_percent_change': float(
                    (
                        (
                            float((forward_topic_id == topic_id).mean()) if len(forward_df) else 0.0
                        )
                        - float((fit_topic_id == topic_id).mean())
                    )
                    / max(float((fit_topic_id == topic_id).mean()), 1e-9)
                    * 100
                ),
                'example_complaint_1': topic_examples[0] if len(topic_examples) >= 1 else '',
                'example_complaint_2': topic_examples[1] if len(topic_examples) >= 2 else '',
                'example_complaint_3': topic_examples[2] if len(topic_examples) >= 3 else ''
            }
        )

    topic_library_df = pd.DataFrame(topic_library_rows)
    return topic_df, modeled_topic_df, topic_library_df


# -----------------------------------------------------------------------------
# Watchlist views
# -----------------------------------------------------------------------------
def explode_component_groups(modeled_topic_df):
    component_topic_df = modeled_topic_df.copy()
    component_topic_df['component_group'] = (
        component_topic_df['component_groups']
        .fillna('')
        .str.split('|')
    )
    component_topic_df = component_topic_df.explode('component_group')
    component_topic_df['component_group'] = component_topic_df['component_group'].astype('string').str.strip()
    return component_topic_df.loc[
        component_topic_df['component_group'].notna()
        & component_topic_df['component_group'].ne('')
    ].copy()


def complete_group_months_since_first(df, group_cols, month_col='month'):
    max_month = df[month_col].max()
    month_df = pd.DataFrame(
        {
            month_col: pd.date_range(df[month_col].min(), max_month, freq='MS')
        }
    )
    group_start_df = (
        df.groupby(group_cols, dropna=False)[month_col]
        .min()
        .reset_index(name='first_observed_month')
    )
    group_month_df = group_start_df.merge(month_df, how='cross')
    group_month_df = group_month_df.loc[
        group_month_df[month_col] >= group_month_df['first_observed_month']
    ].copy()
    return group_month_df.drop(columns='first_observed_month').reset_index(drop=True)


def add_growth_metrics(df, group_cols, count_col='complaints', window=6):
    df = df.sort_values(group_cols + ['month']).copy()
    grouped = df.groupby(group_cols, dropna=False)[count_col]

    df[f'rolling_{window}mo_avg'] = grouped.transform(
        lambda values: values.shift(1).rolling(window, min_periods=3).mean()
    )
    df[f'rolling_{window}mo_sum'] = grouped.transform(
        lambda values: values.shift(1).rolling(window, min_periods=3).sum()
    )

    baseline = df[f'rolling_{window}mo_avg']
    df[f'growth_vs_{window}mo_avg'] = df[count_col] / baseline.clip(lower=1)
    df[f'growth_delta_{window}mo'] = df[count_col] - baseline.fillna(0)
    df['new_or_reemerging_signal'] = (
        df[f'rolling_{window}mo_sum'].fillna(0).eq(0)
        & df[count_col].ge(COHORT_MIN_COMPLAINTS)
    )
    return df


def add_emerging_signal_flag(df, growth_threshold):
    df = df.copy()
    growth = df['growth_vs_6mo_avg'].replace([np.inf, -np.inf], np.nan)
    df['emerging_signal_flag'] = (
        growth.ge(growth_threshold).fillna(False)
        | df['new_or_reemerging_signal'].fillna(False)
    )
    return df


def add_watchlist_score(df):
    df = df.copy()

    df['volume_score'] = np.log1p(df['complaints'])
    df['growth_score'] = (
        df['growth_vs_6mo_avg']
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1)
        .clip(lower=0.5, upper=4.0)
    )
    df['new_signal_boost'] = np.where(df['new_or_reemerging_signal'], 1.35, 1.0)

    broad = df.get('severity_broad_rate', 0)
    primary = df.get('severity_primary_rate', 0)
    critical_op = df.get('critical_event_near_operation_rate', 0)
    in_op = df.get('in_operation_rate', 0)
    highway = df.get('highway_rate', 0)
    high_speed = df.get('high_speed_rate', 0)

    df['severity_multiplier'] = 1 + 0.75 * broad + 1.25 * primary
    df['operational_risk_multiplier'] = (
        1
        + 0.20 * in_op
        + 0.30 * highway
        + 0.35 * high_speed
        + 0.60 * critical_op
    )
    df['confidence_multiplier'] = 0.75 + df['avg_topic_strength'].fillna(0).clip(0, 1)
    group_multiplier_map = {
        'defect_watchlist': 1.00,
        'caution': 0.65,
        'process': 0.45
    }
    df['topic_group_multiplier'] = df['watchlist_group'].map(group_multiplier_map).fillna(0.60)

    df['watchlist_score'] = (
        df['volume_score']
        * df['growth_score']
        * df['new_signal_boost']
        * df['severity_multiplier']
        * df['operational_risk_multiplier']
        * df['confidence_multiplier']
        * df['topic_group_multiplier']
    )
    return df


def build_watchlist_reason(row):
    reasons = []
    if row.get('new_or_reemerging_signal', False):
        reasons.append(
            f"new/reemerging signal: {int(row['complaints'])} complaints after zero prior 6-mo baseline"
        )
    else:
        reasons.append(f"{row['growth_vs_6mo_avg']:.1f}x 6-mo baseline")

    if row.get('severity_broad_rate', 0) >= RISK_MIN_SEVERITY:
        reasons.append('high severity rate')
    if row.get('critical_event_near_operation_rate', 0) >= RISK_MIN_CRITICAL_OP:
        reasons.append('critical event while in operation')
    if row.get('high_speed_rate', 0) >= RISK_MIN_HIGH_SPEED:
        reasons.append('high-speed context')
    if row.get('avg_topic_strength', 0) >= 0.45:
        reasons.append('strong topic match')
    return '; '.join(reasons)


def build_monitor_reason(row):
    reasons = []
    if row.get('severity_broad_rate', 0) >= RISK_MIN_SEVERITY:
        reasons.append('high severity rate')
    if row.get('critical_event_near_operation_rate', 0) >= RISK_MIN_CRITICAL_OP:
        reasons.append('critical event while in operation')
    if row.get('high_speed_rate', 0) >= RISK_MIN_HIGH_SPEED:
        reasons.append('high-speed context')
    if row.get('avg_topic_strength', 0) >= 0.45:
        reasons.append('strong topic match')
    return '; '.join(reasons) if reasons else 'persistent risk context'


def assign_signal_tier(row):
    if row['complaints'] >= 10 and row['growth_vs_6mo_avg'] >= 2:
        return 'High-confidence signal'
    if row['complaints'] >= 5 and (
        row['growth_vs_6mo_avg'] >= 2
        or row['new_or_reemerging_signal']
    ):
        return 'Moderate signal'
    if row['complaints'] >= COHORT_MIN_COMPLAINTS:
        return 'Early signal'
    return 'Monitor'


def build_cohort_watchlist_views(component_topic_df):
    cohort_group_cols = [
        'maketxt',
        'modeltxt',
        'yeartxt',
        'component_group',
        'topic_id',
        'topic_label',
        'topic_category',
        'watchlist_group'
    ]

    cohort_topic_raw = (
        component_topic_df
        .groupby(['month'] + cohort_group_cols, dropna=False)
        .agg(
            mfr_name=('mfr_name', combine_unique_text),
            complaints=(ID_COL, 'nunique'),
            unique_states=('state', lambda values: int(values.dropna().nunique())),
            avg_topic_strength=('topic_strength', 'mean'),
            severity_broad_rate=('severity_broad_flag', 'mean'),
            critical_event_near_operation_rate=('critical_event_near_operation_flag', 'mean'),
            in_operation_rate=('in_operation_flag', 'mean'),
            highway_rate=('highway_flag', 'mean'),
            high_speed_rate=('high_speed_flag', 'mean')
        )
        .reset_index()
    )

    cohort_group_meta = (
        component_topic_df
        .groupby(cohort_group_cols, dropna=False)
        .agg(mfr_name=('mfr_name', combine_unique_text))
        .reset_index()
    )
    cohort_topic_grid = complete_group_months_since_first(component_topic_df, cohort_group_cols)
    cohort_topic_monthly = cohort_topic_grid.merge(
        cohort_group_meta,
        on=cohort_group_cols,
        how='left'
    ).merge(
        cohort_topic_raw,
        on=['month'] + cohort_group_cols + ['mfr_name'],
        how='left'
    )

    cohort_topic_monthly['complaints'] = cohort_topic_monthly['complaints'].fillna(0).astype(int)
    cohort_topic_monthly['unique_states'] = cohort_topic_monthly['unique_states'].fillna(0).astype(int)
    cohort_topic_monthly['avg_topic_strength'] = cohort_topic_monthly['avg_topic_strength'].fillna(0.0)

    for column in [
        'severity_broad_rate',
        'critical_event_near_operation_rate',
        'in_operation_rate',
        'highway_rate',
        'high_speed_rate'
    ]:
        cohort_topic_monthly[column] = cohort_topic_monthly[column].fillna(0.0)

    cohort_topic_monthly = add_growth_metrics(
        cohort_topic_monthly,
        group_cols=cohort_group_cols,
        window=6
    )
    cohort_topic_monthly = add_emerging_signal_flag(
        cohort_topic_monthly,
        growth_threshold=COHORT_GROWTH_THRESHOLD
    )
    cohort_topic_monthly = add_watchlist_score(cohort_topic_monthly)

    cohort_defect_topic_monitor = cohort_topic_monthly.loc[
        cohort_topic_monthly['watchlist_group'].eq('defect_watchlist')
        & cohort_topic_monthly['complaints'].ge(COHORT_MIN_COMPLAINTS)
    ].copy()

    cohort_emerging_watchlist = (
        cohort_defect_topic_monitor
        .loc[cohort_defect_topic_monitor['emerging_signal_flag']]
        .copy()
    )
    cohort_emerging_watchlist['watchlist_reason'] = cohort_emerging_watchlist.apply(
        build_watchlist_reason,
        axis=1
    )
    cohort_emerging_watchlist['signal_tier'] = cohort_emerging_watchlist.apply(
        assign_signal_tier,
        axis=1
    )
    cohort_emerging_watchlist = cohort_emerging_watchlist.sort_values(
        ['month', 'watchlist_score'],
        ascending=[False, False]
    ).reset_index(drop=True)
    cohort_emerging_watchlist['month_rank'] = (
        cohort_emerging_watchlist
        .groupby('month')['watchlist_score']
        .rank(method='first', ascending=False)
        .astype(int)
    )

    cohort_risk_monitor = cohort_defect_topic_monitor.loc[
        ~cohort_defect_topic_monitor['emerging_signal_flag']
        & (
            cohort_defect_topic_monitor['severity_broad_rate'].ge(RISK_MIN_SEVERITY)
            | cohort_defect_topic_monitor['critical_event_near_operation_rate'].ge(RISK_MIN_CRITICAL_OP)
            | cohort_defect_topic_monitor['high_speed_rate'].ge(RISK_MIN_HIGH_SPEED)
        )
    ].copy()
    cohort_risk_monitor['monitor_reason'] = cohort_risk_monitor.apply(build_monitor_reason, axis=1)
    cohort_risk_monitor = cohort_risk_monitor.sort_values(
        ['month', 'watchlist_score'],
        ascending=[False, False]
    ).reset_index(drop=True)
    cohort_risk_monitor['month_rank'] = (
        cohort_risk_monitor
        .groupby('month')['watchlist_score']
        .rank(method='first', ascending=False)
        .astype(int)
    )

    watchlist_cols = [
        'month',
        'mfr_name',
        'maketxt',
        'modeltxt',
        'yeartxt',
        'component_group',
        'topic_id',
        'topic_label',
        'topic_category',
        'complaints',
        'unique_states',
        'rolling_6mo_avg',
        'growth_vs_6mo_avg',
        'growth_delta_6mo',
        'severity_broad_rate',
        'critical_event_near_operation_rate',
        'in_operation_rate',
        'highway_rate',
        'high_speed_rate',
        'avg_topic_strength',
        'watchlist_score',
        'watchlist_reason',
        'signal_tier',
        'month_rank'
    ]

    risk_cols = [
        'month',
        'mfr_name',
        'maketxt',
        'modeltxt',
        'yeartxt',
        'component_group',
        'topic_id',
        'topic_label',
        'topic_category',
        'complaints',
        'unique_states',
        'rolling_6mo_avg',
        'growth_vs_6mo_avg',
        'growth_delta_6mo',
        'severity_broad_rate',
        'critical_event_near_operation_rate',
        'high_speed_rate',
        'avg_topic_strength',
        'watchlist_score',
        'monitor_reason',
        'month_rank'
    ]

    cohort_watchlist_view = cohort_emerging_watchlist[watchlist_cols].copy()
    cohort_risk_monitor_view = cohort_risk_monitor[risk_cols].copy()

    cohort_watchlist_view['yeartxt'] = format_year_series(cohort_watchlist_view['yeartxt'])
    cohort_risk_monitor_view['yeartxt'] = format_year_series(cohort_risk_monitor_view['yeartxt'])

    return cohort_emerging_watchlist, cohort_watchlist_view, cohort_risk_monitor_view


def build_watchlist_summary(component_topic_df, cohort_emerging_watchlist):
    summary_group_cols = [
        'month',
        'maketxt',
        'modeltxt',
        'yeartxt',
        'topic_id',
        'topic_label'
    ]

    flagged_component_cols = summary_group_cols + [
        'component_group',
        'watchlist_score',
        'signal_tier'
    ]

    flagged_component_detail = component_topic_df.merge(
        cohort_emerging_watchlist[flagged_component_cols],
        on=summary_group_cols + ['component_group'],
        how='inner',
        suffixes=('', '_watchlist')
    )

    summary_rows = []
    for group_key, group_df in flagged_component_detail.groupby(summary_group_cols, dropna=False):
        complaint_level_df = (
            group_df
            .sort_values('watchlist_score', ascending=False)
            .drop_duplicates(ID_COL)
        )
        best_tier_rank = (
            group_df[['component_group', 'signal_tier']]
            .drop_duplicates()
            ['signal_tier']
            .map(SIGNAL_TIER_ORDER)
            .fillna(0)
            .max()
        )

        summary_rows.append(
            {
                'month': group_key[0],
                'mfr_name': combine_unique_text(complaint_level_df['mfr_name']),
                'maketxt': group_key[1],
                'modeltxt': group_key[2],
                'yeartxt': group_key[3],
                'topic_id': group_key[4],
                'topic_label': group_key[5],
                'component_groups': ' | '.join(
                    sorted(group_df['component_group'].dropna().astype(str).unique())
                ),
                'complaints': int(complaint_level_df[ID_COL].nunique()),
                'unique_states': int(complaint_level_df['state'].dropna().nunique()),
                'max_component_watchlist_score': float(group_df['watchlist_score'].max()),
                'avg_topic_strength': float(complaint_level_df['topic_strength'].mean()),
                'severity_broad_rate': float(complaint_level_df['severity_broad_flag'].mean()),
                'critical_event_near_operation_rate': float(
                    complaint_level_df['critical_event_near_operation_flag'].mean()
                ),
                'highway_rate': float(complaint_level_df['highway_flag'].mean()),
                'high_speed_rate': float(complaint_level_df['high_speed_flag'].mean()),
                'best_signal_tier': {
                    value: key for key, value in SIGNAL_TIER_ORDER.items()
                }.get(best_tier_rank, 'Monitor')
            }
        )

    if not summary_rows:
        return pd.DataFrame(
            columns=[
                'month',
                'mfr_name',
                'maketxt',
                'modeltxt',
                'yeartxt',
                'topic_id',
                'topic_label',
                'component_groups',
                'complaints',
                'unique_states',
                'max_component_watchlist_score',
                'avg_topic_strength',
                'severity_broad_rate',
                'critical_event_near_operation_rate',
                'highway_rate',
                'high_speed_rate',
                'best_signal_tier'
            ]
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ['month', 'max_component_watchlist_score'],
        ascending=[False, False]
    ).reset_index(drop=True)
    summary_df['yeartxt'] = format_year_series(summary_df['yeartxt'])
    return summary_df


def build_recurring_large_signal_view(cohort_watchlist_view):
    recurring_group_cols = [
        'maketxt',
        'modeltxt',
        'yeartxt',
        'component_group',
        'topic_id',
        'topic_label'
    ]

    recurring_df = (
        cohort_watchlist_view
        .assign(signal_tier_rank=cohort_watchlist_view['signal_tier'].map(SIGNAL_TIER_ORDER).fillna(0))
        .groupby(recurring_group_cols, dropna=False)
        .agg(
            mfr_name=('mfr_name', combine_unique_text),
            months_flagged=('month', 'nunique'),
            first_month=('month', 'min'),
            latest_month=('month', 'max'),
            max_complaints=('complaints', 'max'),
            avg_complaints=('complaints', 'mean'),
            median_complaints=('complaints', 'median'),
            max_growth_vs_6mo_avg=('growth_vs_6mo_avg', 'max'),
            avg_topic_strength=('avg_topic_strength', 'mean'),
            max_watchlist_score=('watchlist_score', 'max'),
            high_confidence_months=('signal_tier', lambda values: int((values == 'High-confidence signal').sum())),
            moderate_or_higher_months=(
                'signal_tier',
                lambda values: int(values.isin(['Moderate signal', 'High-confidence signal']).sum())
            ),
            best_signal_tier_rank=('signal_tier_rank', 'max')
        )
        .reset_index()
    )

    tier_rank_to_label = {value: key for key, value in SIGNAL_TIER_ORDER.items()}
    recurring_df['best_signal_tier'] = recurring_df['best_signal_tier_rank'].map(tier_rank_to_label)
    recurring_df = recurring_df.drop(columns='best_signal_tier_rank')
    recurring_df['avg_complaints'] = recurring_df['avg_complaints'].round(2)
    recurring_df['median_complaints'] = recurring_df['median_complaints'].round(2)
    recurring_df['max_growth_vs_6mo_avg'] = recurring_df['max_growth_vs_6mo_avg'].round(2)
    recurring_df['avg_topic_strength'] = recurring_df['avg_topic_strength'].round(3)
    recurring_df['max_watchlist_score'] = recurring_df['max_watchlist_score'].round(3)
    recurring_df['yeartxt'] = format_year_series(recurring_df['yeartxt'])

    return recurring_df.sort_values(
        ['months_flagged', 'max_complaints', 'max_watchlist_score'],
        ascending=[False, False, False]
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Supporting clue terms
# -----------------------------------------------------------------------------
def build_clue_terms(component_topic_df, cohort_watchlist_view, stop_words):
    clue_columns = [
        'month',
        'mfr_name',
        'maketxt',
        'modeltxt',
        'yeartxt',
        'component_group',
        'topic_id',
        'topic_label',
        'signal_tier',
        'watchlist_score',
        'term',
        'tfidf_score',
        'doc_support',
        'doc_support_share',
        'min_df_used'
    ]

    if cohort_watchlist_view.empty:
        return pd.DataFrame(columns=clue_columns)

    latest_month = cohort_watchlist_view['month'].max()
    latest_watchlist = cohort_watchlist_view.loc[
        cohort_watchlist_view['month'].eq(latest_month)
    ].copy()

    clue_rows = []
    for _, watch_row in latest_watchlist.iterrows():
        mask = (
            component_topic_df['month'].eq(watch_row['month'])
            & component_topic_df['maketxt'].eq(watch_row['maketxt'])
            & component_topic_df['modeltxt'].eq(watch_row['modeltxt'])
            & component_topic_df['yeartxt'].eq(watch_row['yeartxt'])
            & component_topic_df['component_group'].eq(watch_row['component_group'])
            & component_topic_df['topic_id'].eq(watch_row['topic_id'])
        )
        detail_df = (
            component_topic_df.loc[mask]
            .sort_values('topic_strength', ascending=False)
            .drop_duplicates(ID_COL)
        )
        docs = detail_df['nlp_text'].fillna('').astype(str)
        docs = docs.loc[docs.str.strip().ne('')].reset_index(drop=True)
        if docs.empty:
            continue

        min_df_value = 2 if len(docs) >= 5 else 1
        used_min_df = min_df_value

        try:
            vectorizer = TfidfVectorizer(
                lowercase=False,
                stop_words=list(stop_words),
                ngram_range=(1, 2),
                min_df=min_df_value,
                max_features=CLUE_TFIDF_MAX_FEATURES,
                sublinear_tf=True
            )
            matrix = vectorizer.fit_transform(docs)
        except ValueError:
            if min_df_value == 2:
                used_min_df = 1
                try:
                    vectorizer = TfidfVectorizer(
                        lowercase=False,
                        stop_words=list(stop_words),
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=CLUE_TFIDF_MAX_FEATURES,
                        sublinear_tf=True
                    )
                    matrix = vectorizer.fit_transform(docs)
                except ValueError:
                    continue
            else:
                continue

        if matrix.shape[1] == 0:
            continue

        feature_names = np.array(vectorizer.get_feature_names_out())
        mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
        doc_support = np.asarray((matrix > 0).sum(axis=0)).ravel()
        top_idx = np.argsort(mean_scores)[::-1][:CLUE_TOP_N]

        for idx in top_idx:
            if mean_scores[idx] <= 0:
                continue
            clue_rows.append(
                {
                    'month': watch_row['month'],
                    'mfr_name': watch_row['mfr_name'],
                    'maketxt': watch_row['maketxt'],
                    'modeltxt': watch_row['modeltxt'],
                    'yeartxt': watch_row['yeartxt'],
                    'component_group': watch_row['component_group'],
                    'topic_id': watch_row['topic_id'],
                    'topic_label': watch_row['topic_label'],
                    'signal_tier': watch_row['signal_tier'],
                    'watchlist_score': float(watch_row['watchlist_score']),
                    'term': feature_names[idx],
                    'tfidf_score': float(mean_scores[idx]),
                    'doc_support': int(doc_support[idx]),
                    'doc_support_share': float(doc_support[idx] / len(docs)),
                    'min_df_used': int(used_min_df)
                }
            )

    clue_terms_df = pd.DataFrame(clue_rows)
    if clue_terms_df.empty:
        return pd.DataFrame(columns=clue_columns)

    clue_terms_df['yeartxt'] = format_year_series(clue_terms_df['yeartxt'])
    return clue_terms_df


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run_nlp_early_warning_pipeline(
    multi_df,
    sidecar_df,
    output_dir=OUTPUTS_DIR,
    processed_dir=PROCESSED_DATA_DIR,
    publish_status='official',
    random_seed=settings.RANDOM_SEED,
    multi_input_path=None,
    text_sidecar_path=None,
    skip_cache_rebuild=False
):
    ensure_project_directories()
    ensure_topic_contracts()

    output_dir = Path(output_dir)
    processed_dir = Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    cache_target_path = processed_dir / f'{NLP_PREPPED_STEM}.parquet'

    if skip_cache_rebuild:
        if not cache_target_path.exists():
            raise ValueError(
                f'Cannot skip cache rebuild because {cache_target_path} does not exist'
            )
        log_line(f'[nlp-ew] loading existing NLP cache from {cache_target_path}')
        nlp_df = pd.read_parquet(cache_target_path)
        cache_path = cache_target_path
        cache_status = 'reused_existing_cache'
    else:
        if multi_df is None or sidecar_df is None:
            raise ValueError('multi_df and sidecar_df are required unless skip_cache_rebuild=True')
        log_line('[nlp-ew] building NLP-ready complaint cache')
        nlp_df = build_nlp_cache(multi_df, sidecar_df)
        cache_path = write_named_frame(
            nlp_df,
            cache_target_path
        )
        cache_status = 'rebuilt_from_inputs'

    prepared = prepare_topic_model_inputs(nlp_df)
    log_line(
        f'[nlp-ew] topic rows | cache={len(nlp_df):,} '
        f'development={len(prepared["train_df"]):,} forward={len(prepared["forward_df"]):,}'
    )

    topic_scan_df, recommended_topic_k = scan_topic_counts(
        prepared['train_df'],
        prepared['X_train'],
        prepared['feature_names'],
        random_seed=random_seed
    )

    log_line(
        f'[nlp-ew] recommended_topic_k={recommended_topic_k} '
        f'locked_topic_k={TOPIC_K_LOCK}'
    )
    topic_df, modeled_topic_df, topic_library_df = fit_final_topic_model(
        nlp_df,
        prepared,
        random_seed=random_seed
    )

    component_topic_df = explode_component_groups(modeled_topic_df)
    cohort_emerging_watchlist, cohort_watchlist_view, cohort_risk_monitor_view = (
        build_cohort_watchlist_views(component_topic_df)
    )
    cohort_watchlist_summary = build_watchlist_summary(
        component_topic_df,
        cohort_emerging_watchlist
    )
    recurring_large_signal_view = build_recurring_large_signal_view(cohort_watchlist_view)
    clue_terms_df = build_clue_terms(
        component_topic_df,
        cohort_emerging_watchlist,
        prepared['stop_words']
    )

    output_paths = {
        'cache': cache_path,
        'topic_scan': write_named_frame(
            topic_scan_df,
            output_dir / NLP_EARLY_WARNING_TOPIC_SCAN
        ),
        'topic_library': write_named_frame(
            topic_library_df,
            output_dir / NLP_EARLY_WARNING_TOPIC_LIBRARY
        ),
        'complaint_topics': write_named_frame(
            topic_df,
            output_dir / NLP_EARLY_WARNING_COMPLAINT_TOPICS
        ),
        'watchlist': write_named_frame(
            cohort_watchlist_view,
            output_dir / NLP_EARLY_WARNING_WATCHLIST
        ),
        'watchlist_summary': write_named_frame(
            cohort_watchlist_summary,
            output_dir / NLP_EARLY_WARNING_WATCHLIST_SUMMARY
        ),
        'risk_monitor': write_named_frame(
            cohort_risk_monitor_view,
            output_dir / NLP_EARLY_WARNING_RISK_MONITOR
        ),
        'recurring_large_signals': write_named_frame(
            recurring_large_signal_view,
            output_dir / NLP_EARLY_WARNING_RECURRING_SIGNALS
        ),
        'terms': write_named_frame(
            clue_terms_df,
            output_dir / NLP_EARLY_WARNING_TERMS
        )
    }

    latest_month = cohort_watchlist_view['month'].max() if not cohort_watchlist_view.empty else pd.NaT
    manifest = {
        'artifact_role': 'nlp_early_warning_official',
        'scope': 'official lemma-based NLP early-warning pipeline',
        'publish_status': publish_status,
        'random_seed': int(random_seed),
        'runtime_seconds': round(perf_counter() - start, 2),
        'input_refs': {
            'component_multilabel_input': (
                str(multi_input_path) if multi_input_path is not None else COMPONENT_MULTILABEL_CASES_STEM
            ),
            'text_sidecar_input': (
                str(text_sidecar_path) if text_sidecar_path is not None else COMPONENT_TEXT_SIDECAR_STEM
            )
        },
        'text_pipeline': {
            'text_source_col': 'cdescr_model_text',
            'clean_text_col': 'nlp_text',
            'topic_text_col': 'nlp_text_lemma',
            'cache_status': cache_status,
            'spacy_model_name': SPACY_MODEL_NAME,
            'min_clean_word_count': MIN_NLP_WORD_COUNT,
            'sparse_recall_survivor_max': SPARSE_RECALL_SURVIVOR_MAX,
            'preserved_stop_words': sorted(PRESERVE_STOP_WORDS),
            'extra_stop_words': sorted(NLP_EXTRA_STOP_WORDS)
        },
        'topic_model': {
            'topic_k_candidates': list(TOPIC_K_CANDIDATES),
            'recommended_topic_k': int(recommended_topic_k),
            'locked_topic_k': int(TOPIC_K_LOCK),
            'strong_threshold': float(TOPIC_STRONG_THRESHOLD),
            'tfidf_min_df': int(TFIDF_MIN_DF),
            'tfidf_max_df': float(TFIDF_MAX_DF),
            'tfidf_max_features': int(TFIDF_MAX_FEATURES),
            'nmf_init': NMF_INIT,
            'nmf_solver': NMF_SOLVER,
            'nmf_max_iter': int(NMF_MAX_ITER)
        },
        'time_windows': {
            'development_end': str(DEVELOPMENT_SPLIT_CUTOFF - pd.Timedelta(days=1)),
            'forward_start': str(DEVELOPMENT_SPLIT_CUTOFF),
            'growth_window_months': 6
        },
        'row_counts': {
            'nlp_cache_rows': int(len(nlp_df)),
            'excluded_topic_rows': int(nlp_df['topic_model_exclude_flag'].sum()),
            'development_rows_used': int(len(prepared['train_df'])),
            'forward_rows_used': int(len(prepared['forward_df'])),
            'modeled_topic_rows': int(len(modeled_topic_df)),
            'component_topic_rows': int(len(component_topic_df)),
            'watchlist_rows': int(len(cohort_watchlist_view)),
            'risk_monitor_rows': int(len(cohort_risk_monitor_view)),
            'watchlist_summary_rows': int(len(cohort_watchlist_summary)),
            'recurring_large_signal_rows': int(len(recurring_large_signal_view)),
            'clue_term_rows': int(len(clue_terms_df))
        },
        'latest_watchlist_month': (
            latest_month.strftime('%Y-%m-%d') if pd.notna(latest_month) else None
        ),
        'artifacts': {key: str(path) for key, path in output_paths.items()}
    }

    manifest_path = output_dir / NLP_EARLY_WARNING_OFFICIAL_MANIFEST
    write_json(manifest, manifest_path)
    output_paths['manifest'] = manifest_path

    log_line(f'[nlp-ew] wrote {manifest_path}')
    return {
        'nlp_df': nlp_df,
        'topic_scan_df': topic_scan_df,
        'topic_df': topic_df,
        'topic_library_df': topic_library_df,
        'cohort_watchlist_view': cohort_watchlist_view,
        'cohort_risk_monitor_view': cohort_risk_monitor_view,
        'cohort_watchlist_summary': cohort_watchlist_summary,
        'recurring_large_signal_view': recurring_large_signal_view,
        'clue_terms_df': clue_terms_df,
        'manifest': manifest,
        'output_paths': output_paths
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUTS_DIR
    processed_dir = Path(args.processed_dir) if args.processed_dir else PROCESSED_DATA_DIR

    if args.skip_cache_rebuild:
        multi_df = None
        sidecar_df = None
        multi_input_path = None
        sidecar_input_path = None
    else:
        multi_df, multi_input_path = load_frame(
            COMPONENT_MULTILABEL_CASES_STEM,
            input_path=args.multi_input_path
        )
        sidecar_df, sidecar_input_path = load_frame(
            COMPONENT_TEXT_SIDECAR_STEM,
            input_path=args.text_sidecar_path
        )

    run_nlp_early_warning_pipeline(
        multi_df,
        sidecar_df,
        output_dir=output_dir,
        processed_dir=processed_dir,
        publish_status=args.publish_status,
        random_seed=args.random_seed,
        multi_input_path=multi_input_path,
        text_sidecar_path=sidecar_input_path,
        skip_cache_rebuild=args.skip_cache_rebuild
    )


if __name__ == '__main__':
    main()

import os
import io
import json
import html
import base64
import datetime as dt
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf

try:
    import folium
    from folium import JsCode
except Exception:
    folium = None
    JsCode = None

# Optional project modules already created in this hackathon
try:
    from nutrition_data import get_nutrition, nutrition_for_grams, format_nutrition_report
except Exception:
    get_nutrition = nutrition_for_grams = format_nutrition_report = None

try:
    from dhanya_ai_assistant import create_chat, ask_assistant
except Exception:
    create_chat = ask_assistant = None


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Dhanya AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
HERO_IMAGE_PATH = APP_DIR / "dhanya_hero_traditional.png"
MODEL_PATH = APP_DIR / "datagenesis_cnn.keras"
CLASS_NAMES_PATH = APP_DIR / "class_names.txt"

IMAGE_SIZE = (128, 128)

SUPPORTED_CLASSES = [
    "Bamboo rice",
    "Barnyard millet",
    "Chick Peas",
    "Finger millet",
    "green gram",
    "Oats",
    "pearl millet",
    "peas",
    "Rice",
    "Wheat",
]

CATEGORY_MAP = {
    "Bamboo rice": "Grains",
    "Barnyard millet": "Millets",
    "Chick Peas": "Pulses",
    "Finger millet": "Millets",
    "green gram": "Pulses",
    "Oats": "Grains",
    "pearl millet": "Millets",
    "peas": "Pulses",
    "Rice": "Grains",
    "Wheat": "Grains",
}

SCIENTIFIC_NAMES = {
    "Bamboo rice": "Oryza sativa (bamboo-grown rice; local usage varies)",
    "Barnyard millet": "Echinochloa spp.",
    "Chick Peas": "Cicer arietinum",
    "Finger millet": "Eleusine coracana",
    "green gram": "Vigna radiata",
    "Oats": "Avena sativa",
    "pearl millet": "Cenchrus americanus",
    "peas": "Pisum sativum",
    "Rice": "Oryza sativa",
    "Wheat": "Triticum aestivum",
}

ALTERNATE_NAMES = {
    "Bamboo rice": "Bamboo seed rice / Moongil arisi",
    "Barnyard millet": "Kuthiraivali / Udalu",
    "Chick Peas": "Chickpea / Bengal gram / Chana",
    "Finger millet": "Ragi",
    "green gram": "Mung bean / Moong",
    "Oats": "Oat grain",
    "pearl millet": "Bajra / Kambu",
    "peas": "Garden pea",
    "Rice": "Paddy / Arisi",
    "Wheat": "Godhuma",
}

USES_MAP = {
    "Bamboo rice": "Rice-based meals, traditional preparations, porridges, and specialty dishes.",
    "Barnyard millet": "Porridge, upma, idli/dosa-style preparations, and mixed-grain meals.",
    "Chick Peas": "Sundal, hummus, curries, salads, roasted snacks, and flour-based recipes.",
    "Finger millet": "Ragi porridge, dosa, mudde, roti, malt, and baked foods.",
    "green gram": "Sundal, dal, sprouts, khichdi, curries, and soups.",
    "Oats": "Porridge, overnight oats, dosa-style batter, smoothies, and baked foods.",
    "pearl millet": "Kambu koozh, roti, dosa-style recipes, porridges, and mixed-grain foods.",
    "peas": "Curries, pulao, soups, salads, mixed-vegetable dishes, and snacks.",
    "Rice": "Rice meals, idli/dosa batter, pongal, kanji, and many regional preparations.",
    "Wheat": "Roti, chapati, bread, pasta, upma, and other flour-based foods.",
}

CULTIVATION_INFO = {
    "Bamboo rice": "Cultivation is associated with traditional and forest-linked contexts; availability varies by region.",
    "Barnyard millet": "Commonly grown in parts of peninsular and central India; exact production distribution varies by year.",
    "Chick Peas": "Widely cultivated in India, especially across major pulse-growing belts.",
    "Finger millet": "Important in southern and parts of central India, especially dryland farming regions.",
    "green gram": "Grown widely in pulse-producing regions of India across multiple seasons.",
    "Oats": "Cultivated as a cool-season crop in several Indian states; forage oats are also common.",
    "pearl millet": "A major dryland millet crop, especially important in western and northern India.",
    "peas": "Cultivated in several cooler-season vegetable and pulse-growing regions.",
    "Rice": "A major cereal crop grown across many parts of India under diverse production systems.",
    "Wheat": "A major rabi cereal, with large production concentrated in north and central Indian plains.",
}

DIET_IDEAS = {
    "Bamboo rice": ["Cooked rice bowl", "Porridge / kanji", "Traditional rice dishes"],
    "Barnyard millet": ["Millet upma", "Porridge", "Idli / dosa-style batter"],
    "Chick Peas": ["Chickpea salad", "Sundal", "Chana curry", "Hummus"],
    "Finger millet": ["Ragi porridge", "Ragi dosa", "Ragi roti", "Ragi malt"],
    "green gram": ["Sundal", "Moong dal", "Sprouts", "Khichdi"],
    "Oats": ["Porridge", "Overnight oats", "Oats dosa", "Fruit-and-oat bowl"],
    "pearl millet": ["Kambu koozh", "Bajra roti", "Millet porridge"],
    "peas": ["Pea salad", "Pea curry", "Vegetable pulao", "Soup"],
    "Rice": ["Rice bowl", "Pongal", "Kanji", "Idli / dosa"],
    "Wheat": ["Chapati", "Roti", "Bread", "Wheat upma"],
}

# Reference-only comparison values for a few extra foods commonly requested
# in the demo. Nutrition values remain a reference basis, not personalized
# medical advice.

DAILY_INTAKE_GUIDANCE = {
    "Children": [
        {"group": "1–3 years", "sex": "All", "cereals_millets_g": 100, "pulses_beans_g": 50, "energy_kcal": 1110, "protein_g": 38},
        {"group": "4–6 years", "sex": "All", "cereals_millets_g": 160, "pulses_beans_g": 60, "energy_kcal": 1370, "protein_g": 46},
        {"group": "7–9 years", "sex": "All", "cereals_millets_g": 200, "pulses_beans_g": 65, "energy_kcal": 1710, "protein_g": 59},
    ],
    "Teenagers": [
        {"group": "10–12 years", "sex": "Boys", "cereals_millets_g": 280, "pulses_beans_g": 90, "energy_kcal": 2230, "protein_g": 76},
        {"group": "10–12 years", "sex": "Girls", "cereals_millets_g": 250, "pulses_beans_g": 85, "energy_kcal": 2060, "protein_g": 70},
        {"group": "13–15 years", "sex": "Boys", "cereals_millets_g": 390, "pulses_beans_g": 130, "energy_kcal": 2860, "protein_g": 95},
        {"group": "13–15 years", "sex": "Girls", "cereals_millets_g": 300, "pulses_beans_g": 100, "energy_kcal": 2410, "protein_g": 81},
        {"group": "16–18 years", "sex": "Boys", "cereals_millets_g": 450, "pulses_beans_g": 150, "energy_kcal": 3300, "protein_g": 107},
        {"group": "16–18 years", "sex": "Girls", "cereals_millets_g": 315, "pulses_beans_g": 105, "energy_kcal": 2490, "protein_g": 85},
    ],
    "Adults": [
        {"group": "Sedentary", "sex": "Men", "cereals_millets_g": 230, "pulses_beans_g": 75, "energy_kcal": 1900, "protein_g": 64},
        {"group": "Moderate", "sex": "Men", "cereals_millets_g": 320, "pulses_beans_g": 105, "energy_kcal": 2400, "protein_g": 82},
        {"group": "Sedentary", "sex": "Women", "cereals_millets_g": 180, "pulses_beans_g": 60, "energy_kcal": 1660, "protein_g": 55},
        {"group": "Moderate", "sex": "Women", "cereals_millets_g": 250, "pulses_beans_g": 85, "energy_kcal": 2125, "protein_g": 68},
    ],
    "Elderly": [
        {"group": ">60 years", "sex": "Men", "cereals_millets_g": 170, "pulses_beans_g": 75, "energy_kcal": 1740, "protein_g": 62},
        {"group": ">60 years", "sex": "Women", "cereals_millets_g": 140, "pulses_beans_g": 70, "energy_kcal": 1530, "protein_g": 56},
    ],
}



# General meal-window suggestions are culinary guidance, not a universal
# "best time" or medical recommendation.

# Selected major / documented cultivation regions for the current demo.
# These are intended as an educational map layer, not an exhaustive official
# state ranking for each crop. The displayed source note identifies the basis.
CULTIVATION_STATES = {
    "Bamboo rice": {
        "states": ["Assam", "Meghalaya", "Odisha", "Kerala"],
        "note": "Traditional/regional occurrence; bamboo rice is a niche product rather than a standard national production crop.",
    },
    "Barnyard millet": {
        "states": ["Uttarakhand", "Tamil Nadu", "Karnataka", "Uttar Pradesh", "Madhya Pradesh", "Maharashtra", "Andhra Pradesh", "Telangana"],
        "note": "Selected states based on Indian agricultural references describing barnyard millet cultivation and recommended varieties.",
    },
    "Chick Peas": {
        "states": ["Madhya Pradesh", "Maharashtra", "Rajasthan", "Karnataka", "Gujarat"],
        "note": "Selected major pulse-growing regions; state contribution varies by agricultural year.",
    },
    "Finger millet": {
        "states": ["Karnataka", "Tamil Nadu", "Andhra Pradesh", "Odisha", "Maharashtra", "Uttarakhand"],
        "note": "Selected states based on ICAR material on finger-millet adaptation and cultivation.",
    },
    "green gram": {
        "states": ["Rajasthan", "Madhya Pradesh", "Maharashtra", "Karnataka", "Odisha", "Bihar", "Tamil Nadu", "Gujarat", "Andhra Pradesh", "Telangana"],
        "note": "Selected states based on ICAR-IIPR and government green-gram area/production references.",
    },
    "Oats": {
        "states": ["Uttarakhand", "Haryana", "Rajasthan", "Punjab", "Andhra Pradesh", "Karnataka", "Tamil Nadu"],
        "note": "Selected states from ICAR-listed oat adaptation and growing regions.",
    },
    "pearl millet": {
        "states": ["Rajasthan", "Maharashtra", "Gujarat", "Uttar Pradesh", "Haryana"],
        "note": "ICAR identifies these as the major pearl-millet growing states.",
    },
    "peas": {
        "states": ["Uttar Pradesh", "Bihar", "Haryana", "Punjab", "Himachal Pradesh", "Odisha", "Karnataka"],
        "note": "Selected major pea-growing states from National Horticulture Board material.",
    },
    "Rice": {
        "states": ["Uttar Pradesh", "West Bengal", "Telangana", "Andhra Pradesh", "Tamil Nadu", "Bihar", "Odisha", "Punjab"],
        "note": "Selected major rice-producing / cultivating states; exact rankings vary by year.",
    },
    "Wheat": {
        "states": ["Uttar Pradesh", "Madhya Pradesh", "Punjab", "Haryana", "Rajasthan", "Bihar"],
        "note": "Selected major wheat-growing regions; exact production shares vary by year.",
    },
}

CULTIVATION_GEOJSON_FILE = "india-states-simplified.geojson"

MEAL_WINDOW_GUIDANCE = {
    "Bamboo rice": "Lunch or dinner",
    "Barnyard millet": "Breakfast or lunch",
    "Chick Peas": "Lunch, dinner, or an evening snack",
    "Finger millet": "Breakfast or evening meal",
    "green gram": "Breakfast, lunch, or an evening snack",
    "Oats": "Breakfast or an evening snack",
    "pearl millet": "Breakfast or lunch",
    "peas": "Lunch or dinner",
    "Rice": "Lunch or dinner",
    "Wheat": "Breakfast or lunch",
}

STORAGE_REFERENCE = {
    "Bamboo rice": {
        "window": "Use the package best-before date.",
        "note": "Store dry, airtight, cool and away from moisture and pests.",
    },
    "Barnyard millet": {
        "window": "Use the package best-before date.",
        "note": "Store dry, airtight, cool and away from direct heat.",
    },
    "Chick Peas": {
        "window": "USDA dried-bean reference: about 1–2 years from purchase for best quality.",
        "note": "Keep dry and airtight; discard if mold, pests or an off smell is present.",
    },
    "Finger millet": {
        "window": "Use the package best-before date.",
        "note": "Whole grain and flour have different storage behaviour; follow the package.",
    },
    "green gram": {
        "window": "Dry-legume reference: use the package best-before date; quality can decline with age.",
        "note": "Keep dry, airtight and protected from insects.",
    },
    "Oats": {
        "window": "USDA oatmeal reference: about 6–12 months after opening in the pantry.",
        "note": "Keep tightly closed in a cool, dry place and follow the package date.",
    },
    "pearl millet": {
        "window": "Use the package best-before date.",
        "note": "Store dry and airtight; heat and humidity can shorten quality.",
    },
    "peas": {
        "window": "Dry-legume reference: use the package best-before date.",
        "note": "Keep dry, airtight and protected from insects.",
    },
    "Rice": {
        "window": "USDA white-rice reference: about 1 year after opening in the pantry.",
        "note": "Keep tightly sealed and dry; cooked rice has a much shorter refrigerated window.",
    },
    "Wheat": {
        "window": "Use the package best-before date.",
        "note": "Whole grain wheat and wheat flour have different storage behaviour; follow the package.",
    },
}

SHOP_QUERIES = {
    "Bamboo rice": "bamboo rice moongil arisi",
    "Barnyard millet": "barnyard millet kuthiraivali",
    "Chick Peas": "chickpeas kabuli chana",
    "Finger millet": "finger millet ragi",
    "green gram": "green gram whole moong",
    "Oats": "oats 1 kg",
    "pearl millet": "pearl millet bajra",
    "peas": "dried peas",
    "Rice": "rice 1 kg",
    "Wheat": "wheat 1 kg",
}

SHOP_NOTE = (
    "Links open marketplace search results so the user can choose a brand, pack size, "
    "seller, delivery option, and current price. Product availability and prices can change."
)

EXTRA_COMPARE = {
    "Almonds": {
        "category": "Nuts",
        "energy_kcal": 579,
        "protein_g": 21.15,
        "carbs_g": 21.55,
        "fibre_g": 12.50,
        "fat_g": 49.93,
        "source": "USDA FoodData Central-style reference values; varies by product.",
    }
}

TRANSLATIONS = {
    "English": {
        "dashboard": "Dashboard",
        "identify": "Identify Food",
        "assistant": "AI Assistant",
        "explore": "Explore Foods",
        "nutrition": "Nutrition & Diet",
        "daily_intake": "Daily Intake",
        "shop_foods": "Shop Foods",
        "budget_planner": "Budget Planner",
        "cultivation_map": "Cultivation Map",
        "compare": "Compare",
        "history": "History",
        "about": "About",
        "language": "Language / மொழி / भाषा",
        "upload": "Upload an image",
        "camera": "Use camera",
        "predict": "Predict",
        "confidence": "Confidence",
        "category": "Category",
        "who": "Who can include it?",
        "diet": "How to include in a diet",
        "uses": "Uses",
        "science": "Scientific name",
        "other_names": "Other names",
        "cultivation": "Cultivation in India",
        "new_prediction": "Make a new prediction",
    },
    "Tamil": {
        "dashboard": "டாஷ்போர்டு",
        "identify": "உணவை அடையாளம் காண்க",
        "assistant": "AI உதவியாளர்",
        "explore": "உணவுகளைப் பார்வையிடுக",
        "nutrition": "ஊட்டச்சத்து & உணவு",
        "daily_intake": "தினசரி உணவு அளவு",
        "shop_foods": "உணவுகளை வாங்குக",
        "budget_planner": "பட்ஜெட் திட்டம்",
        "cultivation_map": "சாகுபடி வரைபடம்",
        "compare": "ஒப்பிடுக",
        "history": "வரலாறு",
        "about": "பற்றி",
        "language": "Language / மொழி / भाषा",
        "upload": "படத்தை பதிவேற்றவும்",
        "camera": "கேமரா பயன்படுத்தவும்",
        "predict": "கணிக்கவும்",
        "confidence": "நம்பிக்கை",
        "category": "வகை",
        "who": "யார் உணவில் சேர்க்கலாம்?",
        "diet": "உணவில் எப்படி சேர்ப்பது",
        "uses": "பயன்பாடுகள்",
        "science": "அறிவியல் பெயர்",
        "other_names": "மற்ற பெயர்கள்",
        "cultivation": "இந்தியாவில் சாகுபடி",
        "new_prediction": "புதிய கணிப்பு",
    },
    "Hindi": {
        "dashboard": "डैशबोर्ड",
        "identify": "भोजन पहचानें",
        "assistant": "AI सहायक",
        "explore": "खाद्य पदार्थ देखें",
        "nutrition": "पोषण और आहार",
        "daily_intake": "दैनिक आहार मात्रा",
        "shop_foods": "खाद्य खरीदें",
        "budget_planner": "बजट योजना",
        "cultivation_map": "खेती मानचित्र",
        "compare": "तुलना",
        "history": "इतिहास",
        "about": "परिचय",
        "language": "Language / மொழி / भाषा",
        "upload": "चित्र अपलोड करें",
        "camera": "कैमरा उपयोग करें",
        "predict": "पूर्वानुमान",
        "confidence": "विश्वास",
        "category": "श्रेणी",
        "who": "कौन आहार में शामिल कर सकता है?",
        "diet": "आहार में कैसे शामिल करें",
        "uses": "उपयोग",
        "science": "वैज्ञानिक नाम",
        "other_names": "अन्य नाम",
        "cultivation": "भारत में खेती",
        "new_prediction": "नई पहचान",
    },
}

CATEGORY_EMOJI = {"Millets": "🌾", "Pulses": "🫘", "Grains": "🌿", "Nuts": "🥜"}

# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "language": "English",
        "page": "Dashboard",
        "prediction": None,
        "prediction_confidence": 0.0,
        "prediction_top3": [],
        "prediction_image": None,
        "history": [],
        "assistant": None,
        "assistant_messages": [],
        "last_report_food": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_class_names():
    if CLASS_NAMES_PATH.exists():
        names = [
            line.strip()
            for line in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(names) == 10:
            return names
    return SUPPORTED_CLASSES


MODEL = load_model()
CLASS_NAMES = load_class_names()


def predict_image(pil_image):
    if MODEL is None:
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, 0)

    probs = MODEL.predict(arr, verbose=0)[0]
    order = np.argsort(probs)[::-1]

    top3 = [(CLASS_NAMES[i], float(probs[i])) for i in order[:3]]
    food, conf = top3[0]

    # Practical uncertainty gate for demo use.
    # This is not a formal OOD detector. It is used to avoid presenting
    # a low-confidence prediction as certain.
    second_conf = top3[1][1] if len(top3) > 1 else 0.0
    uncertain = conf < 0.60 or (conf < 0.80 and (conf - second_conf) < 0.10)

    return food, conf, top3, uncertain


# ============================================================
# CSS / UI
# ============================================================
def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --dhanya-ink: #302a22;
            --dhanya-deep: #5b2b26;
            --dhanya-maroon: #7a3b32;
            --dhanya-leaf: #58733d;
            --dhanya-leaf-dark: #3f5b2d;
            --dhanya-grain: #c79a3b;
            --dhanya-brass: #b7832d;
            --dhanya-paper: #f6efdd;
            --dhanya-cream: #fffaf0;
            --dhanya-border: rgba(91, 43, 38, .12);
        }

        html, body, [class*="css"] {
            font-family: "Segoe UI", Arial, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 8%, rgba(199,154,59,.15), transparent 20%),
                radial-gradient(circle at 86% 12%, rgba(88,115,61,.12), transparent 22%),
                linear-gradient(180deg, #fffdf8 0%, #f6efdd 100%);
            color: var(--dhanya-ink);
        }

        /* Subtle hand-crafted heritage pattern */
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: .16;
            background-image:
                radial-gradient(circle at 12px 12px, transparent 0 7px, rgba(122,59,50,.18) 7px 8px, transparent 8px 18px),
                radial-gradient(circle at 6px 6px, rgba(199,154,59,.16) 0 2px, transparent 2px 10px);
            background-size: 36px 36px, 28px 28px;
            mask-image: linear-gradient(to bottom, black, transparent 82%);
            z-index: 0;
        }

        .stMainBlockContainer {
            position: relative;
            z-index: 1;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(250,246,233,.98), rgba(240,232,210,.98));
            border-right: 1px solid var(--dhanya-border);
        }


        /* Keep Dhanya AI navigation permanently visible */
        [data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
            transform: none !important;
            width: 335px !important;
            min-width: 335px !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            width: 335px !important;
        }

        /* Keep the native Streamlit sidebar collapse/expand arrow visible */
        [data-testid="collapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 999999 !important;
        }

        [data-testid="stSidebarCollapseButton"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 999999 !important;
        }

        /* Keep the main content comfortably sized beside the fixed sidebar */
        [data-testid="stAppViewContainer"] {
            overflow-x: hidden !important;
        }


        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.1rem;
        }

        .dhanya-brand {
            display:flex;
            align-items:center;
            gap:12px;
            padding:12px 12px 16px 12px;
            margin-bottom:10px;
            border-bottom:1px solid rgba(122,59,50,.10);
        }

        .dhanya-brand-mark {
            width:48px;
            height:48px;
            border-radius:50%;
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:27px;
            background:
                radial-gradient(circle, #fff8df 0 34%, #ecd28f 35% 58%, #a86f2c 59% 62%, #fff8df 63%);
            box-shadow:0 5px 16px rgba(90,55,26,.16);
            border:1px solid rgba(124,84,33,.25);
        }

        .dhanya-brand-name {
            font-family: Georgia, "Times New Roman", serif;
            font-size:1.42rem;
            font-weight:800;
            color:var(--dhanya-deep);
            letter-spacing:.3px;
        }

        .dhanya-brand-sub {
            font-size:.72rem;
            color:#7b6c61;
            margin-top:2px;
            letter-spacing:.2px;
        }

        .hero {
            position:relative;
            padding: 8px 0 12px 0;
            margin-bottom:6px;
        }

        .hero::after {
            content:"✦  ✦  ✦";
            display:block;
            color:var(--dhanya-grain);
            font-size:.76rem;
            letter-spacing:8px;
            margin-top:9px;
            opacity:.75;
        }

        .hero h1 {
            font-family: Georgia, "Times New Roman", serif;
            font-size:3rem;
            line-height:1.02;
            margin:0 0 10px 0;
            color:var(--dhanya-deep);
            letter-spacing:-.5px;
        }

        .hero p {
            font-size:1.05rem;
            color:#665c54;
            max-width:830px;
            line-height:1.6;
        }

        .pill {
            display:inline-block;
            padding:7px 12px;
            border-radius:999px;
            font-size:.84rem;
            font-weight:700;
            margin:4px 5px 0 0;
            background:#f7ecd1;
            color:var(--dhanya-deep);
            border:1px solid rgba(167,119,42,.18);
        }

        .glass-card {
            background:rgba(255,252,243,.92);
            border:1px solid rgba(91,43,38,.10);
            border-radius:20px;
            padding:20px;
            box-shadow:0 12px 30px rgba(72,52,28,.08);
            backdrop-filter: blur(7px);
            position:relative;
            overflow:hidden;
        }

        .glass-card::before {
            content:"";
            position:absolute;
            left:0;
            top:0;
            width:100%;
            height:4px;
            background:linear-gradient(90deg, var(--dhanya-maroon), var(--dhanya-grain), var(--dhanya-leaf));
        }

        .metric-card {
            background:rgba(255,252,244,.90);
            border:1px solid rgba(91,43,38,.10);
            border-radius:18px;
            padding:16px 18px;
            min-height:120px;
            box-shadow:0 8px 20px rgba(72,52,28,.07);
            transition:transform .18s ease, box-shadow .18s ease;
        }

        .metric-card:hover {
            transform:translateY(-3px);
            box-shadow:0 13px 28px rgba(72,52,28,.11);
        }

        .metric-label {
            color:#776b61;
            font-size:.9rem;
            margin-bottom:5px;
        }

        .metric-value {
            color:var(--dhanya-deep);
            font-size:1.55rem;
            font-weight:800;
        }

        .result-card {
            background:
                linear-gradient(135deg, rgba(255,254,248,.96), rgba(247,237,211,.93));
            border:1px solid rgba(88,115,61,.20);
            border-radius:24px;
            padding:20px;
            box-shadow:0 14px 36px rgba(67,59,26,.10);
            position:relative;
            overflow:hidden;
        }

        .result-card::after {
            content:"🌾  •  🫘  •  🌿";
            position:absolute;
            right:18px;
            top:12px;
            opacity:.42;
            font-size:1rem;
        }

        .result-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size:1.9rem;
            font-weight:800;
            color:var(--dhanya-deep);
            margin-bottom:4px;
        }

        .small-muted {
            font-size:.9rem;
            color:#7d7268;
        }

        .section-title {
            margin-top:22px;
            margin-bottom:11px;
            color:var(--dhanya-deep);
            font-family: Georgia, "Times New Roman", serif;
            font-size:1.35rem;
            font-weight:800;
        }

        .tag {
            display:inline-block;
            background:#edf3e5;
            color:var(--dhanya-leaf-dark);
            border:1px solid #d6e2c5;
            border-radius:999px;
            padding:6px 11px;
            margin:3px 5px 3px 0;
            font-size:.82rem;
            font-weight:700;
        }

        .warning-box {
            padding:14px 16px;
            border-radius:16px;
            background:#fff5df;
            border:1px solid #ecd29a;
            color:#6a5228;
        }

        /* Light traditional styling for native Streamlit widgets */
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] [data-baseweb="input"] > div,
        [data-testid="stTextInput"] [data-baseweb="input"] > div,
        [data-testid="stTextArea"] [data-baseweb="textarea"] > div {
            background:#fffdf7 !important;
            color:#302a22 !important;
            border-color:rgba(91,43,38,.16) !important;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] div {
            color:#302a22 !important;
        }

        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            background:#fffdf7 !important;
            color:#302a22 !important;
            -webkit-text-fill-color:#302a22 !important;
            caret-color:#5b2b26 !important;
        }

        [data-testid="stSlider"] [role="slider"] {
            background:#7a3b32 !important;
            border-color:#7a3b32 !important;
        }

        [data-baseweb="popover"] {
            background:#fffdf7 !important;
            color:#302a22 !important;
        }

        [data-baseweb="menu"] {
            background:#fffdf7 !important;
            color:#302a22 !important;
        }

        [data-baseweb="menu"] [role="option"] {
            color:#302a22 !important;
        }

        [data-baseweb="menu"] [role="option"]:hover,
        [data-baseweb="menu"] [aria-selected="true"] {
            background:#f2e4c4 !important;
            color:#5b2b26 !important;
        }

        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label {
            color:#5b2b26 !important;
        }

        /* Streamlit navigation/radio polish */
        [data-testid="stSidebar"] [role="radiogroup"] > label {
            background:rgba(255,252,244,.55);
            border:1px solid transparent;
            border-radius:12px;
            padding:7px 10px !important;
            margin:3px 0 !important;
            transition:all .16s ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
            background:#fff8e9;
            border-color:rgba(122,59,50,.10);
        }

        [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
            background:linear-gradient(90deg,#f3e4bb,#edf2e4);
            border-color:rgba(122,59,50,.16);
            box-shadow:inset 4px 0 0 var(--dhanya-maroon);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius:12px;
            border:1px solid rgba(122,59,50,.16);
            background:linear-gradient(180deg,#fffdf5,#f5e7c7);
            color:var(--dhanya-deep);
            font-weight:700;
            min-height:42px;
            transition:transform .16s ease, box-shadow .16s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform:translateY(-1px);
            box-shadow:0 8px 18px rgba(72,52,28,.10);
        }

        .stProgress > div > div > div {
            background:linear-gradient(90deg,var(--dhanya-maroon),var(--dhanya-grain),var(--dhanya-leaf));
        }

        .footer-note {
            margin-top:28px;
            padding:14px 0 8px 0;
            color:#8a8076;
            font-size:.82rem;
            text-align:center;
            border-top:1px solid rgba(91,43,38,.08);
        }

        .heritage-ribbon {
            margin:2px 0 18px 0;
            padding:10px 14px;
            border-top:1px solid rgba(122,59,50,.13);
            border-bottom:1px solid rgba(122,59,50,.13);
            text-align:center;
            color:#745f4b;
            font-family:Georgia, "Times New Roman", serif;
            font-size:.88rem;
            letter-spacing:.6px;
            background:rgba(255,249,232,.5);
        }

        .no-print { display:block; }


        /* Final polish: reduce blank space and make navigation cleaner */
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 1180px !important;
            padding-top: 1.35rem !important;
            padding-bottom: 1.6rem !important;
        }

        /* Keep Streamlit's native top toolbar/menu visible */
        [data-testid="stHeader"] {
            background: transparent !important;
        }

        [data-testid="stToolbar"] {
            visibility: visible !important;
            opacity: 1 !important;
            display: flex !important;
        }

        [data-testid="stStatusWidget"] {
            display: block !important;
        }

        .hero h1 a {
            display:none !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {
            display:none !important;
        }

        [data-testid="stSidebar"] [role="radiogroup"] > label {
            padding-left:12px !important;
            min-height:38px;
            display:flex !important;
            align-items:center !important;
            color:#5b2b26 !important;
        }

        /* Force sidebar navigation text to remain readable on the cream sidebar */
        [data-testid="stSidebar"] [role="radiogroup"] > label,
        [data-testid="stSidebar"] [role="radiogroup"] > label *,
        [data-testid="stSidebar"] [role="radiogroup"] > label p,
        [data-testid="stSidebar"] [role="radiogroup"] > label span {
            color:#5b2b26 !important;
        }

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] small {
            color:#5b2b26 !important;
        }

        .heritage-hero {
            display:grid;
            grid-template-columns: minmax(0,1.45fr) minmax(220px,.62fr);
            gap:18px;
            align-items:center;
            margin: 2px 0 12px 0;
        }

        .heritage-hero-copy {
            padding: 2px 2px 4px 0;
        }

        .eyebrow {
            display:inline-flex;
            align-items:center;
            gap:7px;
            padding:6px 11px;
            border-radius:999px;
            background:#f2e4c4;
            border:1px solid rgba(122,59,50,.12);
            color:#6c4b28;
            font-size:.78rem;
            font-weight:800;
            letter-spacing:.45px;
            text-transform:uppercase;
        }

        .heritage-hero-title {
            font-family: Georgia, "Times New Roman", serif;
            color:var(--dhanya-deep);
            font-size:2.55rem;
            line-height:1.02;
            margin:9px 0 6px;
            letter-spacing:-.55px;
        }

        .heritage-tagline {
            font-family: Georgia, "Times New Roman", serif;
            color:var(--dhanya-maroon);
            font-size:1.18rem;
            font-weight:700;
            line-height:1.25;
            margin:0 0 7px;
        }

        .heritage-tagline-en {
            color:#7c6c5d;
            font-size:.86rem;
            font-weight:600;
            margin:0 0 10px;
        }

        .heritage-hero-copy p {
            color:#665e57;
            font-size:.94rem;
            line-height:1.5;
            max-width:690px;
            margin:0;
        }

        .hero-orb {
            min-height:185px;
            border-radius:26px;
            border:1px solid rgba(122,59,50,.13);
            background:
                radial-gradient(circle at 50% 46%, #fff9e8 0 16%, transparent 16.5%),
                radial-gradient(circle at 50% 46%, transparent 0 27%, rgba(199,154,59,.40) 27.5% 28.4%, transparent 29%),
                radial-gradient(circle at 50% 46%, transparent 0 39%, rgba(88,115,61,.30) 39.5% 40.2%, transparent 41%),
                radial-gradient(circle at 50% 46%, transparent 0 52%, rgba(122,59,50,.18) 52.5% 53.1%, transparent 54%),
                linear-gradient(145deg,#fff9eb,#f1e5c3);
            box-shadow:0 14px 30px rgba(72,52,28,.08);
            display:flex;
            align-items:center;
            justify-content:center;
            position:relative;
            overflow:hidden;
        }

        .hero-orb-inner {
            width:112px;
            height:112px;
            border-radius:50%;
            display:flex;
            align-items:center;
            justify-content:center;
            background:linear-gradient(145deg,#f7e2aa,#dce7c7);
            border:2px solid rgba(122,59,50,.14);
            box-shadow:0 9px 20px rgba(72,52,28,.13);
            font-size:54px;
        }

        .hero-orb-label {
            position:absolute;
            bottom:14px;
            left:0;
            right:0;
            text-align:center;
            color:#745f4b;
            font-family:Georgia, "Times New Roman", serif;
            font-size:.88rem;
            letter-spacing:.65px;
        }

        .quick-actions {
            display:grid;
            grid-template-columns:repeat(3,1fr);
            gap:14px;
            margin: 16px 0 6px 0;
        }

        .quick-card {
            background:rgba(255,251,241,.90);
            border:1px solid rgba(91,43,38,.10);
            border-radius:18px;
            padding:16px 17px;
            box-shadow:0 8px 20px rgba(72,52,28,.06);
            transition:transform .16s ease, box-shadow .16s ease;
        }

        .quick-card:hover {
            transform:translateY(-2px);
            box-shadow:0 12px 26px rgba(72,52,28,.09);
        }

        .quick-card-icon {
            font-size:1.35rem;
            margin-bottom:7px;
        }

        .quick-card-title {
            color:var(--dhanya-deep);
            font-family:Georgia, "Times New Roman", serif;
            font-size:1.03rem;
            font-weight:800;
        }

        .quick-card-text {
            color:#746c64;
            font-size:.86rem;
            line-height:1.48;
            margin-top:5px;
        }



        .map-card {
            background:rgba(255,252,243,.94);
            border:1px solid rgba(91,43,38,.10);
            border-radius:20px;
            padding:16px;
            box-shadow:0 10px 26px rgba(72,52,28,.07);
        }

        .map-legend {
            display:flex;
            gap:10px;
            flex-wrap:wrap;
            margin:8px 0 4px;
        }

        .map-legend-item {
            display:inline-flex;
            align-items:center;
            gap:7px;
            color:#6f655d;
            font-size:.82rem;
            font-weight:700;
        }

        .map-dot {
            width:11px;
            height:11px;
            display:inline-block;
            border-radius:50%;
        }

        .shop-card {
            background:rgba(255,252,243,.92);
            border:1px solid rgba(91,43,38,.10);
            border-radius:18px;
            padding:17px;
            min-height:190px;
            box-shadow:0 8px 22px rgba(72,52,28,.06);
            transition:transform .16s ease, box-shadow .16s ease;
        }

        .shop-card:hover {
            transform:translateY(-2px);
            box-shadow:0 12px 28px rgba(72,52,28,.09);
        }

        .shop-card-title {
            color:var(--dhanya-deep);
            font-family:Georgia, "Times New Roman", serif;
            font-size:1.08rem;
            font-weight:800;
        }

        .shop-card-meta {
            color:#7b7067;
            font-size:.82rem;
            margin:3px 0 8px;
        }

        .section-rule {
            height:1px;
            margin:18px 0 2px 0;
            background:linear-gradient(90deg, transparent, rgba(122,59,50,.18), rgba(199,154,59,.35), transparent);
        }

        @media (max-width: 900px) {
            .heritage-hero { grid-template-columns:1fr; }
            .hero-orb { min-height:165px; }
            .quick-actions { grid-template-columns:1fr; }
            .heritage-hero-title { font-size:2.2rem; }
            .heritage-tagline { font-size:1.04rem; }
        }


        /* =====================================================
           FINAL LIGHT UI OVERRIDES
           Purpose: keep every native Streamlit control readable
           on the Dhanya cream / parchment theme.
           This block changes presentation only; model and page
           logic remain unchanged.
           ===================================================== */
        :root {
            color-scheme: light !important;
        }

        html, body {
            background:#f6efdd !important;
            color:#302a22 !important;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            color:#302a22 !important;
        }

        [data-testid="stAppViewContainer"] .main,
        [data-testid="stAppViewContainer"] .main * {
            color-scheme:light !important;
        }

        /* Typography */
        .stMarkdown,
        .stMarkdown p,
        .stMarkdown li,
        .stCaption,
        .stText,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] * {
            color:#302a22 !important;
        }

        .stMarkdown h1,
        .stMarkdown h2,
        .stMarkdown h3,
        .stMarkdown h4,
        .stMarkdown h5,
        .stMarkdown h6 {
            color:#5b2b26 !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] * {
            color:#5b2b26 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="select"] span {
            background:#fffdf7 !important;
            color:#302a22 !important;
        }

        /* Selectbox / multiselect / number input / text input / textarea */
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] [data-baseweb="input"] > div,
        [data-testid="stTextInput"] [data-baseweb="input"] > div,
        [data-testid="stTextArea"] [data-baseweb="textarea"] > div,
        [data-testid="stDateInput"] [data-baseweb="input"] > div,
        [data-testid="stTimeInput"] [data-baseweb="input"] > div {
            background:#fffdf7 !important;
            color:#302a22 !important;
            border:1px solid rgba(91,43,38,.16) !important;
            box-shadow:0 2px 8px rgba(72,52,28,.035) !important;
        }

        [data-testid="stSelectbox"] input,
        [data-testid="stMultiSelect"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stDateInput"] input,
        [data-testid="stTimeInput"] input {
            color:#302a22 !important;
            -webkit-text-fill-color:#302a22 !important;
            caret-color:#7a3b32 !important;
            background:#fffdf7 !important;
        }

        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {
            color:#95877b !important;
            opacity:1 !important;
        }

        /* Dropdown popovers */
        [data-baseweb="popover"],
        [data-baseweb="menu"],
        [data-baseweb="menu"] ul {
            background:#fffdf7 !important;
            color:#302a22 !important;
            border-color:rgba(91,43,38,.12) !important;
        }

        [data-baseweb="menu"] [role="option"] {
            background:#fffdf7 !important;
            color:#302a22 !important;
        }

        [data-baseweb="menu"] [role="option"]:hover,
        [data-baseweb="menu"] [role="option"][aria-selected="true"] {
            background:#f3e7c9 !important;
            color:#5b2b26 !important;
        }

        /* Multiselect chips */
        [data-testid="stMultiSelect"] [data-baseweb="tag"] {
            background:#edf3e5 !important;
            color:#3f5b2d !important;
            border:1px solid #d6e2c5 !important;
        }

        [data-testid="stMultiSelect"] [data-baseweb="tag"] span,
        [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
            color:#3f5b2d !important;
            fill:#3f5b2d !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] section {
            background:#fffdf7 !important;
            border:1.5px dashed rgba(122,59,50,.25) !important;
            border-radius:18px !important;
        }

        [data-testid="stFileUploader"] section:hover {
            border-color:rgba(122,59,50,.45) !important;
            background:#fffaf0 !important;
        }

        [data-testid="stFileUploader"] button {
            background:#f4e7c8 !important;
            color:#5b2b26 !important;
            border:1px solid rgba(122,59,50,.16) !important;
        }

        [data-testid="stFileUploader"] button * {
            color:#5b2b26 !important;
        }

        /* Buttons */
        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] button {
            background:linear-gradient(180deg,#fffdf7 0%,#f1e3c3 100%) !important;
            color:#5b2b26 !important;
            -webkit-text-fill-color:#5b2b26 !important;
            border:1px solid rgba(122,59,50,.18) !important;
            box-shadow:0 4px 12px rgba(72,52,28,.06) !important;
        }

        .stButton > button *,
        .stDownloadButton > button *,
        [data-testid="stFormSubmitButton"] button * {
            color:#5b2b26 !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            background:#f5e8ca !important;
            border-color:rgba(122,59,50,.28) !important;
        }

        /* Alerts / notices */
        [data-testid="stAlert"] {
            border-radius:14px !important;
            border:1px solid rgba(91,43,38,.11) !important;
            box-shadow:0 5px 14px rgba(72,52,28,.045) !important;
        }

        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span {
            color:#463d35 !important;
        }

        /* Expanders */
        [data-testid="stExpander"] {
            background:#fffdf7 !important;
            border:1px solid rgba(91,43,38,.10) !important;
            border-radius:16px !important;
        }

        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
            color:#5b2b26 !important;
        }

        /* Tabs */
        [data-testid="stTabs"] [role="tab"] {
            color:#6b5b50 !important;
        }

        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color:#5b2b26 !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background:#7a3b32 !important;
        }

        /* Radio / checkbox / toggle */
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label,
        [data-testid="stToggle"] label {
            color:#4d423a !important;
        }

        [data-testid="stRadio"] label span,
        [data-testid="stCheckbox"] label span,
        [data-testid="stToggle"] label span {
            color:#4d423a !important;
        }

        /* Slider */
        [data-testid="stSlider"] [data-baseweb="slider"] {
            color:#5b2b26 !important;
        }

        [data-testid="stSlider"] [role="slider"] {
            background:#7a3b32 !important;
            border-color:#7a3b32 !important;
        }

        /* DataFrames / tables: use light host panel and readable surrounding labels */
        [data-testid="stDataFrame"],
        [data-testid="stDataEditor"] {
            background:#fffdf7 !important;
            border:1px solid rgba(91,43,38,.10) !important;
            border-radius:14px !important;
            overflow:hidden !important;
        }

        /* Images */
        [data-testid="stImage"] img {
            border-radius:18px !important;
            border:1px solid rgba(91,43,38,.10) !important;
            box-shadow:0 7px 20px rgba(72,52,28,.07) !important;
        }

        /* Code / JSON viewers */
        [data-testid="stCodeBlock"],
        [data-testid="stJson"] {
            border-radius:14px !important;
        }

        /* Links */
        a, a:visited {
            color:#7a3b32 !important;
        }

        a:hover {
            color:#5b2b26 !important;
        }

        /* Remove accidental dark inline form controls */
        input, textarea, select, button {
            color-scheme:light !important;
        }

        @media print {
            @page { size: A4; margin: 14mm; }

            [data-testid="stSidebar"],
            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            .stDeployButton,
            .no-print {
                display:none !important;
            }

            .stApp {
                background:white !important;
            }

            .glass-card,
            .metric-card,
            .result-card {
                box-shadow:none !important;
                background:white !important;
            }

            .dhanya-brand, .heritage-ribbon {
                display:none !important;
            }

            .main .block-container {
                max-width:100% !important;
                padding:0 !important;
            }

            * {
                -webkit-print-color-adjust: exact !important;
                print-color-adjust: exact !important;
            }
        }

        @media (max-width: 900px) {
            .hero h1 { font-size:2.35rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()


# ============================================================
# HELPERS
# ============================================================
def t(key):
    return TRANSLATIONS[st.session_state.language].get(key, key)


def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()


def format_pct(value):
    return f"{value * 100:.2f}%"


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="dhanya-brand">
              <div class="dhanya-brand-mark">🌾</div>
              <div>
                <div class="dhanya-brand-name">Dhanya AI</div>
                <div class="dhanya-brand-sub">Grains • Pulses • Millets</div>
              </div>
            </div>
            <div class="heritage-ribbon">உணவு • அறிவு • பாரம்பரியம்</div>
            """,
            unsafe_allow_html=True,
        )

        st.session_state.language = st.selectbox(
            t("language"),
            ["English", "Tamil", "Hindi"],
            index=["English", "Tamil", "Hindi"].index(st.session_state.language),
        )

        current_date = dt.datetime.now().strftime("%d %B %Y")
        st.markdown(
            f"""
            <div style="
                margin:10px 0 14px 0;
                padding:9px 12px;
                border:1px solid rgba(122,59,50,.10);
                border-radius:12px;
                background:rgba(255,250,239,.72);
                color:#6c5b4e;
                font-size:.84rem;
                font-weight:700;">
                📅 Today&nbsp;&nbsp; {html.escape(current_date)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        page_options = [
            ("🏠", t("dashboard"), "Dashboard"),
            ("🔎", t("identify"), "Identify Food"),
            ("🤖", t("assistant"), "AI Assistant"),
            ("🌾", t("explore"), "Explore Foods"),
            ("🛒", t("shop_foods"), "Shop Foods"),
            ("💰", t("budget_planner"), "Budget Planner"),
            ("🗺️", t("cultivation_map"), "Cultivation Map"),
            ("🥗", t("nutrition"), "Nutrition & Diet"),
            ("📅", t("daily_intake"), "Daily Intake"),
            ("⚖️", t("compare"), "Compare"),
            ("📜", t("history"), "History"),
            ("ℹ️", t("about"), "About"),
        ]

        labels = [f"{icon} {label}" for icon, label, _ in page_options]
        current_index = next(
            (i for i, (_, _, key) in enumerate(page_options) if key == st.session_state.page),
            0,
        )

        selected = st.radio(
            "Navigation",
            labels,
            index=current_index,
            label_visibility="collapsed",
        )
        selected_key = page_options[labels.index(selected)][2]
        if selected_key != st.session_state.page:
            st.session_state.page = selected_key
            st.rerun()

        st.markdown("---")
        st.caption("Custom CNN • 10K Training Pipeline")
        st.caption("Food identification + nutrition intelligence")


def render_page_header(title, subtitle=None):
    st.markdown(f'<div class="hero"><h1>{html.escape(title)}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p>{html.escape(subtitle)}</p></div>", unsafe_allow_html=True)
    else:
        st.markdown("</div>", unsafe_allow_html=True)


def render_voice_button(food_name, confidence, language):
    lang_code = {
        "English": "en-IN",
        "Tamil": "ta-IN",
        "Hindi": "hi-IN",
    }.get(language, "en-IN")

    if language == "Tamil":
        text = f"கணிக்கப்பட்ட உணவு {food_name}. நம்பிக்கை {confidence:.1%}."
    elif language == "Hindi":
        text = f"पहचाना गया भोजन {food_name}. विश्वास {confidence:.1%}."
    else:
        text = f"Predicted class is {food_name}. Confidence is {confidence:.1%}."

    payload = json.dumps(text)
    safe_lang = json.dumps(lang_code)

    components.html(
        f"""
        <script>
        function dhanyaSpeak() {{
            const text = {payload};
            const lang = {safe_lang};
            if (!window.parent.speechSynthesis) {{
                alert("Speech synthesis is not available in this browser.");
                return;
            }}
            window.parent.speechSynthesis.cancel();
            const utter = new window.parent.SpeechSynthesisUtterance(text);
            utter.lang = lang;
            utter.rate = 0.95;
            window.parent.speechSynthesis.speak(utter);
        }}
        </script>
        <button onclick="dhanyaSpeak()"
            style="padding:10px 14px;border-radius:12px;border:1px solid #d7e6c5;
                   background:#edf4e4;color:#4e6e30;font-weight:700;cursor:pointer;">
            🔊 Voice
        </button>
        """,
        height=55,
    )


def render_print_button():
    components.html(
        """
        <script>
        function dhanyaPrint() {
            try {
                window.parent.focus();
                window.parent.print();
            } catch (e) {
                window.print();
            }
        }
        </script>
        <button onclick="dhanyaPrint()"
            style="padding:10px 14px;border-radius:12px;border:1px solid #eadcb5;
                   background:#fffaf0;color:#6b5120;font-weight:700;cursor:pointer;">
            🖨️ Print / Save as PDF
        </button>
        """,
        height=55,
    )


def add_history(food, confidence, top3):
    record = {
        "time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "food": food,
        "category": CATEGORY_MAP.get(food, "Food"),
        "confidence": float(confidence),
        "top3": [(n, float(c)) for n, c in top3],
    }
    st.session_state.history.insert(0, record)
    st.session_state.history = st.session_state.history[:50]


def render_metric(label, value, emoji=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{emoji} {html.escape(label)}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_food_reference(food):
    if get_nutrition is None:
        return None
    try:
        return get_nutrition(food)
    except Exception:
        return None


def nutrition_value(food, grams):
    if nutrition_for_grams is None:
        return None
    try:
        return nutrition_for_grams(food, grams)
    except Exception:
        return None


def render_nutrition_section(food_name, compact=False):
    ref = get_food_reference(food_name)

    st.markdown(
        '<div class="section-title">🥗 Nutrition & Serving</div>',
        unsafe_allow_html=True,
    )

    grams = st.slider(
        "Select serving size (g)",
        min_value=25,
        max_value=250,
        value=100,
        step=25,
        key=f"serving_{food_name}_{'compact' if compact else 'full'}",
    )

    vals = nutrition_value(food_name, grams)

    if vals is None:
        st.info(
            "Nutrition reference data is not available for this class yet. "
            "The identification result is still available."
        )
        return

    # Support dict-like objects from the existing nutrition_data.py module.
    def val(*keys, default=None):
        for key in keys:
            if isinstance(vals, dict) and key in vals:
                return vals[key]
            if hasattr(vals, key):
                return getattr(vals, key)
        return default

    energy = val("energy_kcal", "calories_kcal", "kcal")
    protein = val("protein_g", "protein")
    carbs = val("carbs_g", "carbohydrates_g", "carbohydrates")
    fibre = val("fibre_g", "fiber_g", "fibre")
    fat = val("fat_g", "fat")
    calcium = val("calcium_mg", "calcium")
    iron = val("iron_mg", "iron")
    magnesium = val("magnesium_mg", "magnesium")
    phosphorus = val("phosphorus_mg", "phosphorus")
    potassium = val("potassium_mg", "potassium")

    cols = st.columns(4)
    metric_items = [
        ("Energy", f"{energy:.1f} kcal" if energy is not None else "—", "🔥"),
        ("Protein", f"{protein:.2f} g" if protein is not None else "—", "💪"),
        ("Carbohydrates", f"{carbs:.2f} g" if carbs is not None else "—", "🌾"),
        ("Fibre", f"{fibre:.2f} g" if fibre is not None else "—", "🌿"),
    ]
    for col, item in zip(cols, metric_items):
        with col:
            render_metric(*item)

    st.markdown(
        '<div class="section-title">Mineral Profile</div>',
        unsafe_allow_html=True,
    )
    mcols = st.columns(5)
    minerals = [
        ("Calcium", calcium),
        ("Iron", iron),
        ("Magnesium", magnesium),
        ("Phosphorus", phosphorus),
        ("Potassium", potassium),
    ]
    for col, (name, number) in zip(mcols, minerals):
        with col:
            render_metric(name, f"{number:.2f} mg" if number is not None else "—")

    if not compact:
        highlights = []
        if protein is not None and protein >= 15:
            highlights.append("High plant-protein contribution")
        if fibre is not None and fibre >= 8:
            highlights.append("Provides dietary fibre")
        mineral_names = []
        if iron is not None and iron >= 3:
            mineral_names.append("iron")
        if magnesium is not None and magnesium >= 70:
            mineral_names.append("magnesium")
        if phosphorus is not None and phosphorus >= 150:
            mineral_names.append("phosphorus")
        if mineral_names:
            highlights.append("Provides " + ", ".join(mineral_names))

        if highlights:
            st.markdown(
                '<div class="section-title">✨ Nutrition Highlights</div>',
                unsafe_allow_html=True,
            )
            for h in highlights:
                st.markdown(f"• {html.escape(h)}")

        if ref is not None:
            source = None
            if isinstance(ref, dict):
                source = ref.get("source") or ref.get("reference") or ref.get("basis")
            else:
                source = getattr(ref, "source", None) or getattr(ref, "reference", None)

            if source:
                st.caption(f"Reference: {source}")
            st.caption(
                "Values are reference estimates on the app's stated basis and may vary "
                "with variety, processing, moisture, and preparation."
            )

        if format_nutrition_report is not None:
            try:
                report = format_nutrition_report(food_name, grams)
                st.download_button(
                    "⬇️ Download Nutrition Report",
                    data=report,
                    file_name=f"{food_name.replace(' ', '_')}_{grams}g_nutrition.txt",
                    mime="text/plain",
                )
            except Exception:
                pass

        st.markdown('<div class="no-print">', unsafe_allow_html=True)
        render_print_button()
        st.markdown("</div>", unsafe_allow_html=True)


def render_food_details(food):
    category = CATEGORY_MAP.get(food, "Food")

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="result-title">{html.escape(food)}</div>
            <div class="small-muted">{CATEGORY_EMOJI.get(category, '🍽️')} {html.escape(category)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{t('science')}**")
        st.write(SCIENTIFIC_NAMES.get(food, "Reference not available"))

        st.markdown(f"**{t('other_names')}**")
        st.write(ALTERNATE_NAMES.get(food, "—"))

    with c2:
        st.markdown(f"**{t('uses')}**")
        st.write(USES_MAP.get(food, "—"))

        st.markdown(f"**{t('cultivation')}**")
        st.write(CULTIVATION_INFO.get(food, "Reference not available"))


    st.markdown('<div class="section-title">When to Include It</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="glass-card">
            <div style="font-weight:800;color:#5b2b26;">🍽️ Suggested meal window</div>
            <div style="font-size:1.05rem;margin-top:4px;color:#4f5b50;">
                {html.escape(MEAL_WINDOW_GUIDANCE.get(food, "Any suitable meal window"))}
            </div>
            <div class="small-muted" style="margin-top:6px;">
                This is a general meal-planning suggestion, not a universal medical or nutritional rule.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    storage = STORAGE_REFERENCE.get(
        food,
        {"window": "Use the package best-before date.", "note": "Store in a cool, dry, airtight place."},
    )
    with st.expander("🗓️ Storage & Best-Before Guide", expanded=False):
        st.markdown(f"**Reference window:** {html.escape(storage['window'])}")
        st.write(storage["note"])

        purchase_key = f"purchase_{food.replace(' ', '_')}"
        best_before_key = f"best_before_{food.replace(' ', '_')}"

        purchase_date = st.date_input(
            "Purchase date",
            value=dt.date.today(),
            key=purchase_key,
        )
        best_before_date = st.date_input(
            "Best-before / use-by date from package",
            value=purchase_date + dt.timedelta(days=30),
            min_value=purchase_date,
            key=best_before_key,
        )
        days_left = (best_before_date - dt.date.today()).days

        if days_left > 0:
            st.success(f"Approx. {days_left} days until the package date.")
        elif days_left == 0:
            st.warning("The package date is today. Check the label and product condition before use.")
        else:
            st.error("The package date has passed. Do not treat the app as overriding the manufacturer's date.")

        st.caption(
            "FSSAI guidance: shelf life varies with food type, packaging and storage conditions; "
            "check the package best-before/use-by date and store cereals/pulses in clean, dry, preferably airtight containers."
        )

    st.markdown(f"**{t('diet')}**")
    ideas = DIET_IDEAS.get(food, [])
    if ideas:
        st.markdown(
            "".join([f'<span class="tag">{html.escape(x)}</span>' for x in ideas]),
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="warning-box" style="margin-top:14px;">
        Food information here is educational. It is not a diagnosis or a substitute for
        advice from a qualified healthcare professional, especially for allergies,
        medical conditions, pregnancy, or special diets.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_result():
    if not st.session_state.prediction:
        return

    food = st.session_state.prediction
    conf = st.session_state.prediction_confidence
    top3 = st.session_state.prediction_top3
    image = st.session_state.prediction_image
    category = CATEGORY_MAP.get(food, "Food")

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    left, right = st.columns([1, 1.35])

    with left:
        if image is not None:
            st.image(image, use_container_width=True)
        else:
            st.info("Prediction image not available.")

    with right:
        st.markdown(f'<div class="result-title">Predicted: {html.escape(food)}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="pill">{CATEGORY_EMOJI.get(category, "🍽️")} {html.escape(category)}</span>'
            f'<span class="pill">🎯 {conf:.2%} confidence</span>',
            unsafe_allow_html=True,
        )

        st.progress(min(max(conf, 0.0), 1.0), text=f"Confidence: {conf:.2%}")

        st.markdown("### Top 3 model outputs")
        table = pd.DataFrame(
            {
                "Class": [x[0] for x in top3],
                "Confidence": [f"{x[1] * 100:.2f}%" for x in top3],
            }
        )
        st.dataframe(table, use_container_width=True, hide_index=True)

        render_voice_button(food, conf, st.session_state.language)

    st.markdown("</div>", unsafe_allow_html=True)

    render_food_details(food)
    render_nutrition_section(food)


# ============================================================
# PAGES
# ============================================================
def page_dashboard():
    left, right = st.columns([1.45, 0.75], gap="large")

    with left:
        st.markdown(
            '<div class="eyebrow">🌾 Traditional Food • Modern AI</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="heritage-hero-title">Dhanya AI</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="heritage-tagline">மண்ணின் மரபு, உணவின் அறிவு.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="heritage-tagline-en">From Our Roots to Your Plate.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="
                color:#665e57;
                font-size:.94rem;
                line-height:1.5;
                max-width:690px;
                margin:0 0 10px 0;">
                Identify Indian millets, pulses and grains, explore food intelligence,
                reference nutrition, compare foods, and ask the AI assistant.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="margin-top:8px;">
                <span class="pill">Custom CNN</span>
                <span class="pill">10-class vision model</span>
                <span class="pill">10K training images</span>
                <span class="pill">English • Tamil • Hindi</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if HERO_IMAGE_PATH.exists():
            st.image(
                str(HERO_IMAGE_PATH),
                use_container_width=True,
            )
        else:
            st.info(
                "Add dhanya_hero_traditional.png to the project folder "
                "to display the traditional hero image."
            )

    st.markdown(
        '<div class="section-rule"></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("Supported classes", "10", "🌾")
    with c2:
        render_metric("Model", "Custom CNN", "🧠")
    with c3:
        render_metric("Training set", "10,000", "📊")
    with c4:
        render_metric("Languages", "3", "🗣️")


    st.markdown('<div class="section-title">Explore by Category</div>', unsafe_allow_html=True)

    category_layout = [
        (
            "🌾",
            "Millets",
            ["Barnyard millet", "Finger millet", "pearl millet"],
            "Traditional millet grains for porridge, dosa-style and roti preparations.",
        ),
        (
            "🫘",
            "Pulses",
            ["Chick Peas", "green gram", "peas"],
            "Protein-rich pulse and legume classes in the current model.",
        ),
        (
            "🌿",
            "Grains",
            ["Bamboo rice", "Oats", "Rice", "Wheat"],
            "Major cereal and grain classes supported by Dhanya AI.",
        ),
    ]

    cat_cols = st.columns(3)
    for col, (icon, name, classes, desc) in zip(cat_cols, category_layout):
        with col:
            class_tags = "".join(
                f'<span class="tag">{html.escape(x)}</span>' for x in classes
            )
            st.markdown(
                f"""
                <div class="glass-card" style="min-height:190px;">
                    <div style="font-size:1.55rem;">{icon}</div>
                    <div style="font-size:1.18rem;font-family:Georgia,serif;font-weight:800;color:#5b2b26;margin-top:6px;">
                        {name}
                    </div>
                    <div style="color:#746c64;font-size:.87rem;line-height:1.45;margin:6px 0 9px;">
                        {desc}
                    </div>
                    {class_tags}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Explore Dhanya AI</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="quick-actions">
            <div class="quick-card">
                <div class="quick-card-icon">🔎</div>
                <div class="quick-card-title">Identify Food</div>
                <div class="quick-card-text">Upload or capture one food sample and see class, category, confidence and top predictions.</div>
            </div>
            <div class="quick-card">
                <div class="quick-card-icon">🥗</div>
                <div class="quick-card-title">Nutrition & Diet</div>
                <div class="quick-card-text">Explore reference calories, protein, carbohydrates, fibre and minerals by serving size.</div>
            </div>
            <div class="quick-card">
                <div class="quick-card-icon">🤖</div>
                <div class="quick-card-title">AI Assistant</div>
                <div class="quick-card-text">Ask food and recipe questions using the detected food as context.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Current session</div>', unsafe_allow_html=True)
    if st.session_state.prediction:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="small-muted">Latest identification</div>
                <div class="result-title">{html.escape(st.session_state.prediction)}</div>
                <span class="pill">{CATEGORY_EMOJI.get(CATEGORY_MAP.get(st.session_state.prediction, 'Food'), '🍽️')} {html.escape(CATEGORY_MAP.get(st.session_state.prediction, 'Food'))}</span>
                <span class="pill">🎯 {st.session_state.prediction_confidence:.2%} confidence</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔎 Open latest prediction", use_container_width=True):
            set_page("Identify Food")
    else:
        st.info("No prediction yet. Open Identify Food to start.")

def page_identify():
    render_page_header(
        t("identify"),
        "Use a clean, single-food image for the best result.",
    )

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True,
        )
        source = st.radio(
            "Input source",
            [t("upload"), t("camera")],
            horizontal=True,
        )

        image_data = None
        if source == t("upload"):
            image_data = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )
        else:
            image_data = st.camera_input("Capture a food image")

        st.markdown("</div>", unsafe_allow_html=True)

        if image_data is not None:
            try:
                preview = Image.open(image_data).convert("RGB")
                st.session_state.pending_image = preview
            except Exception:
                st.error("Could not read the image.")
        else:
            st.session_state.pending_image = None

        if getattr(st.session_state, "pending_image", None) is not None:
            if st.button(f"🔮 {t('predict')}", use_container_width=True):
                try:
                    with st.spinner("Running Dhanya AI custom CNN..."):
                        food, conf, top3, uncertain = predict_image(
                            st.session_state.pending_image
                        )

                    st.session_state.prediction_image = st.session_state.pending_image.copy()

                    if uncertain:
                        st.session_state.prediction = None
                        st.session_state.prediction_confidence = conf
                        st.session_state.prediction_top3 = top3
                        st.warning(
                            f"Low-confidence or ambiguous sample. The model's top class was "
                            f"'{food}' at {conf:.2%}. Please use a clearer single-food image. "
                            f"This gate is a practical uncertainty check, not a formal "
                            f"out-of-distribution detector."
                        )
                    else:
                        st.session_state.prediction = food
                        st.session_state.prediction_confidence = conf
                        st.session_state.prediction_top3 = top3
                        add_history(food, conf, top3)
                        st.session_state.last_report_food = food
                        st.success(f"Prediction complete: {food}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    with right:
        st.markdown(
            """
            <div class="glass-card">
              <div style="font-size:1.08rem;font-weight:800;">📷 Better input quality</div>
              <p style="color:#68767a;">
                Use a well-lit image with one main food sample. Avoid severe blur,
                heavy occlusion, very small objects, or multiple different food classes
                in the same frame.
              </p>
              <div>
                <span class="tag">Good lighting</span>
                <span class="tag">Single sample</span>
                <span class="tag">Clear focus</span>
                <span class="tag">Simple background</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="warning-box" style="margin-top:15px;">
              The current classifier is trained on the 10 supported classes.
              An unfamiliar food can still receive a class score; the confidence gate
              reduces this risk but does not provide formal open-set recognition.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    render_prediction_result()


def page_assistant():
    render_page_header(
        t("assistant"),
        "Ask about the detected food, recipes, serving ideas, and general nutrition.",
    )

    if create_chat is None or ask_assistant is None:
        st.error(
            "AI assistant module is not available. Make sure dhanya_ai_assistant.py "
            "and the Gemini package are installed."
        )
        return

    if st.session_state.assistant is None:
        try:
            st.session_state.assistant = create_chat()
        except Exception as e:
            st.error(f"Could not initialize the assistant: {e}")
            return

    if st.session_state.prediction:
        st.info(
            f"Detected food context: **{st.session_state.prediction}** • "
            f"{st.session_state.prediction_confidence:.2%} confidence • "
            f"{CATEGORY_MAP.get(st.session_state.prediction, 'Food')}"
        )
    else:
        st.caption("No detected food yet. The assistant can still answer general food questions.")

    for role, message in st.session_state.assistant_messages:
        with st.chat_message(role):
            st.markdown(message)

    prompt = st.chat_input("Ask Dhanya AI...")
    if prompt:
        st.session_state.assistant_messages.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        food = st.session_state.prediction
        conf = st.session_state.prediction_confidence

        with st.chat_message("assistant"):
            try:
                answer = ask_assistant(
                    st.session_state.assistant,
                    prompt,
                    detected_food=food,
                    confidence=conf,
                    category=CATEGORY_MAP.get(food) if food else None,
                )
                st.markdown(answer)
                st.session_state.assistant_messages.append(("assistant", answer))
            except TypeError:
                # Compatibility fallback for a simpler assistant signature.
                try:
                    answer = ask_assistant(st.session_state.assistant, prompt)
                    st.markdown(answer)
                    st.session_state.assistant_messages.append(("assistant", answer))
                except Exception as e:
                    st.error(f"Assistant request failed: {e}")
            except Exception as e:
                st.error(f"Assistant request failed: {e}")

    st.markdown(
        """
        <div class="warning-box" style="margin-top:16px;">
          AI-generated responses are informational. They should not be used to diagnose
          or treat disease. Allergies, medical conditions, pregnancy, or special diets
          need professional guidance.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_explore():
    render_page_header(
        t("explore"),
        "Browse supported foods and their scientific, culinary, and cultivation context.",
    )

    categories = ["All", "Millets", "Pulses", "Grains"]
    selected_category = st.selectbox("Filter by category", categories)

    names = [
        f for f in CLASS_NAMES
        if selected_category == "All" or CATEGORY_MAP.get(f) == selected_category
    ]

    cols = st.columns(3)
    for idx, food in enumerate(names):
        with cols[idx % 3]:
            category = CATEGORY_MAP.get(food, "Food")
            st.markdown(
                f"""
                <div class="glass-card" style="min-height:165px;">
                    <div style="font-size:1.3rem;">{CATEGORY_EMOJI.get(category, '🍽️')}</div>
                    <div style="font-size:1.05rem;font-weight:800;margin-top:6px;">
                        {html.escape(food)}
                    </div>
                    <div class="small-muted">{html.escape(category)}</div>
                    <p style="color:#68767a;">{html.escape(USES_MAP.get(food, ''))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Learn more", key=f"learn_{food}"):
                st.session_state.explore_food = food
                st.rerun()

    if getattr(st.session_state, "explore_food", None):
        st.markdown("---")
        st.markdown(
            f"### {CATEGORY_EMOJI.get(CATEGORY_MAP.get(st.session_state.explore_food, 'Food'), '🍽️')} "
            f"{st.session_state.explore_food}"
        )
        render_food_details(st.session_state.explore_food)
        if st.button("Close details"):
            st.session_state.explore_food = None
            st.rerun()


def page_budget_planner():
    render_page_header(
        t("budget_planner"),
        "Plan how much of a selected food you can buy within a weekly or monthly budget.",
    )

    st.markdown(
        """
        <div class="glass-card">
            <div style="font-size:1.08rem;font-weight:800;color:#5b2b26;">
                💰 Budget-to-Quantity Planner
            </div>
            <p style="color:#6d645c;line-height:1.55;margin-bottom:6px;">
                Enter the current local price for the food you want to buy.
                Dhanya AI then converts your budget into an estimated quantity.
            </p>
            <div class="small-muted">
                Prices vary by brand, pack size, seller and location, so the app does not hardcode a fake “current price”.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])

    with left:
        food = st.selectbox(
            "Food",
            CLASS_NAMES,
            index=CLASS_NAMES.index(st.session_state.prediction)
            if st.session_state.prediction in CLASS_NAMES else 0,
            key="budget_food",
        )

        plan_type = st.radio(
            "Plan",
            ["Weekly", "Monthly"],
            horizontal=True,
            key="budget_plan_type",
        )

        preset = st.selectbox(
            "Budget",
            ["₹100", "₹250", "₹500", "₹750", "₹1,000", "₹2,000", "Custom"],
            index=2,
            key="budget_preset",
        )

        if preset == "Custom":
            budget = st.number_input(
                "Custom budget (₹)",
                min_value=1.0,
                value=500.0,
                step=50.0,
                key="budget_custom",
            )
        else:
            budget = float(preset.replace("₹", "").replace(",", ""))

        price_per_kg = st.number_input(
            "Current price (₹ per kg)",
            min_value=0.01,
            value=100.0,
            step=5.0,
            help="Use the current price you see on your local shop or marketplace listing.",
            key="budget_price_per_kg",
        )

    with right:
        quantity_kg = budget / price_per_kg
        quantity_g = quantity_kg * 1000
        packs_500g = int(quantity_g // 500)
        leftover_g = quantity_g - packs_500g * 500

        st.markdown(
            f"""
            <div class="result-card">
                <div class="small-muted">{html.escape(plan_type)} plan</div>
                <div class="result-title">{html.escape(food)}</div>
                <span class="pill">💰 Budget ₹{budget:,.0f}</span>
                <span class="pill">₹{price_per_kg:,.0f}/kg</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric("Estimated quantity", f"{quantity_kg:.2f} kg", "⚖️")
        with c2:
            render_metric("Approx. grams", f"{quantity_g:.0f} g", "🌾")
        with c3:
            render_metric("500g packs*", f"{packs_500g}", "📦")

        if leftover_g > 0:
            st.caption(f"*Equivalent to about {leftover_g:.0f} g remaining after whole 500 g packs.")

    st.markdown('<div class="section-title">Low → High Budget Examples</div>', unsafe_allow_html=True)
    preset_budgets = [100, 250, 500, 750, 1000, 2000]
    example_rows = []
    for amount in preset_budgets:
        kg = amount / price_per_kg
        example_rows.append(
            {
                "Budget": f"₹{amount:,}",
                "Estimated quantity": f"{kg:.2f} kg",
                "Approx. 500g packs": int((kg * 1000) // 500),
            }
        )
    st.dataframe(
        pd.DataFrame(example_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="warning-box" style="margin-top:14px;">
          This is a budgeting calculator, not a recommended daily intake.
          Use the current selling price for your preferred brand/pack size.
          Monthly plans are direct monthly budgets; they are not intended to replace
          the Daily Intake guidance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Quick shopping</div>',
        unsafe_allow_html=True,
    )
    st.button(
        "🛒 Open Shop Foods",
        use_container_width=True,
        on_click=set_page,
        args=("Shop Foods",),
    )


def page_nutrition():
    render_page_header(
        t("nutrition"),
        "Reference nutrition values by serving size. Values are not medical prescriptions.",
    )

    food_list = CLASS_NAMES + ["Almonds"]
    default_index = 0
    if st.session_state.prediction in food_list:
        default_index = food_list.index(st.session_state.prediction)

    food = st.selectbox("Select food", food_list, index=default_index)

    if food == "Almonds":
        d = EXTRA_COMPARE["Almonds"]
        grams = st.slider("Serving size (g)", 25, 250, 100, 25)
        factor = grams / 100.0

        cols = st.columns(4)
        metrics = [
            ("Energy", f"{d['energy_kcal']*factor:.1f} kcal", "🔥"),
            ("Protein", f"{d['protein_g']*factor:.2f} g", "💪"),
            ("Carbohydrates", f"{d['carbs_g']*factor:.2f} g", "🌾"),
            ("Fibre", f"{d['fibre_g']*factor:.2f} g", "🌿"),
        ]
        for c, item in zip(cols, metrics):
            with c:
                render_metric(*item)
        st.caption(d["source"])
        return

    render_nutrition_section(food)



@st.cache_data(show_spinner=False)
def load_india_geojson():
    geojson_path = APP_DIR / CULTIVATION_GEOJSON_FILE

    if not geojson_path.exists():
        raise FileNotFoundError(
            f"Missing {CULTIVATION_GEOJSON_FILE}. "
            "Place it in the same folder as web_app.py."
        )

    with geojson_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_polygon_rings(geometry):
    """Yield polygon exterior rings from Polygon/MultiPolygon geometry."""
    if not geometry:
        return
    gtype = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if gtype == "Polygon":
        if coords:
            yield coords[0]
    elif gtype == "MultiPolygon":
        for polygon in coords:
            if polygon:
                yield polygon[0]


def page_cultivation_map():
    render_page_header(
        t("cultivation_map"),
        "Explore selected cultivation regions across India for the detected food classes.",
    )

    default_index = (
        CLASS_NAMES.index(st.session_state.prediction)
        if st.session_state.prediction in CLASS_NAMES
        else 0
    )

    food = st.selectbox(
        "Select food class",
        CLASS_NAMES,
        index=default_index,
        key="cultivation_food",
    )

    info = CULTIVATION_STATES.get(
        food,
        {
            "states": [],
            "note": "Selected regions are not available for this class yet.",
        },
    )
    highlighted_states = info["states"]

    st.markdown(
        f"""
        <div class="map-card">
            <div style="font-size:1.18rem;font-weight:800;color:#5b2b26;">
                {CATEGORY_EMOJI.get(CATEGORY_MAP.get(food, "Food"), "🌾")} {html.escape(food)}
            </div>
            <div class="small-muted" style="margin-top:4px;">
                {len(highlighted_states)} selected cultivation / production regions
            </div>
            <div class="map-legend">
                <span class="map-legend-item">
                    <span class="map-dot" style="background:#b26a3c;"></span>
                    Selected region
                </span>
                <span class="map-legend-item">
                    <span class="map-dot" style="background:#eee7d7;"></span>
                    Other Indian state
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def get_state_name(feature):
        props = (feature or {}).get("properties", {}) or {}
        return str(
            props.get("ST_NM")
            or props.get("State_Name")
            or props.get("STATE")
            or props.get("name")
            or props.get("NAME_1")
            or props.get("State")
            or props.get("state")
            or props.get("st_nm")
            or ""
        ).strip()

    def normalize_state(name):
        value = (
            str(name)
            .lower()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .replace("&", "and")
        )
        aliases = {
            "orissa": "odisha",
            "uttaranchal": "uttarakhand",
            "pondicherry": "puducherry",
            "telengana": "telangana",
        }
        return aliases.get(value, value)

    selected_normalized = {normalize_state(name) for name in highlighted_states}

    try:
        india_geojson = load_india_geojson()
    except Exception as exc:
        st.error("India map file was not found or could not be read.")
        st.caption(f"Map data error: {exc}")
        return

    # Render the India-only map on the server with Pillow. This avoids relying on
    # Leaflet/CDN JavaScript inside an iframe, which can appear blank in some browsers.
    map_w, map_h = 1100, 780
    img = Image.new("RGB", (map_w, map_h), "#fbf8ef")
    draw = ImageDraw.Draw(img)

    lon_min, lon_max = 67.5, 98.5
    lat_min, lat_max = 5.0, 37.5
    pad = 36

    def project(pt):
        lon, lat = float(pt[0]), float(pt[1])
        x = pad + (lon - lon_min) / (lon_max - lon_min) * (map_w - 2 * pad)
        y = map_h - pad - (lat - lat_min) / (lat_max - lat_min) * (map_h - 2 * pad)
        return int(round(x)), int(round(y))

    feature_count = 0
    for feature in india_geojson.get("features", []):
        state_name = get_state_name(feature)
        is_selected = normalize_state(state_name) in selected_normalized
        fill = (178, 106, 60) if is_selected else (238, 231, 215)
        edge = (123, 74, 61) if is_selected else (185, 176, 159)
        width = 3 if is_selected else 2

        for ring in _iter_polygon_rings(feature.get("geometry") or {}) or []:
            pts = [project(pt) for pt in ring if len(pt) >= 2]
            if len(pts) < 3:
                continue
            draw.polygon(pts, fill=fill)
            draw.line(pts + [pts[0]], fill=edge, width=width)
            feature_count += 1

    if feature_count == 0:
        st.error("The India GeoJSON file contains no drawable polygon boundaries.")
        return

    # Keep the visual language aligned with the app's traditional Indian theme.
    draw.text((pad, 10), "🇮🇳 India", fill=(91, 43, 38))
    st.image(img, use_container_width=True)

    st.markdown(
        '<div class="section-title">Selected regions</div>',
        unsafe_allow_html=True,
    )

    region_cols = st.columns(3)
    for idx, state in enumerate(highlighted_states):
        with region_cols[idx % 3]:
            st.markdown(
                f'<span class="tag">📍 {html.escape(state)}</span>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="warning-box" style="margin-top:13px;">
          <b>Map note:</b> {html.escape(info["note"])}
          Exact cultivation intensity and rankings can change by agricultural year.
          This is a selected-region educational layer, not a complete state-wise production census.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "India state boundaries are loaded from the local india-states-simplified.geojson file. "
        "Crop-region references: ICAR, ICAR-IIPR, PIB/Ministry of Agriculture, "
        "and National Horticulture Board materials, as applicable."
    )

def page_shop_foods():
    render_page_header(
        t("shop_foods"),
        "Choose a food class, then open marketplace search results on Amazon India or Flipkart.",
    )

    if st.session_state.prediction in CLASS_NAMES:
        st.info(
            f"Latest detected food: **{st.session_state.prediction}** "
            f"• {CATEGORY_MAP.get(st.session_state.prediction, 'Food')} "
            f"• {st.session_state.prediction_confidence:.2%} confidence"
        )

    st.markdown(
        f"""
        <div class="glass-card">
            <div style="font-size:1.08rem;font-weight:800;color:#5b2b26;">
                🛒 Shop the recognised foods
            </div>
            <p style="color:#6d645c;line-height:1.55;margin-bottom:5px;">
                Choose a food below to open its marketplace search page.
            </p>
            <div class="small-muted">{html.escape(SHOP_NOTE)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_food = st.selectbox(
        "Select a food",
        CLASS_NAMES,
        index=CLASS_NAMES.index(st.session_state.prediction)
        if st.session_state.prediction in CLASS_NAMES else 0,
        key="shop_selected_food",
    )

    query = SHOP_QUERIES[selected_food]
    amazon_url = "https://www.amazon.in/s?k=" + urllib.parse.quote_plus(query)
    flipkart_url = "https://www.flipkart.com/search?q=" + urllib.parse.quote_plus(query)

    st.markdown(
        f"""
        <div class="shop-card" style="margin-top:14px;">
            <div style="font-size:1.5rem;">{CATEGORY_EMOJI.get(CATEGORY_MAP.get(selected_food, 'Food'), '🍽️')}</div>
            <div class="shop-card-title">{html.escape(selected_food)}</div>
            <div class="shop-card-meta">{html.escape(CATEGORY_MAP.get(selected_food, 'Food'))} • Search: {html.escape(query)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2 = st.columns(2)
    with b1:
        st.link_button(
            "🛍️ Shop on Amazon India",
            amazon_url,
            use_container_width=True,
        )
    with b2:
        st.link_button(
            "🛒 Shop on Flipkart",
            flipkart_url,
            use_container_width=True,
        )

    st.markdown(
        '<div class="section-title">All supported foods</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for idx, food in enumerate(CLASS_NAMES):
        category = CATEGORY_MAP.get(food, "Food")
        q = SHOP_QUERIES[food]
        a_url = "https://www.amazon.in/s?k=" + urllib.parse.quote_plus(q)
        f_url = "https://www.flipkart.com/search?q=" + urllib.parse.quote_plus(q)

        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="shop-card">
                    <div style="font-size:1.25rem;">{CATEGORY_EMOJI.get(category, '🍽️')}</div>
                    <div class="shop-card-title">{html.escape(food)}</div>
                    <div class="shop-card-meta">{html.escape(category)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.link_button("Amazon", a_url, use_container_width=True)
            with c2:
                st.link_button("Flipkart", f_url, use_container_width=True)

    st.caption(
        "Marketplace search links are used instead of fixed product pages so the user can see current listings. "
        "Prices, stock and delivery depend on the marketplace and location."
    )



def page_daily_intake():
    render_page_header(
        t("daily_intake"),
        "General daily food-group guidance based on ICMR-NIN Dietary Guidelines 2024.",
    )

    st.markdown(
        """
        <div class="glass-card">
            <div style="font-size:1.1rem;font-weight:800;color:#5b2b26;">
                🌿 Daily Intake Guidance
            </div>
            <p style="color:#6d645c;line-height:1.55;margin-bottom:8px;">
                Use this section as a general reference for balanced food-group intake.
                The values are based on the body weights and activity categories used
                in the ICMR-NIN 2024 guidance.
            </p>
            <span class="pill">Children</span>
            <span class="pill">Teenagers</span>
            <span class="pill">Adults</span>
            <span class="pill">Elderly</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    group = st.segmented_control(
        "Select age group",
        ["Children", "Teenagers", "Adults", "Elderly"],
        default="Children",
        key="daily_intake_group",
    )

    rows = DAILY_INTAKE_GUIDANCE[group]

    # Keep the main summary as a compact table.
    df = pd.DataFrame(
        [
            {
                "Age / Activity": r["group"],
                "Sex": r["sex"],
                "Cereals / Millets (g/day)": r["cereals_millets_g"],
                "Pulses & Beans (g/day)": r["pulses_beans_g"],
                "Energy* (kcal/day)": r["energy_kcal"],
                "Protein* (g/day)": r["protein_g"],
            }
            for r in rows
        ]
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">What this means</div>',
        unsafe_allow_html=True,
    )

    selected_index = st.selectbox(
        "Choose a reference row",
        range(len(rows)),
        format_func=lambda i: f"{rows[i]['group']} • {rows[i]['sex']}",
        key=f"daily_reference_{group}",
    )
    selected = rows[selected_index]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("Cereals / Millets", f"{selected['cereals_millets_g']} g/day", "🌾")
    with c2:
        render_metric("Pulses & Beans", f"{selected['pulses_beans_g']} g/day", "🫘")
    with c3:
        render_metric("Energy*", f"~{selected['energy_kcal']} kcal", "🔥")
    with c4:
        render_metric("Protein*", f"{selected['protein_g']} g", "💪")

    if group == "Children":
        st.markdown(
            """
            <div class="glass-card" style="margin-top:14px;">
                <div style="font-weight:800;color:#5b2b26;">🌾 Millet note</div>
                <p style="color:#6d645c;line-height:1.55;">
                    ICMR-NIN notes that for children up to 10 years, about 20% of
                    cereals (by raw weight) can be from millets.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif group == "Adults":
        st.markdown(
            """
            <div class="glass-card" style="margin-top:14px;">
                <div style="font-weight:800;color:#5b2b26;">🌾 Millet note</div>
                <p style="color:#6d645c;line-height:1.55;">
                    ICMR-NIN notes that for adults, about 20–30% of cereals
                    (by raw weight) should be from millets.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="warning-box" style="margin-top:15px;">
          <b>Important:</b> These are general food-group reference values, not a
          personalized diet prescription. The guideline notes that quantities can
          increase or decrease with body weight and physical activity. People with
          medical conditions, allergies, pregnancy, or special dietary needs should
          seek individualized professional advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Source: ICMR–National Institute of Nutrition (NIN), Dietary Guidelines for Indians, 2024, Table 1.6."
    )


def page_compare():
    render_page_header(
        t("compare"),
        "Compare measurable reference nutrition attributes without treating the comparison as a medical ranking.",
    )

    food_options = CLASS_NAMES + ["Almonds"]
    c1, c2 = st.columns(2)

    with c1:
        food_a = st.selectbox("Food A", food_options, index=0)
    with c2:
        default_b = 1 if len(food_options) > 1 else 0
        food_b = st.selectbox("Food B", food_options, index=default_b)

    grams = st.slider("Comparison serving size (g)", 25, 250, 100, 25)

    def get_compare(food):
        if food == "Almonds":
            d = EXTRA_COMPARE["Almonds"]
            f = grams / 100.0
            return {
                "Energy (kcal)": d["energy_kcal"] * f,
                "Protein (g)": d["protein_g"] * f,
                "Carbohydrates (g)": d["carbs_g"] * f,
                "Fibre (g)": d["fibre_g"] * f,
                "Fat (g)": d["fat_g"] * f,
            }

        v = nutrition_value(food, grams)
        if v is None:
            return None

        def vv(*keys):
            for k in keys:
                if isinstance(v, dict) and k in v:
                    return v[k]
                if hasattr(v, k):
                    return getattr(v, k)
            return np.nan

        return {
            "Energy (kcal)": vv("energy_kcal", "kcal"),
            "Protein (g)": vv("protein_g", "protein"),
            "Carbohydrates (g)": vv("carbs_g", "carbohydrates_g", "carbohydrates"),
            "Fibre (g)": vv("fibre_g", "fiber_g", "fibre"),
            "Fat (g)": vv("fat_g", "fat"),
        }

    a = get_compare(food_a)
    b = get_compare(food_b)

    if a is None or b is None:
        st.warning("Reference data is unavailable for one of the selected foods.")
        return

    rows = []
    for metric in a:
        rows.append(
            {
                "Attribute": metric,
                food_a: f"{a[metric]:.2f}",
                food_b: f"{b[metric]:.2f}",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="warning-box">
          Values shown are reference nutrition values for the selected serving size.
          They should be interpreted as a measurable comparison, not as a universal
          claim that one food is medically or nutritionally “better” for every person.
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_history():
    render_page_header(
        t("history"),
        "Review predictions from this browser session.",
    )

    if not st.session_state.history:
        st.info("No predictions recorded yet.")
        return

    df = pd.DataFrame(st.session_state.history)
    df["confidence"] = df["confidence"].map(lambda x: f"{x:.2%}")
    st.dataframe(
        df[["time", "food", "category", "confidence"]],
        use_container_width=True,
        hide_index=True,
    )

    selected = st.selectbox("Open a history item", list(range(len(st.session_state.history))))
    item = st.session_state.history[selected]

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="result-title">{html.escape(item['food'])}</div>
            <div class="small-muted">
                {html.escape(item['time'])} • {html.escape(item['category'])} • {item['confidence']:.2%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Top-3 outputs")
    st.dataframe(
        pd.DataFrame(
            {
                "Class": [x[0] for x in item["top3"]],
                "Confidence": [f"{x[1]:.2%}" for x in item["top3"]],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Open full details"):
        st.session_state.prediction = item["food"]
        st.session_state.prediction_confidence = item["confidence"]
        st.session_state.prediction_top3 = item["top3"]
        st.session_state.prediction_image = None
        set_page("Identify Food")


def page_about():
    render_page_header(
        t("about"),
        "Dhanya AI — a hackathon prototype for Indian millets, pulses and grains.",
    )

    st.markdown(
        """
        <div class="glass-card">
            <div style="font-size:1.2rem;font-weight:800;">🧠 Model architecture</div>
            <p>
              The current identification model in the app is our custom CNN image
              classifier trained as a custom CNN baseline from scratch.
              The training pipeline uses a balanced 10,000-image augmented training set
              built from 200 original images across 10 classes, with separate holdout
              validation and test images.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric("Training images", "10,000", "📊")
    with c2:
        render_metric("Original source images", "200", "🗂️")
    with c3:
        render_metric("Classes", "10", "🏷️")

    st.markdown('<div class="section-title">Supported classes</div>', unsafe_allow_html=True)
    st.markdown(
        "".join(
            f'<span class="tag">{html.escape(f)}</span>'
            for f in CLASS_NAMES
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Responsible-use notes</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - The classifier is limited to its trained classes.
        - A high confidence score does not prove that an image is an in-distribution sample.
        - Nutrition values are reference estimates and can vary with food variety and preparation.
        - The assistant is for educational food information, not diagnosis or treatment.
        - A future Swin Transformer branch can be evaluated as an advanced architecture for comparison.
        """
    )


# ============================================================
# APP ROUTER
# ============================================================
render_sidebar()

page = st.session_state.page

if page == "Dashboard":
    page_dashboard()
elif page == "Identify Food":
    page_identify()
elif page == "AI Assistant":
    page_assistant()
elif page == "Explore Foods":
    page_explore()
elif page == "Shop Foods":
    page_shop_foods()
elif page == "Budget Planner":
    page_budget_planner()
elif page == "Cultivation Map":
    page_cultivation_map()
elif page == "Nutrition & Diet":
    page_nutrition()
elif page == "Daily Intake":
    page_daily_intake()
elif page == "Compare":
    page_compare()
elif page == "History":
    page_history()
elif page == "About":
    page_about()
else:
    page_dashboard()

st.markdown(
    f'<div class="footer-note">Dhanya AI • Food intelligence prototype • {html.escape(dt.datetime.now().strftime("%d %B %Y"))}</div>',
    unsafe_allow_html=True,
)

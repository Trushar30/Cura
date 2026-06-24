#!/usr/bin/env python3
"""
============================================================================
Healthcare Recommendation System - Data Generation Script
============================================================================
Generates all synthetic datasets mirroring real Kaggle healthcare dataset
structures. This ensures we have consistent, comprehensive data for all
recommendation model pipelines.

Datasets generated:
  1. symptoms_disease.csv      - Symptom-to-disease mapping (~4900 rows)
  2. disease_description.csv   - Disease descriptions (41 diseases)
  3. disease_precaution.csv    - Precautions per disease
  4. disease_medication.csv    - Medications, diet, exercise per disease
  5. synthetic_users.csv       - 500 user profiles
  6. synthetic_interactions.csv - 10,000 user interactions
  7. synthetic_reviews.csv     - 2,000 user reviews with text
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Output directory ─────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ============================================================================
# 1. DISEASE & SYMPTOM KNOWLEDGE BASE
# ============================================================================

DISEASES = {
    "Fungal infection": {
        "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
        "description": "Fungal infection is a common skin condition caused by fungi. It can affect the skin, nails, and hair. Symptoms include itching, redness, and skin rashes that may spread across the body.",
        "precautions": ["bath twice", "use detol or neem in bathing water", "keep infected area dry", "use clean cloths"],
        "medication": ["Antifungal Cream", "Fluconazole", "Clotrimazole"],
        "diet": ["Probiotics", "Garlic", "Coconut oil", "Turmeric"],
        "exercise": ["Light walking", "Yoga", "Stretching"],
        "specialist": "Dermatologist"
    },
    "Allergy": {
        "symptoms": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes"],
        "description": "An allergy is an immune system response to a foreign substance that is not typically harmful to the body. Allergens can include pollen, pet dander, dust mites, and certain foods.",
        "precautions": ["apply calamine", "cover area with bandage", "use ice to compress itching", "avoid allergens"],
        "medication": ["Cetirizine", "Loratadine", "Fexofenadine"],
        "diet": ["Vitamin C rich foods", "Local honey", "Green tea", "Ginger"],
        "exercise": ["Indoor exercises", "Swimming (chlorinated pool)", "Light yoga"],
        "specialist": "Allergist"
    },
    "GERD": {
        "symptoms": ["stomach_pain", "acidity", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"],
        "description": "Gastroesophageal Reflux Disease (GERD) occurs when stomach acid frequently flows back into the esophagus. This backwash can irritate the lining of the esophagus and cause heartburn.",
        "precautions": ["avoid fatty spicy food", "avoid lying down after eating", "maintain healthy weight", "exercise regularly"],
        "medication": ["Omeprazole", "Pantoprazole", "Ranitidine"],
        "diet": ["Oatmeal", "Ginger", "Lean meats", "Non-citrus fruits"],
        "exercise": ["Walking after meals", "Gentle yoga", "Cycling"],
        "specialist": "Gastroenterologist"
    },
    "Chronic cholesterol": {
        "symptoms": ["obesity", "heart_failure", "breathlessness", "sweating", "high_cholesterol"],
        "description": "Chronic cholesterol refers to consistently elevated levels of cholesterol in the blood, which can lead to atherosclerosis and increase the risk of heart disease and stroke.",
        "precautions": ["exercise regularly", "eat healthy", "reduce salt intake", "avoid smoking"],
        "medication": ["Atorvastatin", "Rosuvastatin", "Simvastatin"],
        "diet": ["Oats", "Nuts", "Fatty fish", "Olive oil"],
        "exercise": ["Brisk walking", "Jogging", "Swimming", "Cycling"],
        "specialist": "Cardiologist"
    },
    "Diabetes": {
        "symptoms": ["fatigue", "weight_loss", "excessive_hunger", "increased_appetite", "polyuria", "blurred_vision"],
        "description": "Diabetes is a chronic condition that occurs when the body cannot effectively process blood sugar (glucose). Type 2 diabetes is the most common form and is often linked to lifestyle factors.",
        "precautions": ["monitor blood sugar", "follow diabetic diet", "exercise regularly", "consult doctor regularly"],
        "medication": ["Metformin", "Glipizide", "Insulin"],
        "diet": ["Leafy greens", "Whole grains", "Lean protein", "Berries"],
        "exercise": ["Walking 30 min daily", "Swimming", "Resistance training"],
        "specialist": "Endocrinologist"
    },
    "Bronchial Asthma": {
        "symptoms": ["breathlessness", "cough", "high_fever", "mucoid_sputum", "family_history"],
        "description": "Bronchial asthma is a chronic inflammatory disease of the airways characterized by recurrent episodes of wheezing, breathlessness, chest tightness, and coughing.",
        "precautions": ["avoid allergens", "keep inhaler handy", "avoid cold air", "stay indoors during pollution"],
        "medication": ["Salbutamol Inhaler", "Montelukast", "Budesonide"],
        "diet": ["Fruits rich in Vitamin C", "Omega-3 fatty acids", "Ginger tea", "Turmeric milk"],
        "exercise": ["Swimming", "Walking", "Breathing exercises"],
        "specialist": "Pulmonologist"
    },
    "Hypertension": {
        "symptoms": ["headache", "chest_pain", "dizziness", "lack_of_concentration", "breathlessness"],
        "description": "Hypertension or high blood pressure is a condition in which the force of blood against artery walls is consistently too high. It is a major risk factor for heart disease and stroke.",
        "precautions": ["meditation", "reduce salt intake", "regular exercise", "proper diet"],
        "medication": ["Amlodipine", "Lisinopril", "Losartan"],
        "diet": ["DASH diet", "Bananas", "Leafy greens", "Low-sodium foods"],
        "exercise": ["Brisk walking", "Cycling", "Swimming", "Yoga"],
        "specialist": "Cardiologist"
    },
    "Migraine": {
        "symptoms": ["headache", "acidity", "indigestion", "blurred_vision", "excessive_hunger", "visual_disturbances"],
        "description": "Migraine is a neurological condition characterized by intense, debilitating headaches often accompanied by nausea, vomiting, and sensitivity to light and sound.",
        "precautions": ["meditation", "reduce stress", "maintain regular sleep", "avoid bright lights"],
        "medication": ["Sumatriptan", "Ibuprofen", "Topiramate"],
        "diet": ["Magnesium-rich foods", "Ginger", "Fatty fish", "Water-rich fruits"],
        "exercise": ["Gentle yoga", "Walking", "Tai chi"],
        "specialist": "Neurologist"
    },
    "Cervical spondylosis": {
        "symptoms": ["neck_pain", "back_pain", "dizziness", "loss_of_balance", "weakness_in_limbs"],
        "description": "Cervical spondylosis is age-related wear and tear affecting the spinal disks in the neck. It is very common and worsens with age, causing neck pain and stiffness.",
        "precautions": ["use heating pad or ice pack", "exercise regularly", "maintain proper posture", "take breaks from sitting"],
        "medication": ["Diclofenac", "Pregabalin", "Methocarbamol"],
        "diet": ["Calcium-rich foods", "Vitamin D", "Anti-inflammatory foods", "Turmeric"],
        "exercise": ["Neck stretches", "Shoulder rolls", "Swimming", "Pilates"],
        "specialist": "Orthopedist"
    },
    "Paralysis (brain hemorrhage)": {
        "symptoms": ["vomiting", "headache", "weakness_of_one_body_side", "altered_sensorium"],
        "description": "Paralysis due to brain hemorrhage occurs when bleeding in the brain damages nerve cells, leading to loss of movement and sensation in parts of the body.",
        "precautions": ["immediate medical attention", "control blood pressure", "rehabilitation therapy", "regular monitoring"],
        "medication": ["Nimodipine", "Mannitol", "Physiotherapy"],
        "diet": ["Soft foods", "High-protein diet", "Fruits and vegetables", "Adequate fluids"],
        "exercise": ["Physiotherapy exercises", "Assisted walking", "Range of motion exercises"],
        "specialist": "Neurologist"
    },
    "Jaundice": {
        "symptoms": ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "yellowish_skin", "dark_urine", "abdominal_pain"],
        "description": "Jaundice is a condition in which the skin and whites of the eyes become yellow due to elevated bilirubin levels. It indicates liver dysfunction or bile duct obstruction.",
        "precautions": ["drink plenty of water", "consume milk thistle", "eat fruits and high fiber food", "avoid alcohol"],
        "medication": ["No specific medication (treat underlying cause)", "Ursodeoxycholic acid", "Vitamin K"],
        "diet": ["Fresh fruits", "Vegetables", "Whole grains", "Lean protein"],
        "exercise": ["Rest initially", "Light walking when recovering", "Gentle stretching"],
        "specialist": "Gastroenterologist"
    },
    "Malaria": {
        "symptoms": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain"],
        "description": "Malaria is a life-threatening disease caused by parasites transmitted through the bite of infected female Anopheles mosquitoes. It causes high fever, chills, and flu-like symptoms.",
        "precautions": ["use mosquito net", "consult doctor", "avoid stagnant water", "use insect repellent"],
        "medication": ["Chloroquine", "Artemether-Lumefantrine", "Quinine"],
        "diet": ["Fluids", "Light foods", "Citrus fruits", "Protein-rich foods"],
        "exercise": ["Complete rest during infection", "Light walking during recovery"],
        "specialist": "Infectious Disease Specialist"
    },
    "Common Cold": {
        "symptoms": ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache", "runny_nose", "congestion"],
        "description": "The common cold is a viral infection of the upper respiratory tract. It is usually harmless and resolves within a week or two. Symptoms include runny nose, sore throat, and cough.",
        "precautions": ["drink vitamin C rich juice", "take warm bath", "avoid cold food", "keep body warm"],
        "medication": ["Paracetamol", "Dextromethorphan", "Pseudoephedrine"],
        "diet": ["Chicken soup", "Citrus fruits", "Ginger tea", "Honey"],
        "exercise": ["Rest", "Light stretching", "Walking if feeling okay"],
        "specialist": "General Physician"
    },
    "Pneumonia": {
        "symptoms": ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "chest_pain", "fast_heart_rate", "rusty_sputum"],
        "description": "Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid. It can be caused by bacteria, viruses, or fungi.",
        "precautions": ["consult doctor immediately", "take prescribed antibiotics", "rest adequately", "stay hydrated"],
        "medication": ["Amoxicillin", "Azithromycin", "Levofloxacin"],
        "diet": ["Warm fluids", "Protein-rich foods", "Fruits", "Whole grains"],
        "exercise": ["Deep breathing exercises", "Rest initially", "Gradual return to activity"],
        "specialist": "Pulmonologist"
    },
    "Dengue": {
        "symptoms": ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "muscle_pain", "loss_of_appetite"],
        "description": "Dengue is a mosquito-borne viral disease that occurs in tropical and subtropical areas. Severe dengue can cause bleeding, organ impairment, and plasma leakage.",
        "precautions": ["drink papaya leaf juice", "avoid aspirin", "keep hydrated", "use mosquito repellent"],
        "medication": ["Paracetamol (no aspirin)", "IV fluids", "Platelet monitoring"],
        "diet": ["Papaya leaf extract", "Coconut water", "Pomegranate juice", "Kiwi"],
        "exercise": ["Complete rest during fever", "Light walking during recovery"],
        "specialist": "Infectious Disease Specialist"
    },
    "Heart attack": {
        "symptoms": ["chest_pain", "breathlessness", "vomiting", "sweating", "fatigue"],
        "description": "A heart attack occurs when blood flow to the heart is severely reduced or blocked, usually by a buildup of fat, cholesterol, and other substances in coronary arteries.",
        "precautions": ["call emergency immediately", "chew aspirin if advised", "rest in comfortable position", "avoid physical exertion"],
        "medication": ["Aspirin", "Nitroglycerin", "Beta-blockers", "ACE inhibitors"],
        "diet": ["Mediterranean diet", "Omega-3 rich foods", "Fruits and vegetables", "Whole grains"],
        "exercise": ["Cardiac rehabilitation", "Walking (post-recovery)", "Light cycling"],
        "specialist": "Cardiologist"
    },
    "Typhoid": {
        "symptoms": ["chills", "vomiting", "fatigue", "high_fever", "headache", "nausea", "constipation", "abdominal_pain", "diarrhoea"],
        "description": "Typhoid fever is a bacterial infection caused by Salmonella typhi. It spreads through contaminated food and water. Symptoms include prolonged fever, weakness, and abdominal pain.",
        "precautions": ["eat high calorie food", "maintain hygiene", "drink boiled water", "complete antibiotic course"],
        "medication": ["Ciprofloxacin", "Azithromycin", "Ceftriaxone"],
        "diet": ["High-calorie foods", "Boiled water", "Bananas", "Rice porridge"],
        "exercise": ["Complete rest during fever", "Gradual activity increase"],
        "specialist": "Infectious Disease Specialist"
    },
    "Hepatitis B": {
        "symptoms": ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "abdominal_pain", "receiving_blood_transfusion"],
        "description": "Hepatitis B is a serious liver infection caused by the hepatitis B virus. It can become chronic and increase the risk of liver failure, liver cancer, or cirrhosis.",
        "precautions": ["consult nearest hospital", "get vaccination", "avoid contact with infected blood", "follow safe practices"],
        "medication": ["Entecavir", "Tenofovir", "Interferon alfa"],
        "diet": ["High-protein diet", "Fresh fruits", "Vegetables", "Avoid alcohol"],
        "exercise": ["Light walking", "Gentle stretching", "Rest as needed"],
        "specialist": "Hepatologist"
    },
    "Hepatitis C": {
        "symptoms": ["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "dark_urine", "abdominal_pain"],
        "description": "Hepatitis C is a viral infection that causes liver inflammation and can lead to serious liver damage. It spreads through contaminated blood.",
        "precautions": ["consult doctor", "avoid alcohol", "get tested regularly", "avoid sharing needles"],
        "medication": ["Sofosbuvir", "Ledipasvir", "Ribavirin"],
        "diet": ["Whole grains", "Lean protein", "Fruits and vegetables", "Coffee (moderate)"],
        "exercise": ["Walking", "Light aerobics", "Yoga"],
        "specialist": "Hepatologist"
    },
    "Hepatitis D": {
        "symptoms": ["joint_pain", "fatigue", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain"],
        "description": "Hepatitis D is a liver disease caused by the hepatitis D virus. It only occurs in people already infected with hepatitis B and can accelerate liver damage.",
        "precautions": ["get hepatitis B vaccination", "consult doctor", "follow medication strictly", "regular liver function tests"],
        "medication": ["Pegylated Interferon alfa", "Supportive care"],
        "diet": ["Balanced diet", "Avoid alcohol", "Fresh fruits", "Lean proteins"],
        "exercise": ["Gentle walking", "Stretching", "Rest when fatigued"],
        "specialist": "Hepatologist"
    },
    "Hepatitis E": {
        "symptoms": ["joint_pain", "fatigue", "vomiting", "high_fever", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain"],
        "description": "Hepatitis E is a liver disease caused by the hepatitis E virus. It is mainly spread through drinking contaminated water. Most cases resolve on their own.",
        "precautions": ["drink clean water", "maintain hygiene", "rest adequately", "avoid alcohol"],
        "medication": ["Supportive treatment", "Ribavirin (severe cases)", "Rest"],
        "diet": ["Boiled water", "Fresh fruits", "Light meals", "Avoid fried foods"],
        "exercise": ["Rest", "Light walking when recovering"],
        "specialist": "Gastroenterologist"
    },
    "Alcoholic hepatitis": {
        "symptoms": ["vomiting", "yellowish_skin", "abdominal_pain", "swelling_of_stomach", "distention_of_abdomen", "fluid_overload"],
        "description": "Alcoholic hepatitis is liver inflammation caused by excessive alcohol consumption. It can lead to liver failure and is potentially life-threatening.",
        "precautions": ["stop alcohol consumption", "consult doctor", "follow medication", "maintain proper diet"],
        "medication": ["Prednisolone", "Pentoxifylline", "Nutritional support"],
        "diet": ["High-protein diet", "Vitamin supplements", "No alcohol", "Balanced meals"],
        "exercise": ["Light walking", "Gentle stretching", "Avoid strenuous activity"],
        "specialist": "Hepatologist"
    },
    "Tuberculosis": {
        "symptoms": ["chills", "vomiting", "fatigue", "weight_loss", "cough", "high_fever", "breathlessness", "sweating", "loss_of_appetite", "blood_in_sputum"],
        "description": "Tuberculosis (TB) is a potentially serious bacterial infection that mainly affects the lungs. It spreads through airborne droplets when an infected person coughs or sneezes.",
        "precautions": ["cover mouth while coughing", "complete TB medication course", "consult doctor regularly", "isolate if infectious"],
        "medication": ["Isoniazid", "Rifampicin", "Pyrazinamide", "Ethambutol"],
        "diet": ["High-calorie foods", "Protein-rich meals", "Fresh fruits", "Milk and eggs"],
        "exercise": ["Rest during active TB", "Breathing exercises", "Gradual walking"],
        "specialist": "Pulmonologist"
    },
    "Peptic ulcer disease": {
        "symptoms": ["vomiting", "indigestion", "loss_of_appetite", "abdominal_pain", "passage_of_gases"],
        "description": "Peptic ulcer disease involves open sores that develop on the inside lining of the stomach and the upper portion of the small intestine, often caused by H. pylori bacteria.",
        "precautions": ["avoid fatty food", "eat small frequent meals", "reduce stress", "avoid NSAIDs"],
        "medication": ["Omeprazole", "Amoxicillin + Clarithromycin", "Sucralfate"],
        "diet": ["Probiotics", "Fiber-rich foods", "Lean meats", "Non-citrus fruits"],
        "exercise": ["Walking", "Gentle yoga", "Stress-reduction exercises"],
        "specialist": "Gastroenterologist"
    },
    "AIDS": {
        "symptoms": ["muscle_wasting", "patches_in_throat", "high_fever", "extra_marital_contacts"],
        "description": "AIDS (Acquired Immunodeficiency Syndrome) is a chronic condition caused by HIV that damages the immune system, making the body vulnerable to opportunistic infections.",
        "precautions": ["avoid unprotected sex", "consult doctor immediately", "follow ART medication", "regular blood tests"],
        "medication": ["Antiretroviral Therapy (ART)", "Tenofovir", "Emtricitabine"],
        "diet": ["High-protein diet", "Fresh fruits and vegetables", "Whole grains", "Adequate calories"],
        "exercise": ["Moderate aerobic exercise", "Strength training", "Yoga"],
        "specialist": "Immunologist"
    },
    "Gastroenteritis": {
        "symptoms": ["vomiting", "diarrhoea", "dehydration", "abdominal_pain", "nausea", "fever"],
        "description": "Gastroenteritis is inflammation of the stomach and intestines, typically resulting from bacterial or viral infections. It causes watery diarrhea, abdominal cramps, and vomiting.",
        "precautions": ["stay hydrated", "maintain hygiene", "eat bland food", "consult doctor if severe"],
        "medication": ["ORS (Oral Rehydration Salts)", "Ondansetron", "Loperamide"],
        "diet": ["BRAT diet (Banana, Rice, Applesauce, Toast)", "Clear fluids", "Electrolyte drinks"],
        "exercise": ["Rest", "Light walking when recovering"],
        "specialist": "Gastroenterologist"
    },
    "Urinary tract infection": {
        "symptoms": ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine"],
        "description": "A urinary tract infection (UTI) is an infection in any part of the urinary system. Most infections involve the lower urinary tract — the bladder and the urethra.",
        "precautions": ["drink plenty of water", "increase vitamin C intake", "maintain hygiene", "urinate frequently"],
        "medication": ["Nitrofurantoin", "Trimethoprim", "Ciprofloxacin"],
        "diet": ["Cranberry juice", "Water", "Probiotics", "Vitamin C-rich foods"],
        "exercise": ["Light walking", "Pelvic floor exercises", "Yoga"],
        "specialist": "Urologist"
    },
    "Psoriasis": {
        "symptoms": ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails"],
        "description": "Psoriasis is a chronic autoimmune condition that causes rapid buildup of skin cells, resulting in thick, silvery scales and itchy, dry, red patches on the skin.",
        "precautions": ["moisturize skin", "avoid triggers", "use prescribed medication", "manage stress"],
        "medication": ["Methotrexate", "Cyclosporine", "Topical corticosteroids"],
        "diet": ["Anti-inflammatory foods", "Fish oil", "Colorful fruits and vegetables", "Whole grains"],
        "exercise": ["Swimming", "Walking", "Yoga", "Low-impact exercises"],
        "specialist": "Dermatologist"
    },
    "Impetigo": {
        "symptoms": ["skin_rash", "high_fever", "blister", "red_sore_around_nose", "yellow_crust_ooze"],
        "description": "Impetigo is a common and highly contagious skin infection that mainly affects infants and young children. It causes red sores that can break open, ooze fluid, and form crusts.",
        "precautions": ["soak affected area in warm water", "use antibiotic cream", "avoid touching sores", "maintain hygiene"],
        "medication": ["Mupirocin", "Retapamulin", "Oral Cephalexin"],
        "diet": ["Vitamin A-rich foods", "Zinc-rich foods", "Fresh fruits", "Protein"],
        "exercise": ["Avoid contact sports", "Light indoor activities"],
        "specialist": "Dermatologist"
    },
    "Osteoarthritis": {
        "symptoms": ["joint_pain", "neck_pain", "knee_pain", "hip_joint_pain", "swelling_joints", "painful_walking", "stiff_neck"],
        "description": "Osteoarthritis is the most common form of arthritis, occurring when the protective cartilage that cushions the ends of bones wears down over time.",
        "precautions": ["maintain healthy weight", "exercise regularly", "use hot/cold therapy", "avoid joint stress"],
        "medication": ["Acetaminophen", "Ibuprofen", "Naproxen", "Glucosamine"],
        "diet": ["Omega-3 fatty acids", "Green leafy vegetables", "Nuts", "Berries"],
        "exercise": ["Swimming", "Cycling", "Tai chi", "Range of motion exercises"],
        "specialist": "Rheumatologist"
    },
    "Arthritis": {
        "symptoms": ["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking"],
        "description": "Arthritis is inflammation of one or more joints, causing pain and stiffness that can worsen with age. The most common types include osteoarthritis and rheumatoid arthritis.",
        "precautions": ["exercise regularly", "use hot and cold therapy", "maintain healthy weight", "try physical therapy"],
        "medication": ["Methotrexate", "Hydroxychloroquine", "NSAIDs"],
        "diet": ["Fish", "Nuts and seeds", "Fruits and vegetables", "Olive oil"],
        "exercise": ["Water aerobics", "Walking", "Cycling", "Stretching"],
        "specialist": "Rheumatologist"
    },
    "Acne": {
        "symptoms": ["skin_rash", "pus_filled_pimples", "blackheads", "scurring"],
        "description": "Acne is a skin condition that occurs when hair follicles become clogged with oil and dead skin cells. It causes whiteheads, blackheads, or pimples and is most common among teenagers.",
        "precautions": ["wash face twice daily", "avoid touching face", "use non-comedogenic products", "stay hydrated"],
        "medication": ["Benzoyl Peroxide", "Tretinoin", "Clindamycin"],
        "diet": ["Low-glycemic foods", "Zinc-rich foods", "Omega-3 fatty acids", "Fruits"],
        "exercise": ["Regular exercise (shower after)", "Yoga for stress reduction"],
        "specialist": "Dermatologist"
    },
    "Varicose veins": {
        "symptoms": ["fatigue", "cramps", "bruising", "obesity", "swollen_legs", "prominent_veins_on_calf"],
        "description": "Varicose veins are swollen, twisted veins visible just under the surface of the skin, usually in the legs. They result from weakened or damaged vein walls and valves.",
        "precautions": ["lie down flat and raise leg high", "use compression stockings", "avoid standing long", "maintain healthy weight"],
        "medication": ["Diosmin", "Sclerotherapy", "Compression stockings"],
        "diet": ["High-fiber foods", "Flavonoid-rich foods", "Berries", "Citrus fruits"],
        "exercise": ["Walking", "Cycling", "Leg lifts", "Calf raises"],
        "specialist": "Vascular Surgeon"
    },
    "Hypothyroidism": {
        "symptoms": ["fatigue", "weight_gain", "cold_hands_and_feets", "mood_swings", "lethargy", "dizziness", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", "depression", "irritability", "abnormal_menstruation"],
        "description": "Hypothyroidism is a condition in which the thyroid gland does not produce enough thyroid hormones. It can cause fatigue, weight gain, and depression.",
        "precautions": ["take thyroid medication daily", "regular blood tests", "reduce stress", "proper diet"],
        "medication": ["Levothyroxine", "Liothyronine"],
        "diet": ["Iodine-rich foods", "Selenium-rich foods", "Zinc", "High-fiber foods"],
        "exercise": ["Walking", "Yoga", "Strength training", "Swimming"],
        "specialist": "Endocrinologist"
    },
    "Hyperthyroidism": {
        "symptoms": ["fatigue", "mood_swings", "weight_loss", "restlessness", "sweating", "diarrhoea", "fast_heart_rate", "irritability", "muscle_weakness"],
        "description": "Hyperthyroidism is a condition where the thyroid gland produces too much thyroid hormone, leading to accelerated metabolism, rapid heartbeat, and weight loss.",
        "precautions": ["eat healthy", "take medication regularly", "regular check-ups", "manage stress"],
        "medication": ["Methimazole", "Propylthiouracil", "Propranolol"],
        "diet": ["Cruciferous vegetables", "Calcium-rich foods", "Berries", "Vitamin D"],
        "exercise": ["Low-impact exercises", "Walking", "Yoga", "Pilates"],
        "specialist": "Endocrinologist"
    },
    "Hypoglycemia": {
        "symptoms": ["vomiting", "fatigue", "anxiety", "sweating", "headache", "nausea", "blurred_vision", "excessive_hunger", "palpitations"],
        "description": "Hypoglycemia is a condition characterized by abnormally low blood sugar (glucose) levels. It is most common in people with diabetes and can cause confusion and fainting.",
        "precautions": ["carry glucose tablets", "eat regular meals", "monitor blood sugar", "avoid skipping meals"],
        "medication": ["Glucose tablets", "Glucagon injection", "Dextrose IV"],
        "diet": ["Complex carbohydrates", "Protein-rich snacks", "Whole grains", "Nuts"],
        "exercise": ["Monitor glucose before exercise", "Keep snacks nearby", "Moderate activity"],
        "specialist": "Endocrinologist"
    },
    "Drug Reaction": {
        "symptoms": ["itching", "skin_rash", "stomach_pain", "burning_micturition", "spotting_urination"],
        "description": "A drug reaction is an unwanted side effect caused by a medication. It can range from mild skin rashes to severe allergic reactions (anaphylaxis).",
        "precautions": ["stop the drug immediately", "consult doctor", "report to pharmacovigilance", "carry allergy card"],
        "medication": ["Epinephrine (severe cases)", "Antihistamines", "Corticosteroids"],
        "diet": ["Bland foods", "Plenty of water", "Probiotics", "Light meals"],
        "exercise": ["Rest", "Light walking if feeling okay"],
        "specialist": "Allergist"
    },
    "Chicken pox": {
        "symptoms": ["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache", "loss_of_appetite", "red_spots_over_body", "malaise"],
        "description": "Chickenpox is a highly contagious viral infection causing an itchy, blister-like rash on the skin. It mainly affects children but can be more severe in adults.",
        "precautions": ["use neem in bathing", "consume neem leaves", "avoid public places", "keep skin clean"],
        "medication": ["Acyclovir", "Calamine lotion", "Paracetamol"],
        "diet": ["Soft foods", "Cold foods", "Plenty of fluids", "Vitamin A-rich foods"],
        "exercise": ["Rest until blisters crust over", "No contact sports"],
        "specialist": "General Physician"
    },
    "Dimorphic hemorrhoids (piles)": {
        "symptoms": ["constipation", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
        "description": "Hemorrhoids (piles) are swollen veins in the lower part of the anus and rectum. They can cause discomfort, itching, bleeding, and pain during bowel movements.",
        "precautions": ["avoid fatty food", "drink plenty of water", "eat high-fiber food", "avoid straining"],
        "medication": ["Hydrocortisone cream", "Lidocaine", "Psyllium husk"],
        "diet": ["High-fiber foods", "Plenty of water", "Fruits", "Whole grains"],
        "exercise": ["Kegel exercises", "Walking", "Yoga", "Avoid heavy lifting"],
        "specialist": "Proctologist"
    },
    "Vertigo": {
        "symptoms": ["vomiting", "headache", "nausea", "spinning_movements", "loss_of_balance", "unsteadiness"],
        "description": "Vertigo is a sensation of spinning or dizziness, often caused by inner ear problems. It can make you feel like you or your surroundings are moving or spinning.",
        "precautions": ["avoid sudden position changes", "lie down in dark room", "avoid driving", "do Epley maneuver"],
        "medication": ["Meclizine", "Diazepam", "Betahistine"],
        "diet": ["Low-sodium diet", "Adequate hydration", "Ginger", "Avoid caffeine"],
        "exercise": ["Balance exercises", "Epley maneuver", "Vestibular rehabilitation"],
        "specialist": "ENT Specialist"
    },
}

# Extra symptoms pool for diversity
ALL_SYMPTOMS = list(set(
    symptom
    for disease_data in DISEASES.values()
    for symptom in disease_data["symptoms"]
))

# Add more symptoms to increase diversity
EXTRA_SYMPTOMS = [
    "weight_gain", "anxiety", "lethargy", "irregular_sugar_level",
    "sunken_eyes", "dehydration", "loss_of_smell", "muscle_pain",
    "red_spots_over_body", "belly_pain", "receiving_unsterile_injections",
    "skin_peeling", "palpitations", "inflammatory_nails", "yellow_crust_ooze",
    "fluid_overload", "swelled_lymph_nodes", "malaise", "phlegm",
    "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose",
    "congestion", "loss_of_appetite", "constipation", "diarrhoea",
    "mild_fever", "muscle_wasting", "patches_in_throat", "stiff_neck",
    "swelling_joints", "movement_stiffness", "spinning_movements",
    "unsteadiness", "weakness_of_one_body_side", "altered_sensorium",
    "lack_of_concentration", "depression", "irritability", "visual_disturbances",
    "back_pain", "weakness_in_limbs", "painful_walking", "knee_pain",
    "hip_joint_pain", "cold_hands_and_feets", "puffy_face_and_eyes",
    "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
    "abnormal_menstruation", "fast_heart_rate", "restlessness",
    "toxic_look_(typhos)", "pain_behind_the_eyes", "scurring",
    "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine",
    "passage_of_gases", "internal_itching", "extra_marital_contacts",
    "distention_of_abdomen", "swelling_of_stomach", "bloody_stool",
    "irritation_in_anus", "pain_in_anal_region", "pain_during_bowel_movements",
    "prominent_veins_on_calf", "swollen_legs", "bruising", "cramps",
    "small_dents_in_nails", "silver_like_dusting", "blister",
    "red_sore_around_nose", "pus_filled_pimples", "blackheads",
    "spotting_urination", "receiving_blood_transfusion", "dischromic_patches",
    "nodal_skin_eruptions", "mucoid_sputum", "rusty_sputum",
    "blood_in_sputum", "family_history", "obesity", "high_cholesterol",
    "heart_failure", "ulcers_on_tongue", "watering_from_eyes"
]

ALL_SYMPTOMS = list(set(ALL_SYMPTOMS + EXTRA_SYMPTOMS))
ALL_SYMPTOMS.sort()

print(f"Total unique symptoms: {len(ALL_SYMPTOMS)}")
print(f"Total diseases: {len(DISEASES)}")

# ============================================================================
# GENERATE DATASET 1: symptoms_disease.csv
# ============================================================================
print("\n[1/7] Generating symptoms_disease.csv ...")

symptom_disease_rows = []
for disease_name, disease_data in DISEASES.items():
    base_symptoms = disease_data["symptoms"]
    # Generate ~120 rows per disease with symptom variations
    for _ in range(120):
        # Start with base symptoms (sometimes drop 1-2)
        selected = base_symptoms.copy()
        if len(selected) > 2 and random.random() > 0.3:
            drop_count = random.randint(0, min(2, len(selected) - 2))
            selected = random.sample(selected, len(selected) - drop_count)

        # Sometimes add 1-3 random noise symptoms
        if random.random() > 0.4:
            extra_count = random.randint(1, 3)
            extras = random.sample(
                [s for s in ALL_SYMPTOMS if s not in selected],
                min(extra_count, len(ALL_SYMPTOMS) - len(selected))
            )
            selected.extend(extras)

        random.shuffle(selected)

        # Pad to 17 symptom columns
        row = {"Disease": disease_name}
        for i in range(17):
            if i < len(selected):
                row[f"Symptom_{i+1}"] = selected[i]
            else:
                row[f"Symptom_{i+1}"] = np.nan
        symptom_disease_rows.append(row)

df_symptoms = pd.DataFrame(symptom_disease_rows)
df_symptoms = df_symptoms.sample(frac=1, random_state=SEED).reset_index(drop=True)
df_symptoms.to_csv(os.path.join(RAW_DIR, "symptoms_disease.csv"), index=False)
print(f"   ✓ {len(df_symptoms)} rows, {df_symptoms.shape[1]} columns")

# ============================================================================
# GENERATE DATASET 2: disease_description.csv
# ============================================================================
print("[2/7] Generating disease_description.csv ...")

df_desc = pd.DataFrame([
    {"Disease": name, "Description": data["description"]}
    for name, data in DISEASES.items()
])
df_desc.to_csv(os.path.join(RAW_DIR, "disease_description.csv"), index=False)
print(f"   ✓ {len(df_desc)} diseases")

# ============================================================================
# GENERATE DATASET 3: disease_precaution.csv
# ============================================================================
print("[3/7] Generating disease_precaution.csv ...")

precaution_rows = []
for name, data in DISEASES.items():
    row = {"Disease": name}
    for i in range(4):
        if i < len(data["precautions"]):
            row[f"Precaution_{i+1}"] = data["precautions"][i]
        else:
            row[f"Precaution_{i+1}"] = np.nan
    precaution_rows.append(row)

df_precaution = pd.DataFrame(precaution_rows)
df_precaution.to_csv(os.path.join(RAW_DIR, "disease_precaution.csv"), index=False)
print(f"   ✓ {len(df_precaution)} diseases")

# ============================================================================
# GENERATE DATASET 4: disease_medication.csv
# ============================================================================
print("[4/7] Generating disease_medication.csv ...")

medication_rows = []
for name, data in DISEASES.items():
    medication_rows.append({
        "Disease": name,
        "Medication": ", ".join(data["medication"]),
        "Diet": ", ".join(data["diet"]),
        "Exercise": ", ".join(data["exercise"]),
        "Specialist": data["specialist"]
    })

df_medication = pd.DataFrame(medication_rows)
df_medication.to_csv(os.path.join(RAW_DIR, "disease_medication.csv"), index=False)
print(f"   ✓ {len(df_medication)} diseases")

# ============================================================================
# GENERATE DATASET 5: synthetic_users.csv
# ============================================================================
print("[5/7] Generating synthetic_users.csv (500 users) ...")

LOCATIONS = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Bhopal", "Patna", "Indore", "Chandigarh", "Kochi",
    "New York", "London", "Dubai", "Singapore", "Sydney"
]

CHRONIC_CONDITIONS = [
    "None", "Diabetes", "Hypertension", "Asthma", "Thyroid disorder",
    "Heart disease", "Arthritis", "GERD", "Obesity", "Cholesterol"
]

GENDERS = ["Male", "Female", "Other"]

users = []
for uid in range(1, 501):
    age = int(np.clip(np.random.normal(40, 15), 18, 85))
    gender = random.choices(GENDERS, weights=[48, 48, 4])[0]

    # Older people more likely to have chronic conditions
    if age > 50:
        chronic = random.choices(CHRONIC_CONDITIONS, weights=[20, 15, 15, 10, 10, 10, 8, 5, 4, 3])[0]
    elif age > 35:
        chronic = random.choices(CHRONIC_CONDITIONS, weights=[50, 10, 10, 8, 7, 5, 4, 3, 2, 1])[0]
    else:
        chronic = random.choices(CHRONIC_CONDITIONS, weights=[75, 5, 3, 5, 3, 1, 2, 3, 2, 1])[0]

    users.append({
        "user_id": f"U{uid:04d}",
        "age": age,
        "gender": gender,
        "location": random.choice(LOCATIONS),
        "chronic_condition": chronic,
        "registration_date": (
            datetime(2024, 1, 1) + timedelta(days=random.randint(0, 540))
        ).strftime("%Y-%m-%d")
    })

df_users = pd.DataFrame(users)
df_users.to_csv(os.path.join(RAW_DIR, "synthetic_users.csv"), index=False)
print(f"   ✓ {len(df_users)} users")

# ============================================================================
# GENERATE DATASET 6: synthetic_interactions.csv
# ============================================================================
print("[6/7] Generating synthetic_interactions.csv (10,000 interactions) ...")

DISEASE_NAMES = list(DISEASES.keys())
INTERACTION_TYPES = ["search", "click", "view_details", "book_consultation", "save"]

interactions = []
for _ in range(10000):
    user = random.choice(df_users["user_id"].tolist())
    user_data = df_users[df_users["user_id"] == user].iloc[0]

    # Users with chronic conditions more likely to search related diseases
    if user_data["chronic_condition"] != "None":
        related = [d for d in DISEASE_NAMES if user_data["chronic_condition"].lower() in d.lower()]
        if related and random.random() > 0.4:
            disease = random.choice(related)
        else:
            disease = random.choice(DISEASE_NAMES)
    else:
        disease = random.choice(DISEASE_NAMES)

    # Interaction type distribution
    interaction_type = random.choices(
        INTERACTION_TYPES,
        weights=[35, 30, 20, 10, 5]
    )[0]

    # Rating (only for certain interaction types)
    if interaction_type in ["view_details", "book_consultation"]:
        rating = int(np.clip(np.random.normal(3.5, 1.0), 1, 5))
    else:
        rating = np.nan

    # Timestamp (random within 2024-2025)
    base_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 540))
    hour = random.choices(
        range(24),
        weights=[1, 1, 1, 1, 1, 2, 4, 6, 8, 8, 7, 6,
                 5, 5, 6, 7, 8, 8, 7, 5, 4, 3, 2, 1]
    )[0]
    timestamp = base_date.replace(hour=hour, minute=random.randint(0, 59))

    # Search query for search interactions
    disease_data = DISEASES[disease]
    if interaction_type == "search":
        symptoms_subset = random.sample(
            disease_data["symptoms"],
            min(random.randint(1, 3), len(disease_data["symptoms"]))
        )
        search_query = " ".join(symptoms_subset)
    else:
        search_query = ""

    interactions.append({
        "interaction_id": f"INT{len(interactions)+1:06d}",
        "user_id": user,
        "disease_id": disease,
        "interaction_type": interaction_type,
        "search_query": search_query,
        "rating": rating,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "session_duration_sec": random.randint(10, 600),
    })

df_interactions = pd.DataFrame(interactions)
df_interactions.to_csv(os.path.join(RAW_DIR, "synthetic_interactions.csv"), index=False)
print(f"   ✓ {len(df_interactions)} interactions")

# ============================================================================
# GENERATE DATASET 7: synthetic_reviews.csv
# ============================================================================
print("[7/7] Generating synthetic_reviews.csv (2,000 reviews) ...")

POSITIVE_TEMPLATES = [
    "The treatment recommendations were very helpful. I followed the {medication} regimen and felt better within a week.",
    "Great advice on managing {disease}. The diet suggestions especially worked well for me.",
    "I was diagnosed with {disease} and the exercise recommendations have significantly improved my condition.",
    "Very accurate symptom matching. The system correctly identified my {disease} condition.",
    "Excellent recommendation! The {specialist} was very knowledgeable about {disease}.",
    "The precautions suggested for {disease} were practical and easy to follow. Highly recommend.",
    "I've been dealing with {disease} for years. The personalized treatment plan was exactly what I needed.",
    "The medication {medication} prescribed for {disease} worked wonderfully. No side effects at all.",
    "Impressed by the accuracy of the diagnosis. Followed the recommended diet and saw improvement.",
    "The system helped me understand my {disease} better. The description and precautions were very informative.",
]

NEUTRAL_TEMPLATES = [
    "The recommendations for {disease} were okay. Some suggestions helped, others didn't.",
    "Average experience with {disease} treatment. The {medication} worked but took time.",
    "The system identified {disease} but some of the exercise recommendations were not suitable for my age.",
    "Decent recommendations overall. Could use more personalized suggestions for {disease}.",
    "The diet plan for {disease} was generic. Expected more specific recommendations.",
    "Consulted {specialist} as recommended. Treatment is ongoing, can't say much yet.",
]

NEGATIVE_TEMPLATES = [
    "The recommendations for {disease} didn't help at all. Need more accurate suggestions.",
    "Not satisfied with the {medication} recommendation. Had some side effects.",
    "The system couldn't properly identify my symptoms. Got wrong recommendations initially.",
    "Too generic advice for {disease}. Every patient is different and needs personalized care.",
    "The {specialist} recommendation was useful but the medication suggestion was incorrect.",
    "Disappointed with the exercise recommendations for {disease}. Not suitable for elderly patients.",
]

reviews = []
for _ in range(2000):
    user = random.choice(df_users["user_id"].tolist())
    disease = random.choice(DISEASE_NAMES)
    disease_data = DISEASES[disease]
    medication = random.choice(disease_data["medication"])
    specialist = disease_data["specialist"]

    # Rating determines sentiment
    rating = int(np.clip(np.random.normal(3.5, 1.2), 1, 5))

    if rating >= 4:
        template = random.choice(POSITIVE_TEMPLATES)
    elif rating >= 3:
        template = random.choice(NEUTRAL_TEMPLATES)
    else:
        template = random.choice(NEGATIVE_TEMPLATES)

    review_text = template.format(
        disease=disease,
        medication=medication,
        specialist=specialist
    )

    timestamp = (
        datetime(2024, 1, 1) + timedelta(days=random.randint(0, 540))
    ).strftime("%Y-%m-%d")

    reviews.append({
        "review_id": f"REV{len(reviews)+1:05d}",
        "user_id": user,
        "disease_id": disease,
        "rating": rating,
        "review_text": review_text,
        "timestamp": timestamp,
        "helpful_votes": random.randint(0, 50),
    })

df_reviews = pd.DataFrame(reviews)
df_reviews.to_csv(os.path.join(RAW_DIR, "synthetic_reviews.csv"), index=False)
print(f"   ✓ {len(df_reviews)} reviews")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("DATA GENERATION COMPLETE!")
print("=" * 60)
print(f"\nFiles saved to: {RAW_DIR}")
print(f"\n{'File':<35} {'Rows':>8} {'Cols':>6}")
print("-" * 51)
for fname, df in [
    ("symptoms_disease.csv", df_symptoms),
    ("disease_description.csv", df_desc),
    ("disease_precaution.csv", df_precaution),
    ("disease_medication.csv", df_medication),
    ("synthetic_users.csv", df_users),
    ("synthetic_interactions.csv", df_interactions),
    ("synthetic_reviews.csv", df_reviews),
]:
    print(f"  {fname:<33} {len(df):>8} {df.shape[1]:>6}")
print("-" * 51)
print(f"  {'TOTAL':<33} {sum(len(d) for d in [df_symptoms, df_desc, df_precaution, df_medication, df_users, df_interactions, df_reviews]):>8}")

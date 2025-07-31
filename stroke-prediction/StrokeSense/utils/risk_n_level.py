from sklearn.preprocessing import MinMaxScaler
#

# Convert age to age group.
# Age groups are defined as follows:
def age_to_age_group(age):
    """
    Convert age to age group.
    
    Args:
        age (int): Age of the individual.
        
    Returns:
        str: Age group category.
    """
    if age < 50:
        age_group = "Young (<49)"  # Changed to match model expectation
    elif 50 <= age < 65:
        age_group = "Middle (50-64)"
    else:
        age_group = "Older (65+)"
    return age_group 



## low risk (young and Middle (50-64)) + male, 
# moderate risk (young and Middle (50-64)) + female,
# high risk (Older (65+)	)+ female,
# very high risk (Older (65+)	) + male
def age_gender_to_risk(age_category, gender):
    if (age_category == "Young (<49)" or age_category == "Middle (50-64)") and gender == "Male":
        age_gender_risk = "Low Risk"
        
    elif (age_category == "Young (<49)" or age_category == "Middle (50-64)") and gender == "Female":
        age_gender_risk = "Moderate Risk"
        
    elif age_category == "Older (65+)" and gender == "Female":
        age_gender_risk = "High Risk"
    else:  # Older (65+) and Male
        age_gender_risk = "Very High Risk"

    return age_gender_risk


# Determine health risk level based on hypertension, heart disease, and diabetes
# If any of these conditions are present, health risk is Moderate Risk, otherwise Low Risk
def health_risk_level(hypertension, heart_disease, diabetes):
    if hypertension == 1 or heart_disease == 1 or diabetes == 1:
        health_risk = "Moderate Risk" 
    elif hypertension == 0 and heart_disease == 0 and diabetes == 0:
        health_risk = "Low Risk"
    else:
        health_risk = "Moderate Risk"
    return health_risk

work_map = {'Private':3, 'Employed':3, 'Self-employed':2, 'Unemployed':1}
married_map = {'Yes':2, 'No':1}
res_map = {'Urban':2, 'Rural':1}
health_map = {'High Risk':3, 'Moderate Risk':2, 'Low Risk':1}
def stress_level_category(work_type, marital_status, residence_type, health_risk):
    # Compute raw score
    score = 0
    score += work_map.get(work_type, 1)
    score += married_map.get(marital_status, 1)
    score += res_map.get(residence_type, 1)
    score += health_map.get(health_risk, 1)
    # Interaction boost
    if residence_type == 'Urban' and health_risk == 'High Risk':
        score += 1.5

    # --- Normalize score between 0 and 1 ---
    scaler = MinMaxScaler()
    scaler.fit([[1], [10]])  # assume score range (1 to 10)
    scaled_score = scaler.transform([[score]])[0][0]

    # --- Categorize ---
    if scaled_score < 0.3:
        stress_level =  "Low Stress"
    else:
        stress_level = "Moderate Stress"

    return stress_level




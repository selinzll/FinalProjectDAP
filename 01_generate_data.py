import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

COMPANY_PREFIXES = ['Tech', 'Global', 'Euro', 'Digital', 'Smart', 'Innov', 'Prime', 'Alpha', 'Beta', 'Gamma']
COMPANY_SUFFIXES = ['Systems', 'Solutions', 'Industries', 'Group', 'Corp', 'Technologies', 'Services', 'Enterprises', 'Ltd', 'GmbH']

def generate_company_name():
    return f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}"

SECTORS = [
    'Manufacturing', 'Retail', 'Healthcare', 'Finance', 'Logistics',
    'Agriculture and food', 'Construction', 'Energy and utilities',
    'Telecommunications', 'Tourism'
]

SIZES = ['Micro-size (1-9)', 'Small-size (10-49)', 'Medium-size (50-249)', 'Large-size (250+)']

COUNTRIES = [
    'Germany', 'France', 'Italy', 'Spain', 'Netherlands', 'Belgium',
    'Poland', 'Austria', 'Sweden', 'Portugal', 'Greece', 'Romania'
]

BUSINESS_AREAS = [
    'Inv_ProductDesign', 'Inv_ProjectMgmt', 'Inv_Operations',
    'Inv_Collaboration', 'Inv_InboundLogistics', 'Inv_MarketingSales',
    'Inv_Delivery', 'Inv_AdminHR', 'Inv_Procurement', 'Inv_Cybersecurity'
]

READINESS_FACTORS = [
    'Ready_NeedsIdentified', 'Ready_FinancialResources', 'Ready_ITInfra',
    'Ready_ICTSpecialists', 'Ready_ManagementLeadership', 'Ready_StaffSupport',
    'Ready_ProcessAdaptation', 'Ready_Servitisation', 'Ready_ClientMonitoring',
    'Ready_RiskConsidered'
]

BASIC_TECH = [
    'Tech_Connectivity', 'Tech_Website', 'Tech_WebForms',
    'Tech_LiveChats', 'Tech_ECommerce', 'Tech_EMarketing',
    'Tech_EGovernment', 'Tech_RemoteCollaboration', 'Tech_Intranet',
    'Tech_InfoMgmtSystems'
]

ADVANCED_TECH = [
    'AdvTech_Simulation', 'AdvTech_VRAR', 'AdvTech_CADCAM',
    'AdvTech_MES', 'AdvTech_IoT', 'AdvTech_Blockchain', 'AdvTech_3DPrinting'
]

TRAINING = [
    'Train_SkillAssessment', 'Train_TrainingPlan', 'Train_ShortCourses',
    'Train_LearningByDoing', 'Train_JobPlacements', 'Train_ExternalTraining',
    'Train_SubsidisedPrograms'
]

ENGAGEMENT = [
    'Engage_Awareness', 'Engage_TransparentComms', 'Engage_Monitoring',
    'Engage_StaffInvolvement', 'Engage_Autonomy', 'Engage_JobRedesign',
    'Engage_FlexibleWork', 'Engage_DigitalSupport'
]

DATA_MGMT = [
    'Data_Policy', 'Data_NotDigital', 'Data_StoredDigitally',
    'Data_Integrated', 'Data_RealTimeAccess', 'Data_Analytics',
    'Data_ExternalSources', 'Data_Dashboards'
]

SECURITY = [
    'Sec_Policy', 'Sec_ClientDataProtected', 'Sec_StaffTraining',
    'Sec_ThreatMonitoring', 'Sec_Backup', 'Sec_ContinuityPlan'
]

AI_TECH = [
    'AI_NLP', 'AI_ComputerVision', 'AI_AudioProcessing',
    'AI_Robotics', 'AI_BusinessIntelligence'
]

GREEN_PRACTICES = [
    'Green_BusinessModel', 'Green_ServiceProvision', 'Green_Products',
    'Green_Production', 'Green_Emissions', 'Green_EnergyGen',
    'Green_Materials', 'Green_Transport', 'Green_ConsumerBehavior',
    'Green_Paperless'
]

GREEN_POLICIES = [
    'GreenPol_Strategy', 'GreenPol_EMS', 'GreenPol_Procurement',
    'GreenPol_EnergyMonitoring', 'GreenPol_Recycling'
]

def generate_company_metadata(company_id):
    sector = random.choice(SECTORS)
    size = random.choice(SIZES)
    country = random.choice(COUNTRIES)

    sector_base = {
        'Finance': 68,
        'Telecommunications': 65,
        'Manufacturing': 52,
        'Agriculture and food': 48,
        'Construction': 45
    }
    base_score = sector_base.get(sector, 55)

    return {
        'Company_ID': f'COMP{company_id:04d}',
        'Company_Name': generate_company_name(),
        'Sector': sector,
        'Size': size,
        'Country': country,
        'Foundation_Year': random.randint(1980, 2020),
        'Staff_Count': random.randint(5, 500),
        'Base_Score_Factor': base_score
    }

def generate_dimension_scores(base_score, is_after=False):
    scores = {}
    company_variance = np.random.normal(0, 8)
    adjusted_base = np.clip(base_score + company_variance, 20, 85)

    strategy_inv_score = 0
    for area in BUSINESS_AREAS:
        prob = (adjusted_base / 100) * 0.7
        if is_after:
            prob = min(prob + 0.15, 0.95)
        scores[area] = 1 if random.random() < prob else 0
        strategy_inv_score += scores[area]

    strategy_ready_score = 0
    for factor in READINESS_FACTORS:
        prob = (adjusted_base / 100) * 0.65
        if is_after:
            prob = min(prob + 0.20, 0.95)
        scores[factor] = 1 if random.random() < prob else 0
        strategy_ready_score += scores[factor]

    strategy_total = ((strategy_inv_score + strategy_ready_score) / 20) * 100

    readiness_basic_score = 0
    for tech in BASIC_TECH:
        prob = (adjusted_base / 100) * 0.75
        if is_after:
            prob = min(prob + 0.12, 0.98)
        scores[tech] = 1 if random.random() < prob else 0
        readiness_basic_score += scores[tech]

    readiness_adv_score = 0
    for tech in ADVANCED_TECH:
        base_level = int((adjusted_base / 100) * 3.5)
        if is_after:
            base_level = min(base_level + random.randint(1, 2), 5)
        scores[tech] = max(0, base_level + random.randint(-1, 1))
        readiness_adv_score += scores[tech]

    readiness_total = ((readiness_basic_score / 10) * 50 + (readiness_adv_score / 35) * 50)

    human_train_score = 0
    for item in TRAINING:
        prob = (adjusted_base / 100) * 0.60
        if is_after:
            prob = min(prob + 0.25, 0.95)
        scores[item] = 1 if random.random() < prob else 0
        human_train_score += scores[item]

    human_engage_score = 0
    for item in ENGAGEMENT:
        prob = (adjusted_base / 100) * 0.55
        if is_after:
            prob = min(prob + 0.22, 0.95)
        scores[item] = 1 if random.random() < prob else 0
        human_engage_score += scores[item]

    human_total = ((human_train_score + human_engage_score) / 15) * 100

    data_mgmt_score = 0
    for item in DATA_MGMT:
        if item == 'Data_NotDigital':
            prob = max(0.05, 0.30 - (adjusted_base / 100) * 0.25)
            if is_after:
                prob = max(0.02, prob - 0.15)
        else:
            prob = (adjusted_base / 100) * 0.70
            if is_after:
                prob = min(prob + 0.18, 0.95)
        scores[item] = 1 if random.random() < prob else 0
        if item != 'Data_NotDigital':
            data_mgmt_score += scores[item]
        else:
            data_mgmt_score -= scores[item]

    security_score = 0
    for item in SECURITY:
        prob = (adjusted_base / 100) * 0.68
        if is_after:
            prob = min(prob + 0.20, 0.96)
        scores[item] = 1 if random.random() < prob else 0
        security_score += scores[item]

    data_total = ((max(0, data_mgmt_score) / 7) * 60 + (security_score / 6) * 40)

    ai_score = 0
    for tech in AI_TECH:
        base_level = int((adjusted_base / 100) * 3)
        if is_after:
            base_level = min(base_level + random.randint(1, 3), 5)
        scores[tech] = max(0, base_level + random.randint(-1, 1))
        ai_score += scores[tech]

    ai_total = (ai_score / 25) * 100

    green_prac_score = 0
    for item in GREEN_PRACTICES:
        prob = (adjusted_base / 100) * 0.50
        if is_after:
            prob = min(prob + 0.20, 0.85)
        scores[item] = 1 if random.random() < prob else 0
        green_prac_score += scores[item]

    green_pol_score = 0
    for item in GREEN_POLICIES:
        base_level = int((adjusted_base / 100) * 1.5)
        if is_after:
            base_level = min(base_level + 1, 2)
        scores[item] = max(0, min(2, base_level + random.choice([-1, 0, 1])))
        green_pol_score += scores[item]

    green_total = ((green_prac_score / 10) * 60 + (green_pol_score / 10) * 40)

    scores['DimScore_Strategy'] = round(strategy_total, 2)
    scores['DimScore_Readiness'] = round(readiness_total, 2)
    scores['DimScore_HumanCentric'] = round(human_total, 2)
    scores['DimScore_DataMgmt'] = round(data_total, 2)
    scores['DimScore_AutomationAI'] = round(ai_total, 2)
    scores['DimScore_GreenDigital'] = round(green_total, 2)

    scores['Overall_Maturity'] = round(np.mean([
        scores['DimScore_Strategy'],
        scores['DimScore_Readiness'],
        scores['DimScore_HumanCentric'],
        scores['DimScore_DataMgmt'],
        scores['DimScore_AutomationAI'],
        scores['DimScore_GreenDigital']
    ]), 2)

    if scores['Overall_Maturity'] < 45:
        scores['Maturity_Level'] = 'Novice'
    elif scores['Overall_Maturity'] < 75:
        scores['Maturity_Level'] = 'Competent'
    else:
        scores['Maturity_Level'] = 'Leader'

    return scores

def generate_datasets(n_companies=1000):
    print("Generating Digital Maturity Assessment datasets...")
    print(f"Number of companies: {n_companies}")
    print("=" * 60)

    before_data = []
    after_data = []

    for i in range(1, n_companies + 1):
        if i % 100 == 0:
            print(f"Processing company {i}/{n_companies}...")

        metadata = generate_company_metadata(i)

        before_scores = generate_dimension_scores(metadata['Base_Score_Factor'], is_after=False)
        before_record = {**metadata, **before_scores}
        before_record['Assessment_Date'] = (datetime.now() - timedelta(days=random.randint(180, 365))).strftime('%Y-%m-%d')
        before_data.append(before_record)

        after_scores = generate_dimension_scores(metadata['Base_Score_Factor'], is_after=True)
        after_record = {**metadata, **after_scores}
        after_record['Assessment_Date'] = datetime.now().strftime('%Y-%m-%d')
        after_data.append(after_record)

    df_before = pd.DataFrame(before_data)
    df_after = pd.DataFrame(after_data)

    improvements = df_after['Overall_Maturity'] - df_before['Overall_Maturity']

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"\nBEFORE Assessment:")
    print(f"  Mean Maturity: {df_before['Overall_Maturity'].mean():.2f}")
    print(f"  Std Dev: {df_before['Overall_Maturity'].std():.2f}")
    print(f"  Min: {df_before['Overall_Maturity'].min():.2f}")
    print(f"  Max: {df_before['Overall_Maturity'].max():.2f}")

    print(f"\nAFTER Assessment:")
    print(f"  Mean Maturity: {df_after['Overall_Maturity'].mean():.2f}")
    print(f"  Std Dev: {df_after['Overall_Maturity'].std():.2f}")
    print(f"  Min: {df_after['Overall_Maturity'].min():.2f}")
    print(f"  Max: {df_after['Overall_Maturity'].max():.2f}")

    print(f"\nIMPROVEMENT Metrics:")
    print(f"  Mean Growth: {improvements.mean():.2f} points")
    print(f"  Median Growth: {improvements.median():.2f} points")
    print(f"  Std Dev: {improvements.std():.2f}")
    print(f"  Min Growth: {improvements.min():.2f}")
    print(f"  Max Growth: {improvements.max():.2f}")

    from scipy import stats
    t_stat, p_value = stats.ttest_rel(df_after['Overall_Maturity'], df_before['Overall_Maturity'])
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Significant: {'YES' if p_value < 0.001 else 'NO'} (α=0.001)")

    print(f"\nMATURITY LEVEL Distribution (After):")
    print(df_after['Maturity_Level'].value_counts())

    return df_before, df_after

if __name__ == "__main__":
    df_before, df_after = generate_datasets(n_companies=1000)

    print("\n" + "=" * 60)
    print("Saving to Excel files...")

    df_before.to_excel('/Users/zseli/PycharmProjects/FinalProjectDAP/rawdma_before.xlsx', index=False, engine='openpyxl')
    print("✓ Saved: rawdma_before.xlsx")

    df_after.to_excel('/Users/zseli/PycharmProjects/FinalProjectDAP/rawdma_after.xlsx', index=False, engine='openpyxl')
    print("✓ Saved: rawdma_after.xlsx")

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
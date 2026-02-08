import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_and_merge_data():
    print("=" * 70)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("=" * 70)
    
    df_before = pd.read_excel('/Users/zseli/PycharmProjects/FinalProjectDAP/rawdma_before.xlsx')
    df_after = pd.read_excel('/Users/zseli/PycharmProjects/FinalProjectDAP/rawdma_after.xlsx')
    
    print(f"\n✓ Loaded BEFORE dataset: {df_before.shape[0]} rows, {df_before.shape[1]} columns")
    print(f"✓ Loaded AFTER dataset: {df_after.shape[0]} rows, {df_after.shape[1]} columns")
    
    df_merged = df_before.merge(
        df_after,
        on='Company_ID',
        suffixes=('_Before', '_After')
    )
    
    print(f"✓ Merged dataset: {df_merged.shape[0]} companies matched")
    
    dimensions = ['Strategy', 'Readiness', 'HumanCentric', 'DataMgmt', 'AutomationAI', 'GreenDigital']
    
    for dim in dimensions:
        df_merged[f'{dim}_Delta'] = (
            df_merged[f'DimScore_{dim}_After'] - df_merged[f'DimScore_{dim}_Before']
        )
    
    df_merged['Overall_Delta'] = (
        df_merged['Overall_Maturity_After'] - df_merged['Overall_Maturity_Before']
    )
    
    print("\n✓ Calculated Delta (After - Before) for all dimensions")
    
    return df_merged, dimensions

def exploratory_analysis(df, dimensions):
    print("\n" + "=" * 70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    print("\n1. OVERALL MATURITY SCORES")
    print("-" * 70)
    print(f"{'Metric':<20} {'Before':<15} {'After':<15} {'Delta':<15}")
    print("-" * 70)
    print(f"{'Mean':<20} {df['Overall_Maturity_Before'].mean():<15.2f} "
          f"{df['Overall_Maturity_After'].mean():<15.2f} "
          f"{df['Overall_Delta'].mean():<15.2f}")
    print(f"{'Median':<20} {df['Overall_Maturity_Before'].median():<15.2f} "
          f"{df['Overall_Maturity_After'].median():<15.2f} "
          f"{df['Overall_Delta'].median():<15.2f}")
    print(f"{'Std Dev':<20} {df['Overall_Maturity_Before'].std():<15.2f} "
          f"{df['Overall_Maturity_After'].std():<15.2f} "
          f"{df['Overall_Delta'].std():<15.2f}")
    print(f"{'Min':<20} {df['Overall_Maturity_Before'].min():<15.2f} "
          f"{df['Overall_Maturity_After'].min():<15.2f} "
          f"{df['Overall_Delta'].min():<15.2f}")
    print(f"{'Max':<20} {df['Overall_Maturity_Before'].max():<15.2f} "
          f"{df['Overall_Maturity_After'].max():<15.2f} "
          f"{df['Overall_Delta'].max():<15.2f}")
    
    t_stat, p_value = stats.ttest_rel(
        df['Overall_Maturity_After'],
        df['Overall_Maturity_Before']
    )
    
    print(f"\n2. STATISTICAL SIGNIFICANCE TEST (Paired t-test)")
    print("-" * 70)
    print(f"H0: No difference between Before and After")
    print(f"H1: Significant improvement After intervention")
    print(f"\nt-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.2e}")
    print(f"Result: {'REJECT H0 - Significant improvement' if p_value < 0.001 else 'FAIL TO REJECT H0'}")
    
    cohens_d = (df['Overall_Maturity_After'].mean() - df['Overall_Maturity_Before'].mean()) / df['Overall_Maturity_Before'].std()
    print(f"Cohen's d (effect size): {cohens_d:.4f}")
    if cohens_d > 0.8:
        effect_interpretation = "Large effect"
    elif cohens_d > 0.5:
        effect_interpretation = "Medium effect"
    else:
        effect_interpretation = "Small effect"
    print(f"Interpretation: {effect_interpretation}")
    
    print(f"\n3. DIMENSION-LEVEL IMPROVEMENTS")
    print("-" * 70)
    print(f"{'Dimension':<20} {'Before':<12} {'After':<12} {'Delta':<12} {'% Change':<12}")
    print("-" * 70)
    
    dim_stats = []
    for dim in dimensions:
        before_mean = df[f'DimScore_{dim}_Before'].mean()
        after_mean = df[f'DimScore_{dim}_After'].mean()
        delta_mean = df[f'{dim}_Delta'].mean()
        pct_change = (delta_mean / before_mean) * 100 if before_mean > 0 else 0
        
        print(f"{dim:<20} {before_mean:<12.2f} {after_mean:<12.2f} {delta_mean:<12.2f} {pct_change:<12.1f}%")
        dim_stats.append({
            'Dimension': dim,
            'Delta': delta_mean,
            'Pct_Change': pct_change
        })
    
    dim_stats_df = pd.DataFrame(dim_stats).sort_values('Delta', ascending=False)
    
    print(f"\nTop improvement dimension: {dim_stats_df.iloc[0]['Dimension']} (+{dim_stats_df.iloc[0]['Delta']:.2f} points)")
    print(f"Lowest improvement dimension: {dim_stats_df.iloc[-1]['Dimension']} (+{dim_stats_df.iloc[-1]['Delta']:.2f} points)")
    
    print(f"\n4. SECTOR-SPECIFIC PERFORMANCE")
    print("-" * 70)
    sector_stats = df.groupby('Sector_Before').agg({
        'Overall_Maturity_Before': 'mean',
        'Overall_Maturity_After': 'mean',
        'Overall_Delta': 'mean'
    }).sort_values('Overall_Delta', ascending=False)
    
    print(f"{'Sector':<25} {'Before':<12} {'After':<12} {'Growth':<12}")
    print("-" * 70)
    for sector, row in sector_stats.iterrows():
        print(f"{sector:<25} {row['Overall_Maturity_Before']:<12.2f} "
              f"{row['Overall_Maturity_After']:<12.2f} {row['Overall_Delta']:<12.2f}")
    
    return dim_stats_df, sector_stats

def correlation_analysis(df, dimensions):
    print("\n" + "=" * 70)
    print("STEP 3: CORRELATION ANALYSIS")
    print("=" * 70)
    
    before_cols = [f'DimScore_{dim}_Before' for dim in dimensions]
    corr_df = df[before_cols + ['Overall_Maturity_After']].copy()
    corr_df.columns = dimensions + ['Overall_After']
    
    corr_matrix = corr_df.corr()
    
    print("\nCorrelation of BEFORE dimensions with AFTER overall maturity:")
    print("-" * 70)
    correlations = corr_matrix['Overall_After'][:-1].sort_values(ascending=False)
    
    for dim, corr_value in correlations.items():
        print(f"{dim:<20} r = {corr_value:.4f}")
    
    print(f"\nStrongest predictor: {correlations.index[0]} (r = {correlations.iloc[0]:.4f})")
    
    print("\n\nInter-dimension correlations (BEFORE assessment):")
    print("-" * 70)
    
    inter_corr = corr_df[dimensions].corr()
    
    strong_corr = []
    for i, dim1 in enumerate(dimensions):
        for j, dim2 in enumerate(dimensions):
            if i < j:
                corr_val = inter_corr.loc[dim1, dim2]
                strong_corr.append((dim1, dim2, corr_val))
    
    strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\nTop 5 strongest dimension pairs:")
    for dim1, dim2, corr_val in strong_corr[:5]:
        print(f"  {dim1} <-> {dim2}: r = {corr_val:.4f}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df.corr(), annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Dimension Scores vs Overall Maturity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/Users/zseli/PycharmProjects/FinalProjectDAP/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved correlation heatmap: correlation_heatmap.png")
    
    return correlations, inter_corr

def train_ml_model(df, dimensions):
    print("\n" + "=" * 70)
    print("STEP 4: MACHINE LEARNING - LINEAR REGRESSION")
    print("=" * 70)
    
    X_cols = [f'DimScore_{dim}_Before' for dim in dimensions]
    X = df[X_cols].values
    y = df['Overall_Maturity_After'].values
    
    print(f"\n✓ Features (X): {len(X_cols)} dimension scores from BEFORE assessment")
    print(f"✓ Target (y): Overall maturity from AFTER assessment")
    print(f"✓ Sample size: {len(X)} companies")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("\n✓ Model trained successfully")
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print("\n" + "-" * 70)
    print("MODEL PERFORMANCE METRICS")
    print("-" * 70)
    print(f"{'Metric':<25} {'Train':<20} {'Test':<20}")
    print("-" * 70)
    print(f"{'R² Score':<25} {train_r2:<20.4f} {test_r2:<20.4f}")
    print(f"{'RMSE':<25} {train_rmse:<20.4f} {test_rmse:<20.4f}")
    print(f"{'MAE (Test only)':<25} {'':<20} {test_mae:<20.4f}")
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"\n5-Fold Cross-Validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE (Model Coefficients)")
    print("-" * 70)
    print(f"{'Dimension':<25} {'Coefficient':<20} {'Interpretation':<30}")
    print("-" * 70)
    
    coef_df = pd.DataFrame({
        'Dimension': dimensions,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=False)
    
    for _, row in coef_df.iterrows():
        dim = row['Dimension']
        coef = row['Coefficient']
        interpretation = f"+1 point → +{coef:.3f} overall"
        print(f"{dim:<25} {coef:<20.4f} {interpretation:<30}")
    
    print(f"\nIntercept: {model.intercept_:.4f}")
    
    print(f"\n{'='*70}")
    print("KEY INSIGHTS:")
    print(f"{'='*70}")
    print(f"1. Most impactful dimension: {coef_df.iloc[0]['Dimension']}")
    print(f"   → Improving this by 10 points increases overall maturity by {coef_df.iloc[0]['Coefficient'] * 10:.2f} points")
    print(f"\n2. Model explains {test_r2*100:.1f}% of variance in post-intervention maturity")
    print(f"3. Average prediction error: ±{test_rmse:.2f} maturity points")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Overall Maturity (After)', fontsize=11)
    plt.ylabel('Predicted Overall Maturity', fontsize=11)
    plt.title(f'Model Predictions (R² = {test_r2:.3f})', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
    plt.barh(coef_df['Dimension'], coef_df['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Coefficient Value', fontsize=11)
    plt.title('Feature Importance (Coefficients)', fontsize=12, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/users/zseli/PycharmProjects/FinalProjectDAP/ml_model_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved ML results visualization: ml_model_results.png")
    
    return model, coef_df, test_r2, test_rmse

def generate_insights_report(df, dimensions, dim_stats, sector_stats, correlations, coef_df):
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING INSIGHTS REPORT")
    print("=" * 70)
    
    report = []
    report.append("="*80)
    report.append("DIGITAL MATURITY ASSESSMENT - COMPREHENSIVE INSIGHTS REPORT")
    report.append("="*80)
    report.append("")
    
    report.append("EXECUTIVE SUMMARY")
    report.append("-"*80)
    avg_improvement = df['Overall_Delta'].mean()
    pct_improvement = (avg_improvement / df['Overall_Maturity_Before'].mean()) * 100
    companies_improved = len(df[df['Overall_Delta'] > 0])
    
    report.append(f"Total Companies Assessed: {len(df)}")
    report.append(f"Companies Showing Improvement: {companies_improved} ({companies_improved/len(df)*100:.1f}%)")
    report.append(f"Average Maturity Growth: {avg_improvement:.2f} points ({pct_improvement:.1f}% increase)")
    report.append(f"Baseline (Before) Mean: {df['Overall_Maturity_Before'].mean():.2f}")
    report.append(f"Post-Intervention (After) Mean: {df['Overall_Maturity_After'].mean():.2f}")
    report.append("")
    
    report.append("TOP 10 PERFORMERS (Highest Growth)")
    report.append("-"*80)
    top10 = df.nlargest(10, 'Overall_Delta')[
        ['Company_Name_Before', 'Sector_Before', 'Overall_Maturity_Before', 
         'Overall_Maturity_After', 'Overall_Delta']
    ]
    for idx, row in top10.iterrows():
        report.append(f"{row['Company_Name_Before']:<40} | {row['Sector_Before']:<20} | "
                     f"Growth: +{row['Overall_Delta']:.2f} ({row['Overall_Maturity_Before']:.1f} → {row['Overall_Maturity_After']:.1f})")
    report.append("")
    
    report.append("COMPANIES NEEDING ATTENTION (Lowest Growth / Decline)")
    report.append("-"*80)
    bottom10 = df.nsmallest(10, 'Overall_Delta')[
        ['Company_Name_Before', 'Sector_Before', 'Overall_Maturity_Before', 
         'Overall_Maturity_After', 'Overall_Delta']
    ]
    for idx, row in bottom10.iterrows():
        report.append(f"{row['Company_Name_Before']:<40} | {row['Sector_Before']:<20} | "
                     f"Change: {row['Overall_Delta']:+.2f} ({row['Overall_Maturity_Before']:.1f} → {row['Overall_Maturity_After']:.1f})")
    report.append("")
    
    report.append("DIMENSION-LEVEL INSIGHTS")
    report.append("-"*80)
    report.append(f"Highest Growth Dimension: {dim_stats.iloc[0]['Dimension']} (+{dim_stats.iloc[0]['Delta']:.2f})")
    report.append(f"Priority Investment Area: {dim_stats.iloc[-1]['Dimension']} (+{dim_stats.iloc[-1]['Delta']:.2f})")
    report.append("")
    
    report.append("SECTOR-SPECIFIC FINDINGS")
    report.append("-"*80)
    best_sector = sector_stats.index[0]
    worst_sector = sector_stats.index[-1]
    report.append(f"Best Performing Sector: {best_sector} (+{sector_stats.iloc[0]['Overall_Delta']:.2f})")
    report.append(f"Underperforming Sector: {worst_sector} (+{sector_stats.iloc[-1]['Overall_Delta']:.2f})")
    report.append("")
    
    report.append("PREDICTIVE MODEL INSIGHTS")
    report.append("-"*80)
    report.append(f"Most Predictive Dimension: {correlations.index[0]} (r={correlations.iloc[0]:.3f})")
    report.append(f"Highest Impact Lever: {coef_df.iloc[0]['Dimension']} (β={coef_df.iloc[0]['Coefficient']:.3f})")
    report.append("")
    
    report.append("STRATEGIC RECOMMENDATIONS")
    report.append("-"*80)
    report.append(f"1. Prioritize {coef_df.iloc[0]['Dimension']} investments for maximum ROI")
    report.append(f"2. Benchmark against {best_sector} sector best practices")
    report.append(f"3. Provide targeted support to {worst_sector} companies")
    report.append(f"4. Focus on {dim_stats.iloc[-1]['Dimension']} to close capability gaps")
    report.append("")
    report.append("="*80)
    
    report_text = '\n'.join(report)
    with open('/Users/zseli/PycharmProjects/FinalProjectDAP/insights_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n✓ Saved insights report: insights_report.txt")
    print("\nReport Preview:")
    print(report_text)
    
    return report_text

def main():
    print("\n" + "="*70)
    print("DIGITAL MATURITY ASSESSMENT - ANALYSIS & ML PIPELINE")
    print("="*70)
    print("Based on EU Open DMAT Framework")
    print("="*70 + "\n")
    
    df, dimensions = load_and_merge_data()
    dim_stats, sector_stats = exploratory_analysis(df, dimensions)
    correlations, inter_corr = correlation_analysis(df, dimensions)
    model, coef_df, r2, rmse = train_ml_model(df, dimensions)
    insights = generate_insights_report(df, dimensions, dim_stats, sector_stats, correlations, coef_df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - FILES GENERATED")
    print("="*70)
    print("✓ correlation_heatmap.png - Visualization of dimension correlations")
    print("✓ ml_model_results.png - Model performance and feature importance")
    print("✓ insights_report.txt - Comprehensive business insights")
    print("\n" + "="*70)
    
    return df, model, coef_df

if __name__ == "__main__":
    df, model, coef_df = main()
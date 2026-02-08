import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import io

st.set_page_config(
    page_title="Digital Maturity Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-color: #0E1117; 
        --card-bg: #1E1E1E;
        --text-color: #E0E0E0;
        --accent-color: #D32F2F;
        --secondary-text: #A0A0A0;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }

    .stApp {
        background-color: var(--bg-color);
    }

    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
    }

    h1 {
        font-weight: 700;
        font-size: 2.2rem !important;
        margin-bottom: 1.5rem;
    }

    h2 {
        font-weight: 600;
        font-size: 1.5rem !important;
        border-bottom: 1px solid #333;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #161616;
        border-right: 1px solid #333;
    }

    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] div, 
    section[data-testid="stSidebar"] label {
        color: #E0E0E0 !important;
    }

    .stRadio label, .stCheckbox label, .stSlider label, .stSelectbox label {
        color: #E0E0E0 !important;
    }
    div[data-baseweb="select"] span {
        color: #E0E0E0 !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: #E0E0E0 !important;
    }

    div[data-testid="stMetric"] {
        background-color: var(--card-bg);
        padding: 20px;
        border-radius: 4px;
        border-left: 4px solid var(--accent-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--secondary-text) !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #FFFFFF !important;
    }

    .stButton > button {
        background-color: #333;
        color: white !important;
        border: 1px solid #555;
        font-weight: 500;
        border-radius: 4px;
    }

    .stButton > button:hover {
        background-color: var(--accent-color);
        border-color: var(--accent-color);
    }

    .js-plotly-plot {
        background-color: #1E1E1E;
        border-radius: 5px;
        padding: 10px;
    }

    div[data-testid="stDataFrame"] {
        background-color: var(--card-bg);
    }
</style>
""", unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def load_data():
    try:
        df_before = pd.read_excel('rawdma_before.xlsx')
        df_after = pd.read_excel('rawdma_after.xlsx')

        df_before['Company_ID'] = df_before['Company_ID'].astype(str).str.strip()
        df_after['Company_ID'] = df_after['Company_ID'].astype(str).str.strip()

        df = df_before.merge(df_after, on='Company_ID', suffixes=('_Before', '_After'))

        df_obj = df.select_dtypes(['object'])
        df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

        dimensions = ['Strategy', 'Readiness', 'HumanCentric', 'DataMgmt', 'AutomationAI', 'GreenDigital']

        for dim in dimensions:
            col_before = f'DimScore_{dim}_Before'
            col_after = f'DimScore_{dim}_After'

            df[col_before] = pd.to_numeric(df[col_before], errors='coerce').fillna(0)
            df[col_after] = pd.to_numeric(df[col_after], errors='coerce').fillna(0)
            df[f'{dim}_Delta'] = df[col_after] - df[col_before]

        df['Overall_Maturity_Before'] = pd.to_numeric(df['Overall_Maturity_Before'], errors='coerce').fillna(0)
        df['Overall_Maturity_After'] = pd.to_numeric(df['Overall_Maturity_After'], errors='coerce').fillna(0)
        df['Overall_Delta'] = df['Overall_Maturity_After'] - df['Overall_Maturity_Before']

        return df, dimensions
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), []


@st.cache(allow_output_mutation=True)
def train_model(df, dimensions):
    if df.empty: return None
    X = df[[f'DimScore_{dim}_Before' for dim in dimensions]].values
    y = df['Overall_Maturity_After'].values
    model = LinearRegression()
    model.fit(X, y)
    return model


def to_excel_download(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()


df, dimensions = load_data()
ml_model = train_model(df, dimensions)

st.sidebar.markdown("## Digital Maturity Assessment")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "",
    ["Market Overview", "Company Report", "Predictive Tool", "Data Export"]
)

st.sidebar.markdown("---")

sector_filter = "All Sectors"
size_filter = "All Sizes"
company_name = None

if page == "Market Overview":
    st.sidebar.markdown("### Filters")
    sector_filter = st.sidebar.selectbox(
        "Sector",
        ["All Sectors"] + sorted(df['Sector_Before'].unique().tolist())
    )

    size_filter = st.sidebar.selectbox(
        "Company Size",
        ["All Sizes"] + sorted(df['Size_Before'].unique().tolist())
    )

elif page == "Company Report":
    st.sidebar.markdown("### Select Company")
    company_name = st.sidebar.selectbox(
        "Company",
        sorted(df['Company_Name_Before'].unique().tolist())
    )

if page == "Market Overview":
    df_filtered = df.copy()
    if sector_filter != "All Sectors":
        df_filtered = df_filtered[df_filtered['Sector_Before'] == sector_filter]
    if size_filter != "All Sizes":
        df_filtered = df_filtered[df_filtered['Size_Before'] == size_filter]

    st.markdown(f"# Digital Maturity Landscape")
    if sector_filter != "All Sectors" or size_filter != "All Sizes":
        filters_text = []
        if sector_filter != "All Sectors":
            filters_text.append(sector_filter)
        if size_filter != "All Sizes":
            filters_text.append(size_filter)
        st.markdown(f"### {' | '.join(filters_text)}")

    st.markdown(
        f"Analysis of **{len(df_filtered)} organizations** showing digital transformation "
        f"progress from baseline to post-intervention assessments."
    )

    st.markdown("---")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Average Maturity", f"{df_filtered['Overall_Maturity_After'].mean():.1f}",
                  f"+{df_filtered['Overall_Delta'].mean():.1f}")

    with k2:
        st.metric("Average Growth", f"{df_filtered['Overall_Delta'].mean():.1f}")

    with k3:
        leaders = len(df_filtered[df_filtered['Maturity_Level_After'] == 'Leader'])
        st.metric("Industry Leaders", leaders)

    with k4:
        novices = len(df_filtered[df_filtered['Maturity_Level_After'] == 'Novice'])
        st.metric("Early Stage", novices)

    st.markdown("---")

    st.markdown("## Correlation Analysis")

    col_chart, col_text = st.columns([2, 1])

    with col_chart:
        corr_cols = [f'DimScore_{dim}_Before' for dim in dimensions]
        correlations = df_filtered[corr_cols].corrwith(df_filtered['Overall_Maturity_After'])

        corr_df = pd.DataFrame({
            'Dimension': dimensions,
            'Correlation': correlations.values
        }).sort_values('Correlation', ascending=True)

        fig = px.bar(
            corr_df,
            x='Correlation',
            y='Dimension',
            orientation='h',
            text='Correlation',
            title="Impact of Baseline Dimensions on Final Success"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            height=400,
            font=dict(family="Inter", size=12, color="white"),
            xaxis_title="Correlation Strength (0 to 1)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_text:
        st.markdown("""
        ### Drivers of Success

        This chart shows which baseline dimensions are the strongest predictors of high final maturity.

        **Interpretation:**
        - **Longer Bars:** Stronger relationship.
        - **High Correlation:** Investing in this area early yields better long-term results.
        """)

        top_dim = corr_df.iloc[-1]
        st.success(f"""
        **Top Predictor:**

        **{top_dim['Dimension']}**
        (Correlation: {top_dim['Correlation']:.2f})
        """)

    st.markdown("---")
    st.markdown("## Performance Rankings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Top 10 Performers")
        top10 = df_filtered.nlargest(10, 'Overall_Delta')[
            ['Company_Name_Before', 'Sector_Before', 'Overall_Maturity_Before',
             'Overall_Maturity_After', 'Overall_Delta']
        ].copy()
        top10.columns = ['Company', 'Sector', 'Before', 'After', 'Growth']

        st.dataframe(top10.style.format("{:.1f}", subset=['Before', 'After', 'Growth']))

    with col2:
        st.markdown("### Bottom 10 Performers")
        bottom10 = df_filtered.nsmallest(10, 'Overall_Delta')[
            ['Company_Name_Before', 'Sector_Before', 'Overall_Maturity_Before',
             'Overall_Maturity_After', 'Overall_Delta']
        ].copy()
        bottom10.columns = ['Company', 'Sector', 'Before', 'After', 'Growth']

        st.dataframe(bottom10.style.format("{:.1f}", subset=['Before', 'After', 'Growth']))

elif page == "Company Report":
    company_row = df[df['Company_Name_Before'] == company_name].iloc[0]
    sector = company_row['Sector_Before']
    sector_df = df[df['Sector_Before'] == sector]

    st.markdown(f"<h1 style='text-align: center;'>{company_name}</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align: center;'>{sector} | {company_row['Size_Before']} | {company_row['Country_Before']}</p>",
        unsafe_allow_html=True)

    st.markdown("---")

    k1, k2, k3 = st.columns(3)

    with k1:
        st.metric(
            "Overall Maturity",
            f"{company_row['Overall_Maturity_After']:.1f}",
            f"+{company_row['Overall_Delta']:.1f}"
        )

    with k2:
        st.metric(
            "Maturity Level",
            company_row['Maturity_Level_After'],
            ""
        )

    with k3:
        percentile = (df['Overall_Maturity_After'] < company_row['Overall_Maturity_After']).sum() / len(df) * 100
        st.metric(
            "Percentile Rank",
            f"{percentile:.0f}th",
            ""
        )

    st.markdown("---")
    st.markdown("## Performance by Dimension")

    dim_data = []
    for dim in dimensions:
        dim_data.append({
            'Dimension': dim,
            'Before': round(company_row[f'DimScore_{dim}_Before'], 1),
            'After': round(company_row[f'DimScore_{dim}_After'], 1),
            'Growth': round(company_row[f'{dim}_Delta'], 1)
        })

    dim_df = pd.DataFrame(dim_data)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(dim_df)

        best_dim = dim_df.loc[dim_df['Growth'].idxmax(), 'Dimension']
        worst_dim = dim_df.loc[dim_df['Growth'].idxmin(), 'Dimension']

        st.success(f"**Strongest Growth:** {best_dim}")
        st.warning(f"**Needs Attention:** {worst_dim}")

    with col2:
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Before',
            x=dim_df['Dimension'],
            y=dim_df['Before'],
            marker_color='#D32F2F'
        ))

        fig.add_trace(go.Bar(
            name='After',
            x=dim_df['Dimension'],
            y=dim_df['After'],
            marker_color='#2E7D32'
        ))

        fig.update_layout(
            barmode='group',
            title="Dimension Scores: Before vs After",
            yaxis_title="Score",
            height=350,
            template="plotly_dark",
            paper_bgcolor='#1E1E1E',
            plot_bgcolor='#1E1E1E',
            font=dict(family="Inter", size=12, color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## Sector Benchmarking")

    categories = dimensions

    company_values = [company_row[f'DimScore_{dim}_After'] for dim in dimensions]
    sector_values = [sector_df[f'DimScore_{dim}_After'].mean() for dim in dimensions]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=company_values,
        theta=categories,
        fill='toself',
        name=company_name,
        line_color='#D32F2F',
        fillcolor='rgba(211, 47, 47, 0.2)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=sector_values,
        theta=categories,
        fill='toself',
        name=f'{sector} Average',
        line_color='#1976D2',
        fillcolor='rgba(25, 118, 210, 0.2)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=False
            ),
            bgcolor='#1E1E1E'
        ),
        showlegend=True,
        title="Company vs Sector Average",
        height=500,
        template="plotly_dark",
        paper_bgcolor='#1E1E1E',
        font=dict(family="Inter", size=12, color="white")
    )

    st.plotly_chart(fig, use_container_width=True)

    above_avg = sum([1 for i in range(len(company_values)) if company_values[i] > sector_values[i]])

    if above_avg >= 4:
        st.success(f"**Outperforming** sector average in {above_avg} out of 6 dimensions")
    elif above_avg >= 3:
        st.info(f"**Competitive** with sector average ({above_avg} dimensions above average)")
    else:
        st.warning(f"**Below** sector average in {6 - above_avg} dimensions - opportunity for improvement")

    st.markdown("---")
    st.markdown("## Growth Forecast")

    st.markdown("""
    Use the sliders below to simulate potential improvements in each dimension 
    and see the predicted impact on overall maturity.
    """)

    adjustments = {}

    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    for i, dim in enumerate(dimensions):
        with columns[i % 3]:
            current_score = company_row[f'DimScore_{dim}_After']
            adjustments[dim] = st.slider(
                f"{dim} (Current: {current_score:.1f})",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=1.0
            )

    new_scores = [company_row[f'DimScore_{dim}_Before'] + adjustments[dim] for dim in dimensions]
    predicted_maturity = ml_model.predict([new_scores])[0]

    current_maturity = company_row['Overall_Maturity_After']
    potential_gain = predicted_maturity - current_maturity

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Current Maturity", f"{current_maturity:.1f}")

    with col2:
        st.metric("Projected Maturity", f"{predicted_maturity:.1f}")

    with col3:
        st.metric("Potential Gain", f"+{potential_gain:.1f}")

    if potential_gain > 10:
        st.success("Significant potential for improvement with these investments!")
    elif potential_gain > 5:
        st.info("Moderate improvement expected from these changes")
    else:
        st.warning("Consider larger investments for meaningful impact")

elif page == "Predictive Tool":
    st.markdown("# Maturity Prediction Tool")

    st.markdown("""
    Enter hypothetical baseline scores for each dimension to predict the expected 
    post-intervention maturity level using our machine learning model.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    input_scores = {}

    with col1:
        st.markdown("### Input Dimension Scores (0-100)")
        for dim in dimensions[:3]:
            input_scores[dim] = st.slider(
                dim,
                min_value=0,
                max_value=100,
                value=50,
                step=5
            )

    with col2:
        st.markdown("### ")
        for dim in dimensions[3:]:
            input_scores[dim] = st.slider(
                dim,
                min_value=0,
                max_value=100,
                value=50,
                step=5
            )

    if st.button("Predict Maturity"):
        X_input = [[input_scores[dim] for dim in dimensions]]
        predicted_score = ml_model.predict(X_input)[0]

        if predicted_score < 45:
            level = "Novice"
            color = "#D32F2F"
        elif predicted_score < 75:
            level = "Competent"
            color = "#F57C00"
        else:
            level = "Leader"
            color = "#2E7D32"

        st.markdown("---")

        st.markdown(f"## Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Maturity Score", f"{predicted_score:.1f}")

        with col2:
            st.markdown(f"<h3 style='color: {color};'>Level: {level}</h3>", unsafe_allow_html=True)

        with col3:
            percentile = (df['Overall_Maturity_After'] < predicted_score).sum() / len(df) * 100
            st.metric("Estimated Percentile", f"{percentile:.0f}th")

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=predicted_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Digital Maturity Score"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 45], 'color': "#333"},
                    {'range': [45, 75], 'color': "#444"},
                    {'range': [75, 100], 'color': "#555"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig.update_layout(
            height=400,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12, color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("# Data Export")

    st.markdown("""
    Download the complete Digital Maturity Assessment datasets in Excel format.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Baseline Assessment")
        st.markdown("Pre-intervention digital maturity scores for all 1000 companies.")

        before_data = df[[c for c in df.columns if 'Before' in c or c == 'Company_ID']].copy()

        st.download_button(
            label="Download Baseline Data (Excel)",
            data=to_excel_download(before_data),
            file_name="digital_maturity_baseline.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.info(f"**{len(before_data)} companies** | **{len(before_data.columns)} columns**")

    with col2:
        st.markdown("### Post-Intervention Results")
        st.markdown("Final digital maturity scores after transformation initiatives.")

        after_data = df[[c for c in df.columns if 'After' in c or c == 'Company_ID']].copy()

        st.download_button(
            label="Download Results Data (Excel)",
            data=to_excel_download(after_data),
            file_name="digital_maturity_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.info(f"**{len(after_data)} companies** | **{len(after_data.columns)} columns**")

    st.markdown("---")

    st.markdown("### Combined Dataset")
    st.markdown("Complete dataset with both baseline and post-intervention data, plus calculated deltas.")

    st.download_button(
        label="Download Complete Dataset (Excel)",
        data=to_excel_download(df),
        file_name="digital_maturity_complete.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.info(f"**{len(df)} companies** | **{len(df.columns)} columns**")

    with st.expander("Preview Dataset"):
        st.dataframe(df.head(20))
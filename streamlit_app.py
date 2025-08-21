import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Streamlit page
st.set_page_config(
    page_title="VRU & AV Risk Assessment",
    page_icon="🚦",
    layout="wide"
)

st.title("Comparative Risk Assessment of VRUs and AVs Across Urban Intersections")
st.markdown("""
This app presents the key findings from a project investigating the comparative risk assessment of Vulnerable Road Users (VRUs) and Autonomous Vehicles (AVs)
across diverse urban intersection geometries. The analysis uses joint behavioral–environmental modeling to quantify and visualize safety risks.
""")
st.info("Select the analysis type from the sidebar to view results for VRUs or AVs.")

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_name):
    """Loads and preprocesses a specific CSV file."""
    try:
        df = pd.read_csv(file_name)
        cols_to_clean = ['DecisiveOrStyle', 'RoadCondition', 'ChannelQuality']
        for col in cols_to_clean:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_name}' was not found. Please make sure it's in the repository.")
        return None

# --- Main Analysis Function ---
def run_analysis(df, entity_type):
    """Performs the risk analysis and generates plots."""

    # Define conflict points and entry points based on the analysis type
    if entity_type == 'VRUs':
        conflict_points = {'highway': 0.360, 'standard_intersection': 0.832, 't_intersection': 0.672, 'roundabout': 0.931}
        vru_entry_points = {'highway': 2, 'standard_intersection': 8, 't_intersection': 5, 'roundabout': 12}
    elif entity_type == 'AVs':
        conflict_points = {'highway': 0.510, 'standard_intersection': 0.942, 't_intersection': 0.832, 'roundabout': 0.986}
        vru_entry_points = {'highway': 2, 'standard_intersection': 8, 't_intersection': 5, 'roundabout': 12}
    else: # This path is for the 'Combined' case from the first notebook block
        conflict_points = {'highway': 1, 'standard_intersection': 4, 't_intersection': 3, 'roundabout': 10}
        vru_entry_points = {'highway': 2, 'standard_intersection': 8, 't_intersection': 5, 'roundabout': 12}
    
    # Define the scoring function
    def score(cp, vep):
        metric = cp * vep if entity_type in ['VRUs', 'AVs'] else cp + vep
        if metric <= 3:
            return 'Best'
        elif metric <= 12:
            return 'Average'
        else:
            return 'Worst'

    # Create the road dataframe for risk metrics
    road_df = pd.DataFrame({
        'RoadType': conflict_points.keys(),
        'CP': [conflict_points[r] for r in conflict_points.keys()],
        'VEP': [vru_entry_points[r] for r in conflict_points.keys()]
    })
    
    # Calculate combined risk and safety rank
    if entity_type in ['VRUs', 'AVs']:
        road_df['CombinedRisk'] = road_df['CP'] * road_df['VEP']
    else:
        road_df['CombinedRisk'] = road_df['CP'] + road_df['VEP']

    road_df['SafetyRank'] = road_df.apply(lambda x: score(x.CP, x.VEP), axis=1)
    road_df['RoadType'] = road_df['RoadType'].str.replace('_', ' ').str.title()

    st.subheader(f"Combined Risk Metrics for {entity_type}")
    st.dataframe(road_df)
    
    # Calculate weighted frequencies for the plot
    combo_counts = df.groupby(['DecisiveOrStyle', 'RoadCondition', 'ChannelQuality']).size().reset_index(name='Count')
    expanded_rows = []
    for _, row in combo_counts.iterrows():
        for _, r_row in road_df.iterrows():
            if r_row['CombinedRisk'] > 0:
                weighted_count = row['Count'] / len(road_df)
                weight = 1 / r_row['CombinedRisk']
                weighted_value = weighted_count * weight
                expanded_rows.append({
                    'ConditionCombo': f"{row['DecisiveOrStyle']} | {row['RoadCondition']} | {row['ChannelQuality']}",
                    'RoadType': r_row['RoadType'],
                    'WeightedCount': weighted_value,
                    'SafetyRank': r_row['SafetyRank']
                })
    
    if expanded_rows:
        weighted_df = pd.DataFrame(expanded_rows)
        agg_df = weighted_df.groupby(['RoadType', 'SafetyRank', 'ConditionCombo']).sum().reset_index()

        # Create the bar plot
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.barplot(
            data=agg_df,
            x='WeightedCount',
            y='ConditionCombo',
            hue='SafetyRank',
            dodge=True,
            palette={'Best': 'green', 'Average': 'yellow', 'Worst': 'red'},
            ax=ax
        )
        ax.set_title(f'Weighted Frequency of Condition Combos by Road Type and Safety Rank for {entity_type}')
        ax.set_xlabel('Weighted Frequency (Higher = Safer Conditions More Frequent)')
        ax.set_ylabel('Condition Combination')
        ax.legend(title='SafetyRank')
        plt.tight_layout()
        st.subheader(f"Weighted Frequency Plot for {entity_type}")
        st.pyplot(fig)
    else:
        st.warning("Could not generate the plot. Data for weighted frequencies is empty.")

# --- Main App Logic ---
st.sidebar.header("Select Analysis Type")
analysis_type = st.sidebar.selectbox("Choose the entity to analyze:",
                                     ("Combined (CP+VEP)", "VRUs (CP*VEP)", "AVs (CP*VEP)"))

# Load the data based on the selection
if analysis_type == "Combined (CP+VEP)":
    data_file = 'av_vru_combined.csv'
    entity = "Combined"
elif analysis_type == "VRUs (CP*VEP)":
    data_file = 'vru_final.csv'
    entity = "VRUs"
else: # "AVs (CP*VEP)"
    data_file = 'av_final.csv'
    entity = "AVs"

data_df = load_data(data_file)

# Run the analysis if data is successfully loaded
if data_df is not None:
    run_analysis(data_df, entity)

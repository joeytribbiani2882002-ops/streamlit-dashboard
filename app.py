import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="German Credit Risk Dashboard",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset_31_credit-g.csv")

df = load_data()

# -----------------------------
# Title
# -----------------------------
st.title("German Credit Risk Dashboard")

st.markdown("""
This dashboard explores the **German Credit Risk dataset** using filters, KPIs, charts, 
descriptive statistics, and business insights for credit/lending decisions.
""")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters and Controls")

filtered_df = df.copy()

if "purpose" in df.columns:
    purpose_options = sorted(df["purpose"].dropna().unique())
    selected_purpose = st.sidebar.multiselect(
        "Select Loan Purpose",
        purpose_options,
        default=purpose_options
    )
    filtered_df = filtered_df[filtered_df["purpose"].isin(selected_purpose)]

if "housing" in df.columns:
    housing_options = sorted(df["housing"].dropna().unique())
    selected_housing = st.sidebar.multiselect(
        "Select Housing Type",
        housing_options,
        default=housing_options
    )
    filtered_df = filtered_df[filtered_df["housing"].isin(selected_housing)]

if "duration" in df.columns:
    duration_min = int(df["duration"].min())
    duration_max = int(df["duration"].max())

    duration_range = st.sidebar.slider(
        "Select Duration Range",
        min_value=duration_min,
        max_value=duration_max,
        value=(duration_min, duration_max)
    )

    filtered_df = filtered_df[
        (filtered_df["duration"] >= duration_range[0]) &
        (filtered_df["duration"] <= duration_range[1])
    ]

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)

# -----------------------------
# KPI Section
# -----------------------------
st.subheader("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(filtered_df))

if "credit_amount" in filtered_df.columns:
    col2.metric("Average Credit Amount", round(filtered_df["credit_amount"].mean(), 2))
else:
    col2.metric("Average Credit Amount", "N/A")

if "duration" in filtered_df.columns:
    col3.metric("Average Duration", round(filtered_df["duration"].mean(), 2))
else:
    col3.metric("Average Duration", "N/A")

if "class" in filtered_df.columns:
    good_count = filtered_df[filtered_df["class"] == "good"].shape[0]
    col4.metric("Good Credit Cases", good_count)
else:
    col4.metric("Good Credit Cases", "N/A")

# -----------------------------
# Descriptive Statistics
# -----------------------------
st.subheader("Descriptive Statistics")

numeric_columns = filtered_df.select_dtypes(include=["int64", "float64"]).columns

if len(numeric_columns) > 0:
    st.dataframe(filtered_df[numeric_columns].describe(), use_container_width=True)
else:
    st.warning("No numeric columns found for descriptive statistics.")

# -----------------------------
# Charts
# -----------------------------
st.subheader("Charts and Visual Analysis")

# Chart 1: Credit Amount Distribution
if "credit_amount" in filtered_df.columns:
    st.markdown("### Credit Amount Distribution")

    fig, ax = plt.subplots()
    ax.hist(filtered_df["credit_amount"].dropna(), bins=20)
    ax.set_xlabel("Credit Amount")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Credit Amount")

    st.pyplot(fig)

# Chart 2: Loan Purpose Count
if "purpose" in filtered_df.columns:
    st.markdown("### Loan Purpose Count")

    purpose_count = filtered_df["purpose"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(purpose_count.index, purpose_count.values)
    ax.set_xlabel("Purpose")
    ax.set_ylabel("Count")
    ax.set_title("Number of Loans by Purpose")
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig)

# Chart 3: Credit Risk Class
if "class" in filtered_df.columns:
    st.markdown("### Credit Risk Classification")

    class_count = filtered_df["class"].value_counts()

    fig, ax = plt.subplots()
    ax.pie(class_count.values, labels=class_count.index, autopct="%1.1f%%")
    ax.set_title("Good vs Bad Credit Risk")

    st.pyplot(fig)

# Chart 4: Housing Type
if "housing" in filtered_df.columns:
    st.markdown("### Housing Type Distribution")

    housing_count = filtered_df["housing"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(housing_count.index, housing_count.values)
    ax.set_xlabel("Housing Type")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Housing Type")

    st.pyplot(fig)

# -----------------------------
# Regex-Based Grouping
# -----------------------------
st.subheader("Regex-Based Grouping")

if "employment" in filtered_df.columns:
    st.markdown("Employment values were grouped using simple pattern matching.")

    def group_employment(value):
        value = str(value)

        if "unemployed" in value:
            return "Unemployed"
        elif "<1" in value:
            return "Less than 1 year"
        elif "1<=X<4" in value:
            return "1 to 4 years"
        elif "4<=X<7" in value:
            return "4 to 7 years"
        elif ">=7" in value:
            return "7 or more years"
        else:
            return "Unknown"

    filtered_df["employment_group"] = filtered_df["employment"].apply(group_employment)

    st.dataframe(
        filtered_df[["employment", "employment_group"]].head(20),
        use_container_width=True
    )

    employment_group_count = filtered_df["employment_group"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(employment_group_count.index, employment_group_count.values)
    ax.set_xlabel("Employment Group")
    ax.set_ylabel("Count")
    ax.set_title("Grouped Employment Status")
    plt.xticks(rotation=45, ha="right")

    st.pyplot(fig)
else:
    st.info("Employment column not found in dataset.")

# -----------------------------
# Business Insights
# -----------------------------
st.subheader("Business Insights")

st.markdown("""
### Key Findings

- The dashboard helps identify patterns in customer credit behaviour.
- Credit amount and duration are useful indicators for understanding lending risk.
- Loan purpose can show which types of borrowing are most common.
- Housing and employment status may support better credit risk analysis.
- Good and bad credit cases can help lenders understand risk distribution.

### LSEPI Discussion

**Legal:** Credit risk systems must follow data protection laws and avoid unfair discrimination.  
**Social:** Lending decisions can affect people's access to finance and opportunities.  
**Ethical:** Credit scoring should be transparent and avoid biased decision-making.  
**Professional:** Businesses should use accurate data and responsible analysis methods.  
**Innovation:** Dashboards help organisations make faster and more informed lending decisions.
""")

# -----------------------------
# Full Dataset
# -----------------------------
st.subheader("Full Filtered Dataset")
st.dataframe(filtered_df, use_container_width=True)

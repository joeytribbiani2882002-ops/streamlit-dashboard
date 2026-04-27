

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

# =========================
# Page Setup
# =========================
st.set_page_config(
    page_title="German Credit Risk Dashboard",
    layout="wide"
)

# =========================
# Dataset Loader
# =========================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
    df = pd.read_csv(url)

    df.columns = df.columns.str.lower().str.replace(" ", "_")

    if "amount" in df.columns:
        df = df.rename(columns={"amount": "credit_amount"})

    if "class" in df.columns:
        df = df.rename(columns={"class": "credit_risk"})

    if "status" in df.columns:
        df = df.rename(columns={"status": "checking_status"})

    df["credit_risk"] = df["credit_risk"].replace({
        1: "good",
        2: "bad",
        "1": "good",
        "2": "bad"
    })

    return df

df = load_data()

# =========================
# Title
# =========================
st.title("German Credit Risk Dashboard")

st.markdown(
    "This dashboard explores patterns in the German Credit dataset using "
    "interactive filters, KPIs, charts, descriptive statistics, regex-based "
    "grouping, and LSEPI discussion."
)

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("Filters and Controls")

include_unknown = st.sidebar.checkbox("Include Unknown Values", value=True)

age_range = st.sidebar.slider(
    "Select Age Range",
    int(df["age"].min()),
    int(df["age"].max()),
    (int(df["age"].min()), int(df["age"].max()))
)

duration_range = st.sidebar.slider(
    "Select Duration Range",
    int(df["duration"].min()),
    int(df["duration"].max()),
    (int(df["duration"].min()), int(df["duration"].max()))
)

purpose_options = sorted(df["purpose"].dropna().unique().tolist())
housing_options = sorted(df["housing"].dropna().unique().tolist())

purpose_filter = st.sidebar.multiselect(
    "Select Loan Purpose",
    purpose_options
)

housing_filter = st.sidebar.selectbox(
    "Select Housing Type",
    housing_options
)

# =========================
# Preprocessing
# =========================
df_processed = df.copy()

if not include_unknown:
    df_processed = df_processed.replace("unknown", pd.NA).dropna()

df_processed["age_group"] = pd.cut(
    df_processed["age"],
    bins=[0, 25, 60, 100],
    labels=["Young", "Adult", "Senior"]
)

# =========================
# Filtering
# =========================
df_filtered = df_processed[
    (df_processed["age"] >= age_range[0]) &
    (df_processed["age"] <= age_range[1]) &
    (df_processed["duration"] >= duration_range[0]) &
    (df_processed["duration"] <= duration_range[1])
]

if purpose_filter:
    df_filtered = df_filtered[df_filtered["purpose"].isin(purpose_filter)]

df_filtered = df_filtered[df_filtered["housing"] == housing_filter]

# =========================
# KPIs
# =========================
st.subheader("Key Performance Indicators")

if df_filtered.empty:
    st.warning("No data matches the selected filters. Please adjust the filters.")
else:
    total_customers = len(df_filtered)
    good_credit_pct = round((df_filtered["credit_risk"] == "good").mean() * 100, 2)
    avg_credit_amount = round(df_filtered["credit_amount"].mean(), 2)
    avg_duration = round(df_filtered["duration"].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)
    col2.metric("Good Credit %", f"{good_credit_pct}%")
    col3.metric("Average Credit Amount", avg_credit_amount)
    col4.metric("Average Duration", avg_duration)

    # =========================
    # Charts
    # =========================
    st.subheader("Overview Charts")

    colA, colB = st.columns(2)

    with colA:
        fig1, ax1 = plt.subplots()
        ax1.hist(df_filtered["credit_amount"], bins=20)
        ax1.set_title("Credit Amount Distribution")
        ax1.set_xlabel("Credit Amount")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

    with colB:
        fig2, ax2 = plt.subplots()
        df_filtered["purpose"].value_counts().plot(kind="bar", ax=ax2)
        ax2.set_title("Purpose Distribution")
        ax2.set_xlabel("Purpose")
        ax2.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)

    colC, colD = st.columns(2)

    with colC:
        fig3, ax3 = plt.subplots()
        df_filtered.boxplot(column="age", by="credit_risk", ax=ax3)
        ax3.set_title("Age by Credit Class")
        ax3.set_xlabel("Credit Class")
        ax3.set_ylabel("Age")
        plt.suptitle("")
        st.pyplot(fig3)

    with colD:
        fig4, ax4 = plt.subplots()
        df_filtered["housing"].value_counts().plot(kind="bar", ax=ax4)
        ax4.set_title("Housing Distribution")
        ax4.set_xlabel("Housing Type")
        ax4.set_ylabel("Count")
        st.pyplot(fig4)

    # =========================
    # Key Insights
    # =========================
    st.subheader("Key Insights")

    st.write("""
1. The credit amount distribution is right-skewed, which means most customers request smaller loans, while fewer customers request very large loans.

2. The most common borrowing purposes are linked to consumer purchases such as radio/TV, new car, and furniture/equipment.

3. The age boxplot suggests that customers with good credit tend to be slightly older on average than customers with bad credit.

4. The housing chart shows that many customers fall into stable housing categories, which may be linked to stronger financial reliability.
""")

    # =========================
    # Descriptive Statistics
    # =========================
    st.subheader("Descriptive Statistics")

    st.write("Summary Statistics for Credit Amount and Age")
    st.dataframe(df_filtered[["credit_amount", "age"]].describe())

    st.write("Average Credit Amount by Housing Type")
    housing_summary = df.groupby("housing")["credit_amount"].mean().reset_index()
    housing_summary.columns = ["Housing", "Average Credit Amount"]
    st.dataframe(housing_summary)

    with st.expander("Show Detailed Filtered Data"):
        st.dataframe(df_filtered)

# =========================
# Regex Transformation
# =========================
st.subheader("Regex Validation")

def checking_group(value):
    value = str(value)

    if re.search(r"<0", value):
        return "Low Balance"
    elif re.search(r"0<=X<200", value):
        return "Medium Balance"
    elif re.search(r"no checking", value, re.IGNORECASE):
        return "No Account"
    else:
        return "Other"

df_processed["checking_group"] = df_processed["checking_status"].apply(checking_group)

st.write("The table below shows how checking account values were grouped using regular expressions.")
st.dataframe(df_processed[["checking_status", "checking_group"]].drop_duplicates())

st.write("Frequency of Regex-Based Checking Groups")
st.bar_chart(df_processed["checking_group"].value_counts())

# =========================
# LSEPI
# =========================
st.markdown("---")
st.subheader("LSEPI Considerations")

st.markdown("""
**Legal:**  
Financial data must be handled carefully to avoid misuse or privacy concerns.  
**Mitigation:** The dataset used here is anonymised and intended only for educational analysis.

**Social:**  
Credit analysis may influence how different customer groups are perceived.  
**Mitigation:** The dashboard focuses on broad patterns rather than judging individual people.

**Ethical:**  
There is a risk of bias or unfair interpretation when analysing financial data.  
**Mitigation:** Results are presented as descriptive insights only, without making automated decisions.

**Professional:**  
A professional dashboard should be clear, accurate, and responsibly presented.  
**Mitigation:** This dashboard uses structured visuals, summaries, and transparent filtering to support responsible analysis.
""")

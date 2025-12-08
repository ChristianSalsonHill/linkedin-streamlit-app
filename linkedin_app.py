import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ---------- Helper function ----------
def clean_sm(x):
    # Convert survey codes to binary (1 if value == 1, else 0)
    x = np.where(x == 1, 1, 0)
    return x


# ---------- Load & clean data ----------
@st.cache_data
def load_and_clean_data(path: str = "social_media_usage.csv"):
    s = pd.read_csv(path)

    ss = pd.DataFrame(
        {
            "sm_li": clean_sm(s["web1h"]),  # target: LinkedIn user (1 = yes, 0 = no)
            "income": np.where(
                (s["income"] >= 1) & (s["income"] <= 9), s["income"], np.nan
            ),
            "education": np.where(
                (s["educ2"] >= 1) & (s["educ2"] <= 8), s["educ2"], np.nan
            ),
            "parent": clean_sm(s["par"]),
            "married": clean_sm(s["marital"]),
            "female": np.where(s["gender"] == 2, 1, 0),
            "age": np.where((s["age"] >= 18) & (s["age"] <= 98), s["age"], np.nan),
        }
    )

    ss = ss.dropna()
    ss = ss.astype(int)

    return ss


# ---------- Train model ----------
@st.cache_resource
def train_model(ss: pd.DataFrame):
    X = ss[["income", "education", "parent", "married", "female", "age"]]
    y = ss["sm_li"]

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        stratify=y,
        test_size=0.2,
        random_state=987,
    )

    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    accuracy = lr.score(X_test, y_test)

    return lr, accuracy, y_test, y_pred


# ---------- Simple custom CSS ----------
def add_custom_css():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f6fa;
        }

        h1 {
            font-family: "Helvetica Neue", sans-serif;
            font-weight: 700;
        }

        .prediction-card {
            padding: 0.75rem 1rem;
            border-radius: 0.75rem;
            font-weight: 600;
            margin-top: 0.75rem;
        }

        .prediction-ok {
            background-color: #e8f8f1;
            color: #107c41;
        }

        .prediction-bad {
            background-color: #fff4e6;
            color: #b34700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------- Streamlit app ----------
def main():
    st.set_page_config(
        page_title="LinkedIn User Prediction",
        layout="wide",
    )

    add_custom_css()

    # Load data + model
    ss = load_and_clean_data()
    model, accuracy, y_test, y_pred = train_model(ss)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Label dictionaries used for inputs / mapping
    income_options = {
        "1 — Less than $10,000": 1,
        "2 — $10k to under $20k": 2,
        "3 — $20k to under $30k": 3,
        "4 — $30k to under $40k": 4,
        "5 — $40k to under $50k": 5,
        "6 — $50k to under $75k": 6,
        "7 — $75k to under $100k": 7,
        "8 — $100k to under $150k": 8,
        "9 — $150k or more": 9,
    }

    education_options = {
        "1 — Less than high school": 1,
        "2 — HS incomplete / no diploma": 2,
        "3 — HS graduate or GED": 3,
        "4 — Some college, no degree": 4,
        "5 — 2-year associate degree": 5,
        "6 — 4-year college / bachelor’s": 6,
        "7 — Some postgraduate, no degree": 7,
        "8 — Postgraduate or professional degree": 8,
    }

    # Separate glossary text with NO '$' symbols (for display only)
    income_glossary = {
        1: "Less than 10k",
        2: "10k to under 20k",
        3: "20k to under 30k",
        4: "30k to under 40k",
        5: "40k to under 50k",
        6: "50k to under 75k",
        7: "75k to under 100k",
        8: "100k to under 150k",
        9: "150k or more",
    }

    # ---------- Header ----------
    st.markdown(
        """
        <h1>LinkedIn User Prediction App</h1>
        <p style="font-size:16px; color:#555; max-width:900px;">
        Explore how demographic characteristics relate to <strong>LinkedIn usage</strong>. 
        Use the <strong>Prediction</strong> tab to set a user profile and see the model’s prediction. 
        Then browse the <strong>Visuals</strong> and <strong>Metrics</strong> tabs for deeper insights.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Tabs ----------
    tab_pred, tab_visuals, tab_metrics = st.tabs(
        ["Prediction", "Visuals", "Metrics"]
    )

    # ----- Prediction tab -----
    with tab_pred:
        st.subheader("Set User Profile")

        st.caption(
            "Select a demographic profile. Age must be between **18** and **98**, "
            "which matches the range used to train the model."
        )

        income_label = st.selectbox(
            "Household Income",
            list(income_options.keys()),
            index=7,
        )
        income = income_options[income_label]

        education_label = st.selectbox(
            "Education Level",
            list(education_options.keys()),
            index=6,
        )
        education = education_options[education_label]

        parent = 1 if st.selectbox("Parent?", ["No", "Yes"], index=0) == "Yes" else 0
        married = 1 if st.selectbox("Married?", ["No", "Yes"], index=1) == "Yes" else 0
        female = 1 if st.selectbox("Gender", ["Male", "Female"], index=1) == "Female" else 0

        age = st.number_input(
            "Age (18–98)",
            min_value=18,
            max_value=98,
            value=40,
            step=1,
        )

        person = np.array([[income, education, parent, married, female, age]])

        pred_class = model.predict(person)[0]
        prob = model.predict_proba(person)[0][1]

        st.markdown("### Prediction for Selected Profile")

        if pred_class == 1:
            st.markdown(
                f"""
                <div class="prediction-card prediction-ok">
                <strong>LIKELY</strong> a LinkedIn user.<br>
                Probability they use LinkedIn: <strong>{prob:.1%}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-card prediction-bad">
                ⚠️ <strong>NOT likely</strong> a LinkedIn user.<br>
                Probability they use LinkedIn: <strong>{prob:.1%}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ----- Visuals tab -----
    # Data prep

    # Income (codes 1–9)
    income_usage = (
        ss.groupby("income")["sm_li"]
        .mean()
        .reset_index()
        .rename(columns={"sm_li": "Share using LinkedIn"})
    )

    # Education (codes 1–8)
    education_usage = (
        ss.groupby("education")["sm_li"]
        .mean()
        .reset_index()
        .rename(columns={"sm_li": "Share using LinkedIn"})
    )

    # Age: average LinkedIn use by each age
    age_mean = ss.groupby("age", as_index=False)["sm_li"].mean()

    # Example profiles: same everything except age
    example_profiles = pd.DataFrame(
        {
            "income": [7, 7],      # 75k–100k range
            "education": [6, 6],   # 4-year college / bachelor’s
            "parent": [0, 0],
            "married": [1, 1],
            "female": [1, 1],
            "age": [42, 82],
        }
    )
    example_probs = model.predict_proba(example_profiles)[:, 1]

    example_table = pd.DataFrame(
        {
            "Age": [42, 82],
            "Income code": [7, 7],
            "Education code": [6, 6],
            "Parent": ["No", "No"],
            "Married": ["Yes", "Yes"],
            "Female": ["Yes", "Yes"],
            "Predicted probability of LinkedIn use": [f"{p:.1%}" for p in example_probs],
        }
    )

    with tab_visuals:
        st.header("Insights from the Data")

        # ---------- Income (bar + glossary) ----------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("LinkedIn Usage by Income (Codes 1–9)")
            income_chart = (
                alt.Chart(income_usage)
                .mark_bar()
                .encode(
                    x=alt.X("income:O", title="Income code (1–9)"),
                    y=alt.Y(
                        "Share using LinkedIn:Q",
                        axis=alt.Axis(format="%", title="Share using LinkedIn"),
                    ),
                    tooltip=[
                        alt.Tooltip("income:O", title="Income code"),
                        alt.Tooltip("Share using LinkedIn:Q", format=".1%"),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(income_chart, use_container_width=True)

        with col2:
            st.subheader("Income Code Glossary")
            st.caption("Codes match the original survey categories.")
            for code in range(1, 10):
                st.markdown(f"**{code}** — {income_glossary[code]}")

        st.markdown("---")

        # ---------- Education (bar + glossary) ----------
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("LinkedIn Usage by Education (Codes 1–8)")
            education_chart = (
                alt.Chart(education_usage)
                .mark_bar()
                .encode(
                    x=alt.X("education:O", title="Education code (1–8)"),
                    y=alt.Y(
                        "Share using LinkedIn:Q",
                        axis=alt.Axis(format="%", title="Share using LinkedIn"),
                    ),
                    tooltip=[
                        alt.Tooltip("education:O", title="Education code"),
                        alt.Tooltip("Share using LinkedIn:Q", format=".1%"),
                    ],
                )
                .properties(height=360)
            )
            st.altair_chart(education_chart, use_container_width=True)

        with col4:
            st.subheader("Education Code Glossary")
            st.caption("Higher codes correspond to higher education levels.")
            for label, code in sorted(education_options.items(), key=lambda x: x[1]):
                desc = label.split("—", 1)[1].strip()
                st.markdown(f"**{code}** — {desc}")

        st.markdown("---")

        # ---------- Age (smoothed line + comparison table) ----------
        col5, col6 = st.columns(2)

        with col5:
            st.subheader("Probability of LinkedIn Use by Age")
            age_chart = (
                alt.Chart(age_mean)
                .transform_loess("age", "sm_li", bandwidth=0.3)
                .mark_line()
                .encode(
                    x=alt.X("age:Q", title="Age"),
                    y=alt.Y(
                        "sm_li:Q",
                        title="Probability of LinkedIn Use",
                        scale=alt.Scale(domain=[-0.05, 0.7]),
                    ),
                    tooltip=[
                        alt.Tooltip("age:Q", title="Age", format=".0f"),
                        alt.Tooltip("sm_li:Q", title="Estimated probability", format=".2f"),
                    ],
                )
                .properties(
                    height=360,
                    title="Probability of LinkedIn Use by Age",
                )
            )
            st.altair_chart(age_chart, use_container_width=True)

        with col6:
            st.subheader("Example: Age as a Predictor")
            st.caption(
                "These two profiles are identical except for age. "
                "The change in predicted probability highlights how strongly age influences LinkedIn use."
            )
            st.table(example_table)

    # ----- Metrics tab -----
    with tab_metrics:
        st.subheader("Model Performance Metrics (Test Set)")

        st.metric("Accuracy", f"{accuracy:.1%}")
        st.metric("Precision (class = user)", f"{precision:.1%}")
        st.metric("Recall (class = user)", f"{recall:.1%}")
        st.metric("F1 Score", f"{f1:.1%}")

        st.markdown("#### Confusion Matrix")
        cm_df = pd.DataFrame(
            cm,
            columns=["Predicted non-user", "Predicted user"],
            index=["Actual non-user", "Actual user"],
        )
        st.dataframe(cm_df)


if __name__ == "__main__":
    main()

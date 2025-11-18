import numpy as np
import pandas as pd
import streamlit as st

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance


# ------------------------------------------------------------
# Global constants
# ------------------------------------------------------------

INCOME_COL = "total_income"
TARGET_COL = "cashflow"

# Flex categories we allow AI to adjust (0–20% total cut, uneven)
FLEX_CATEGORIES = [
    "utilities",
    "groceries",
    "transport",
    "dining_out",
    "entertainment",
    "clothing",
    "other_variable",      # Misc / impulse
    "unexpected_spend",    # Cushion
]

# Non-flex categories (we do NOT alter these in optimization)
FIXED_CATEGORIES = [
    "housing",
    "insurance",
    "childcare",
    "healthcare",
    "education",
    "debt_min_payments",
    "savings_contrib",
    "investment_contrib",
]


@dataclass
class ModelArtifacts:
    model: MLPRegressor
    feature_cols: list
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


# ------------------------------------------------------------
# Synthetic dataset generator
# ------------------------------------------------------------

def generate_synthetic_dataset(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    income = np.random.uniform(4717, 14150, size=n)
    family_size = np.random.randint(1, 6, size=n)
    num_earners = np.random.randint(1, 3, size=n)
    age_main = np.random.randint(25, 65, size=n)
    home_owner = np.random.binomial(1, 0.6, size=n)

    def clip_ratio(r):
        return np.clip(r, 0, 0.6)

    housing_ratio = np.where(home_owner == 1,
                             np.random.normal(0.23, 0.04, size=n),
                             np.random.normal(0.30, 0.05, size=n))
    utilities_ratio = np.random.normal(0.07, 0.015, size=n)
    groceries_ratio = np.random.normal(0.12, 0.03, size=n)
    transport_ratio = np.random.normal(0.10, 0.03, size=n)
    insurance_ratio = np.random.normal(0.06, 0.015, size=n)
    childcare_ratio = np.where(family_size > 2,
                               np.random.normal(0.07, 0.03, size=n),
                               np.random.normal(0.02, 0.01, size=n))
    healthcare_ratio = np.random.normal(0.06, 0.02, size=n)
    entertainment_ratio = np.random.normal(0.05, 0.02, size=n)
    dining_ratio = np.random.normal(0.05, 0.02, size=n)
    clothing_ratio = np.random.normal(0.03, 0.01, size=n)
    education_ratio = np.random.normal(0.04, 0.02, size=n)
    debt_min_ratio = np.random.normal(0.07, 0.03, size=n)
    savings_ratio = np.random.normal(0.05, 0.02, size=n)
    invest_ratio = np.random.normal(0.05, 0.03, size=n)
    unexpected_ratio = np.abs(np.random.normal(0.03, 0.03, size=n))
    other_var_ratio = np.random.normal(0.04, 0.02, size=n)

    ratios = [
        housing_ratio, utilities_ratio, groceries_ratio, transport_ratio,
        insurance_ratio, childcare_ratio, healthcare_ratio, entertainment_ratio,
        dining_ratio, clothing_ratio, education_ratio, debt_min_ratio,
        savings_ratio, invest_ratio, unexpected_ratio, other_var_ratio
    ]
    ratios = [clip_ratio(r) for r in ratios]
    (housing_ratio, utilities_ratio, groceries_ratio, transport_ratio,
     insurance_ratio, childcare_ratio, healthcare_ratio, entertainment_ratio,
     dining_ratio, clothing_ratio, education_ratio, debt_min_ratio,
     savings_ratio, invest_ratio, unexpected_ratio, other_var_ratio) = ratios

    housing = income * housing_ratio
    utilities = income * utilities_ratio
    groceries = income * groceries_ratio
    transport = income * transport_ratio
    insurance = income * insurance_ratio
    childcare = income * childcare_ratio
    healthcare = income * healthcare_ratio
    entertainment = income * entertainment_ratio
    dining_out = income * dining_ratio
    clothing = income * clothing_ratio
    education = income * education_ratio
    debt_min_payments = income * debt_min_ratio
    savings_contrib = income * savings_ratio
    investment_contrib = income * invest_ratio
    unexpected_spend = income * unexpected_ratio
    other_variable = income * other_var_ratio

    total_outflows = (housing + utilities + groceries + transport + insurance +
                      childcare + healthcare + entertainment + dining_out +
                      clothing + education + debt_min_payments + savings_contrib +
                      investment_contrib + unexpected_spend + other_variable)

    cashflow = income - total_outflows

    df = pd.DataFrame({
        INCOME_COL: income,
        "family_size": family_size,
        "num_earners": num_earners,
        "age_main": age_main,
        "home_owner": home_owner,
        "housing": housing,
        "utilities": utilities,
        "groceries": groceries,
        "transport": transport,
        "insurance": insurance,
        "childcare": childcare,
        "healthcare": healthcare,
        "entertainment": entertainment,
        "dining_out": dining_out,
        "clothing": clothing,
        "education": education,
        "debt_min_payments": debt_min_payments,
        "savings_contrib": savings_contrib,
        "investment_contrib": investment_contrib,
        "unexpected_spend": unexpected_spend,
        "other_variable": other_variable,
        TARGET_COL: cashflow,
    })

    return df


# ------------------------------------------------------------
# Training the MLP model
# ------------------------------------------------------------

def train_cashflow_model(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelArtifacts:

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=800,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    return ModelArtifacts(
        model=model,
        feature_cols=feature_cols,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


# ------------------------------------------------------------
# Quality-of-life–aware optimizer (0–20% cuts)
# ------------------------------------------------------------

def optimize_family(
    row: pd.Series,
    artifacts: ModelArtifacts,
    max_cut: float = 0.2,
    n_steps: int = 5,
) -> dict:
    """
    Quality-of-life–aware optimization.

    - Primary goal: minimal reduction in flex categories such that
      predicted_cashflow_after > 0.
    - 5% of income is reported as a diagnostic flag, not enforced.
    - Cuts are uneven:
        * Heavy flex (dining_out, entertainment, other_variable, unexpected_spend)
          can be cut up to max_cut.
        * Soft flex (utilities, groceries, transport, clothing) only up to max_cut/2.
    """

    model = artifacts.model
    feature_cols = artifacts.feature_cols

    # Split flex into "soft" vs "heavy"
    heavy_flex = ["dining_out", "entertainment", "other_variable", "unexpected_spend"]
    soft_flex = ["utilities", "groceries", "transport", "clothing"]

    income = row[INCOME_COL]
    target_min_cf = 0.05 * income  # for reporting only

    base_features = row[feature_cols].copy()
    base_pred = float(model.predict(base_features.to_frame().T)[0])

    best_features = base_features.copy()
    best_pred = base_pred
    best_cut = 0.0

    # Candidate cut levels (e.g. 0%, 5%, 10%, 15%, 20%)
    cut_levels = np.linspace(0.0, max_cut, n_steps)

    for alpha in cut_levels:
        candidate = base_features.copy()

        # Soft flex: lighter cut
        for col in soft_flex:
            if col in candidate:
                candidate[col] = candidate[col] * (1.0 - 0.5 * alpha)

        # Heavy flex: full cut
        for col in heavy_flex:
            if col in candidate:
                candidate[col] = candidate[col] * (1.0 - alpha)

        pred_cf = float(model.predict(candidate.to_frame().T)[0])

        # First cut level that makes cashflow positive -> accept and stop
        if pred_cf > 0:
            best_features = candidate
            best_pred = pred_cf
            best_cut = alpha
            break

        # Otherwise track best improvement (even if still negative)
        if pred_cf > best_pred:
            best_features = candidate
            best_pred = pred_cf
            best_cut = alpha

    meets_5pct = best_pred >= target_min_cf

    return {
        "income": income,
        "original_features": base_features,
        "optimized_features": best_features,
        "pred_cashflow_before": base_pred,
        "pred_cashflow_after": best_pred,
        "meets_5pct_constraint": meets_5pct,
        "cut_fraction": best_cut,
    }


# ------------------------------------------------------------
# Cache model so we don't retrain each interaction
# ------------------------------------------------------------

@st.cache_resource
def get_trained_model() -> ModelArtifacts:
    df = generate_synthetic_dataset()
    artifacts = train_cashflow_model(df)
    return artifacts


# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------

def main():
    st.title("AI Smart Spending Planner – Live Demo")

    st.markdown(
        """
        This demo shows how our AI can take a monthly budget
        and propose **small, smart changes** that move a family
        from **negative to positive cashflow** while protecting quality of life.

        **Adjustable categories (0–20% cuts, uneven):**
        - Soft cuts (max ~10%): Utilities, Groceries, Transportation, Clothing  
        - Heavier cuts (max ~20%): Dining Out, Entertainment, Misc/Impulse, Unexpected Cushion
        """
    )

    artifacts = get_trained_model()
    feature_cols = artifacts.feature_cols

    st.subheader("1. Enter Household Profile & Current Spending")

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Total Monthly Income ($)", min_value=1000.0, value=6000.0, step=100.0)
        family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3, step=1)
        num_earners = st.number_input("Number of Earners", min_value=1, max_value=4, value=2, step=1)
        age_main = st.number_input("Age of Main Earner", min_value=18, max_value=80, value=40, step=1)
        home_owner = st.selectbox("Home Ownership", options=[0, 1], format_func=lambda x: "Renter" if x == 0 else "Owner")

    with col2:
        st.markdown("**Essential Monthly Spending ($)**")
        housing = st.number_input("Housing (rent/mortgage)", min_value=0.0, value=1800.0, step=50.0)
        utilities = st.number_input("Utilities", min_value=0.0, value=300.0, step=20.0)
        groceries = st.number_input("Groceries", min_value=0.0, value=800.0, step=20.0)
        transport = st.number_input("Transport", min_value=0.0, value=400.0, step=20.0)
        insurance = st.number_input("Insurance", min_value=0.0, value=300.0, step=20.0)
        childcare = st.number_input("Childcare", min_value=0.0, value=300.0, step=20.0)
        healthcare = st.number_input("Healthcare", min_value=0.0, value=200.0, step=20.0)
        debt_min = st.number_input("Debt Minimum Payments", min_value=0.0, value=400.0, step=20.0)

    st.markdown("**Flexible & Financial Monthly Spending ($)**")
    col3, col4 = st.columns(2)

    with col3:
        entertainment = st.number_input("Entertainment", min_value=0.0, value=250.0, step=10.0)
        dining_out = st.number_input("Dining Out", min_value=0.0, value=350.0, step=10.0)
        clothing = st.number_input("Clothing", min_value=0.0, value=150.0, step=10.0)

    with col4:
        education = st.number_input("Education", min_value=0.0, value=200.0, step=10.0)
        savings = st.number_input("Savings Contributions", min_value=0.0, value=200.0, step=10.0)
        investment = st.number_input("Investment Contributions", min_value=0.0, value=200.0, step=10.0)
        unexpected = st.number_input("Unexpected Spending Cushion", min_value=0.0, value=250.0, step=10.0)
        other_variable = st.number_input("Miscellaneous / Impulse", min_value=0.0, value=200.0, step=10.0)

    if st.button("Run AI Smart Plan"):

        # 1. Build feature row
        row_data = {
            INCOME_COL: income,
            "family_size": family_size,
            "num_earners": num_earners,
            "age_main": age_main,
            "home_owner": home_owner,
            "housing": housing,
            "utilities": utilities,
            "groceries": groceries,
            "transport": transport,
            "insurance": insurance,
            "childcare": childcare,
            "healthcare": healthcare,
            "entertainment": entertainment,
            "dining_out": dining_out,
            "clothing": clothing,
            "education": education,
            "debt_min_payments": debt_min,
            "savings_contrib": savings,
            "investment_contrib": investment,
            "unexpected_spend": unexpected,
            "other_variable": other_variable,
        }

        row_df = pd.DataFrame([row_data])
        row_series = row_df.iloc[0]

        # 2. Run optimization
        opt = optimize_family(row_series, artifacts)
        original = opt["original_features"]
        optimized = opt["optimized_features"]

        # 3. Compute original cashflow from user's input
        total_spend_original = (
            housing + utilities + groceries + transport + insurance +
            childcare + healthcare + entertainment + dining_out +
            clothing + education + debt_min + savings + investment +
            unexpected + other_variable
        )
        cashflow_original = income - total_spend_original

        st.subheader("2. Results")

        col_before, col_after = st.columns(2)

        with col_before:
            st.markdown("### Before (Your Current Plan)")
            st.metric("Cashflow", f"${cashflow_original:,.0f}")
            st.metric("Cashflow / Income", f"{cashflow_original / income * 100:,.1f}%")

        with col_after:
            st.markdown("### After (AI-Recommended Plan)")
            cf_after = opt["pred_cashflow_after"]
            st.metric("Predicted Cashflow", f"${cf_after:,.0f}")
            st.metric("Cashflow / Income", f"{cf_after / income * 100:,.1f}%")
            st.markdown(
                f"**Reaches 5% income savings?** "
                f"{'✅ Yes' if opt['meets_5pct_constraint'] else 'ℹ Not required (quality-of-life prioritized)'}"
            )
            st.markdown(f"**Global cut level used (0–20%)**: `{opt['cut_fraction'] * 100:.1f}%`")

        # 4. Monthly Flex Category Changes Table
        st.markdown("### 3. Recommended Changes in Flexible Categories (per month)")

        DISPLAY_LABELS = {
            "utilities": "Utilities",
            "groceries": "Groceries",
            "transport": "Transportation",
            "dining_out": "Dining Out",
            "entertainment": "Entertainment",
            "clothing": "Clothing",
            "other_variable": "Miscellaneous / Impulse",
            "unexpected_spend": "Unexpected Cushion",
        }

        rows = []
        for col in FLEX_CATEGORIES:
            rows.append({
                "Category": DISPLAY_LABELS.get(col, col),
                "Before ($/month)": original[col],
                "After ($/month)": optimized[col],
            })

        changes = pd.DataFrame(rows)
        changes["Change ($/month)"] = changes["After ($/month)"] - changes["Before ($/month)"]
        changes["Change (%/month)"] = (
            changes["After ($/month)"] / changes["Before ($/month)"] - 1.0
        ) * 100.0

        st.dataframe(
            changes.style.format({
                "Before ($/month)": "{:,.0f}",
                "After ($/month)": "{:,.0f}",
                "Change ($/month)": "{:,.0f}",
                "Change (%/month)": "{:,.1f}%"
            })
        )

        # 5. Flex Budgets for 3, 6, 12 months
        st.markdown("### 4. Flex Budgets for the Next 3, 6, and 12 Months")

        horizon_rows = []
        for col in FLEX_CATEGORIES:
            monthly_after = optimized[col]
            horizon_rows.append({
                "Category": DISPLAY_LABELS.get(col, col),
                "Monthly ($)": monthly_after,
                "3-Month ($)": monthly_after * 3,
                "6-Month ($)": monthly_after * 6,
                "12-Month ($)": monthly_after * 12,
            })

        horizons_df = pd.DataFrame(horizon_rows)
        st.dataframe(
            horizons_df.style.format({
                "Monthly ($)": "{:,.0f}",
                "3-Month ($)": "{:,.0f}",
                "6-Month ($)": "{:,.0f}",
                "12-Month ($)": "{:,.0f}",
            })
        )

        # 6. Summary of Flex Total Savings
        total_flex_before = sum(original[col] for col in FLEX_CATEGORIES)
        total_flex_after = sum(optimized[col] for col in FLEX_CATEGORIES)

        st.markdown("#### Summary of Flex Spending & Savings")

        st.write(
            f"**Total flex spending per month**: "
            f"Before ${total_flex_before:,.0f} → After ${total_flex_after:,.0f} "
            f"(Δ ${total_flex_after - total_flex_before:,.0f} / month)"
        )

        st.write(
            f"- 3 months:  ${ (total_flex_before - total_flex_after) * 3:,.0f} saved"
        )
        st.write(
            f"- 6 months:  ${ (total_flex_before - total_flex_after) * 6:,.0f} saved"
        )
        st.write(
            f"- 12 months: ${ (total_flex_before - total_flex_after) * 12:,.0f} saved"
        )

        st.markdown(
            """
            **Interpretation:**
            - Income and non-flex categories are assumed fixed.
            - These 3/6/12-month budgets tell you how much you can safely plan to spend
              in each flexible category while staying on a positive-cashflow track.
            - The flex savings numbers show how much extra cash you free up if you follow the AI plan.
            """
        )


if __name__ == "__main__":
    main()

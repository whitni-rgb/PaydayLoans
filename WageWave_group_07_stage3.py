#!/usr/bin/env python3
"""FIN 6155 — WageWave (Group 07) | Stage 3 random-forest challenger and fairness audit.

Single-file Stage 3 build from raw synthetic data; no other code files required.

Runs the full build from the original, unzipped synthetic data release:
    python WageWave_group_07_stage3.py

Input layout (the only external input is data_release/):
    project/WageWave_group_07_stage3.py
    project/data_release/applications.csv
    project/data_release/outcomes_train.csv
    project/data_release/holdout_ids.csv
    project/data_release/plaid_synth_shard_*.ndjson.gz   (50 files)

Produces outputs/WageWave_group_07_stage3.csv, and supporting feature,
validation, fairness-audit, remediation, and explanation artifacts in outputs/.
Requires Python 3.10+ and numpy, pandas, scipy, scikit-learn, joblib.
Optional orjson makes JSON shard reading faster.

Implementation: reuse Stage 2 pre-registered feature definitions, compare
logistic regression with one random-forest challenger, audit both using the
group-blind 70% reference approval policy, implement a single global 75%
approval-cutoff remediation, and score the 5,000 pending applicants.
'group' enters fairness reporting ONLY after predictions and decisions; it
never enters features, imputation, training, thresholds, or segmentation.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import date, timedelta
from math import isfinite
from pathlib import Path
from statistics import mean, stdev

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import orjson  # optional faster JSON reader; regular json works too
except ImportError:
    orjson = None

# ======= CONFIG: change at most these two lines. =======
WORKING_DIR = Path(__file__).resolve().parent
DATA_DIR = WORKING_DIR / "data_release"
# ===============================================================

OUTPUT_DIR = WORKING_DIR / "outputs"
SUBMISSION_FILENAME = "WageWave_group_07_stage3.csv"


# ===========================================================================
# PAYROLL
# ===========================================================================


# Pre-registered six-month window, not the full transaction history (which
# reaches back into 2025). February-July 2026 is a recent, comparable horizon
# for wage and expense measures, ending before every August/September application.
# This deliberately trades older evidence / seasonal coverage for recency and
# comparability; short-file customers use their observed months within the window.
WINDOW_START = date(2026, 2, 1)
WINDOW_END = date(2026, 7, 31)

# Pre-registered income policy: WAGES ONLY, not all economic income.
# Gig payouts (INCOME_OTHER_INCOME) may be genuine resources but are NOT counted
# as payroll: we specifically measure recurring wages relevant to WageWave's
# paycheck-cycle repayment hypothesis. Other deposits/transfers are also excluded.
# An applicant with transactions but no qualifying INCOME_WAGES deposits gets
# monthly_payroll_income=0; payroll-denominated ratios and volatility are NaN,
# NOT zero. Payroll regularity is NaN if fewer than three distinct payroll dates.
# Imputation and missingness indicators are fitted only on labeled training rows.
WAGE_INCOME_DETAIL = "INCOME_WAGES"
GIG_INCOME_DETAIL = "INCOME_OTHER_INCOME"  # excluded; retained as an explicit policy marker


def deduplicate_transactions(transactions: list[dict]):
    """Keep non-pending records, removing same-account/amount/name copies
    dated on the same day or one day apart, as described in Appendix A.2.
    """
    posted = [t for t in transactions if not t.get("pending", False)]
    ordered = sorted(posted, key=lambda t: t["date"])
    last_kept_date = {}
    cleaned = []
    removed = 0

    for transaction in ordered:
        transaction_date = date.fromisoformat(transaction["date"])
        key = (
            transaction.get("account_id"),
            round(float(transaction["amount"]), 2),
            transaction.get("name"),
        )
        previous = last_kept_date.get(key)
        if previous is not None and (transaction_date - previous).days <= 1:
            removed += 1
            continue
        last_kept_date[key] = transaction_date
        cleaned.append((transaction_date, transaction))

    return cleaned, removed


def payroll_feature(bundle: dict) -> dict:
    """Pre-registered feature: total wage deposits / observed months.

    Observed months span the first through last transaction months in the
    six-month observation window, including months with zero wage deposits.
    A customer with other transactions but no wages has payroll income 0;
    a customer with no observable transactions has missing payroll income.
    """
    customer_id = bundle["customer_id"]
    clean, n_duplicates = deduplicate_transactions(bundle.get("transactions") or [])
    in_window = [
        (day, transaction)
        for day, transaction in clean
        if WINDOW_START <= day <= WINDOW_END
    ]

    if not in_window:
        return {
            "customer_id": customer_id,
            "monthly_payroll_income": float("nan"),
            "months_observed": 0,
            "payroll_deposits": 0,
            "duplicate_transactions_removed": n_duplicates,
        }

    first_day = in_window[0][0]
    last_day = in_window[-1][0]
    months_observed = (last_day.year - first_day.year) * 12 + last_day.month - first_day.month + 1

    total_payroll = 0.0
    count_payroll = 0
    for _, transaction in in_window:
        pfc = transaction.get("personal_finance_category") or {}
        # Pre-registration commits specifically to payroll, not all inflows.
        # The policy excludes INCOME_OTHER_INCOME (gig payouts), which may
        # represent real earnings but is not wage payroll for this hypothesis.
        # TRANSFER_IN_* deposits and unrelated inflows are also excluded.
        if (pfc.get("primary") == "INCOME"
                and pfc.get("detailed") == WAGE_INCOME_DETAIL
                and float(transaction["amount"]) < 0):
            total_payroll -= float(transaction["amount"])
            count_payroll += 1

    return {
        "customer_id": customer_id,
        "monthly_payroll_income": round(total_payroll / months_observed, 2),
        "months_observed": months_observed,
        "payroll_deposits": count_payroll,
        "duplicate_transactions_removed": n_duplicates,
    }


def payroll() -> None:
    shards = sorted(DATA_DIR.glob("plaid_synth_shard_*.ndjson.gz"))
    if len(shards) != 50:
        raise ValueError(f"Expected 50 shards in {DATA_DIR}, found {len(shards)}")

    applications = pd.read_csv(DATA_DIR / "applications.csv", usecols=["customer_id", "application_date"])
    if applications["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer_id in applications.csv")
    if (pd.to_datetime(applications["application_date"]).dt.date <= WINDOW_END).any():
        raise ValueError("Some applications are dated within/before the observation window")

    rows = []
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(payroll_feature(json.loads(line)))

    features = pd.DataFrame(rows)
    if not features["customer_id"].is_unique:
        raise ValueError("Duplicate customer_id in JSON bundles")
    if set(features["customer_id"]) != set(applications["customer_id"]):
        raise ValueError("JSON customer IDs do not match applications.csv")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "payroll_features.csv"
    features.to_csv(path, index=False, float_format="%.2f")

    print(f"Shards read: {len(shards)}")
    print(f"Customers processed: {len(features):,}")
    print(f"Customers without wage deposits: {(features.payroll_deposits == 0).sum():,}")
    print(f"Customers without transaction history in window: {(features.months_observed == 0).sum():,}")
    print(f"Duplicate transactions removed: {features.duplicate_transactions_removed.sum():,}")
    print(f"Mean monthly payroll income: ${features.monthly_payroll_income.mean():,.2f}")
    print(f"Median monthly payroll income: ${features.monthly_payroll_income.median():,.2f}")
    print(f"Wrote: {path}")


# ===========================================================================
# INCOME REGULARITY
# ===========================================================================


def income_regularity_feature(bundle: dict) -> dict:
    """Calculate negative SD of inter-payroll dates, deduplicating first.

    Wage identification is intentionally identical to payroll_feature().
    Multiple payroll deposits on one calendar day count as one pay date;
    without this rule the interval would be zero for same-day deposits.
    """
    clean, _ = deduplicate_transactions(bundle.get("transactions") or [])
    pay_dates: set[date] = set()

    for day, tx in clean:
        if not WINDOW_START <= day <= WINDOW_END:
            continue
        pfc = tx.get("personal_finance_category") or {}
        if (
            pfc.get("primary") == "INCOME"
            and pfc.get("detailed") == WAGE_INCOME_DETAIL
            and float(tx["amount"]) < 0
        ):
            pay_dates.add(day)

    dates = sorted(pay_dates)
    intervals = [(later - earlier).days for earlier, later in zip(dates, dates[1:])]
    gap_sd = stdev(intervals) if len(intervals) >= 2 else float("nan")
    regularity = -gap_sd if gap_sd > 0 else (0.0 if gap_sd == 0 else float("nan"))

    return {
        "customer_id": bundle["customer_id"],
        "income_regularity": regularity,
        "distinct_payroll_dates": len(dates),
        "payroll_intervals": len(intervals),
        "mean_payroll_gap_days": sum(intervals) / len(intervals) if intervals else float("nan"),
    }


def income_regularity() -> None:
    shards = sorted(DATA_DIR.glob("plaid_synth_shard_*.ndjson.gz"))
    if len(shards) != 50:
        raise ValueError(f"Expected 50 shards in {DATA_DIR}; found {len(shards)}")

    apps = pd.read_csv(DATA_DIR / "applications.csv", usecols=["customer_id", "application_date"])
    if apps["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer_id in applications.csv")
    if (pd.to_datetime(apps["application_date"]).dt.date <= WINDOW_END).any():
        raise ValueError("An application is not later than the feature observation window")

    features = []
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    features.append(income_regularity_feature(json.loads(line)))

    df = pd.DataFrame(features)
    if not df["customer_id"].is_unique or set(df["customer_id"]) != set(apps["customer_id"]):
        raise ValueError("Customer IDs do not match the unique application IDs")
    if (df["income_regularity"].dropna() > 0).any():
        raise ValueError("Income regularity must be nonpositive")
    if not df.loc[df["payroll_intervals"] < 2, "income_regularity"].isna().all():
        raise ValueError("Regularity must be missing with fewer than two intervals")

    # Sanity-check against existing payroll output when present.
    previous = OUTPUT_DIR / "payroll_features.csv"
    if previous.exists():
        payroll = pd.read_csv(previous, usecols=["customer_id", "payroll_deposits"])
        comparison = df.merge(payroll, on="customer_id", validate="one_to_one")
        if len(comparison) != len(df):
            raise ValueError("Payroll and income-regularity features have mismatched customer IDs")
        if (comparison["distinct_payroll_dates"] > comparison["payroll_deposits"]).any():
            raise ValueError("More distinct payroll dates than payroll transactions")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "income_regularity_features.csv"
    df.to_csv(path, index=False, float_format="%.6f")

    print(f"Shards read: {len(shards)}")
    print(f"Customers processed: {len(df):,}")
    print(f"Customers with no payroll dates: {(df.distinct_payroll_dates == 0).sum():,}")
    print(f"Customers with only 1–2 payroll dates: {df.distinct_payroll_dates.between(1, 2).sum():,}")
    print(f"Customers with regularity defined: {df.income_regularity.notna().sum():,}")
    print(f"Missing regularity values: {df.income_regularity.isna().sum():,}")
    print(f"Mean income regularity (defined): {df.income_regularity.mean():.3f} days")
    print(f"Median income regularity (defined): {df.income_regularity.median():.3f} days")
    print(f"Wrote: {path}")


# ===========================================================================
# INCOME VOLATILITY
# ===========================================================================


def month_id(year: int, month: int) -> int:
    """Monotonically increasing identifier for a calendar month."""
    return year * 12 + month - 1


def income_volatility_feature(bundle: dict) -> dict:
    """Compute coefficient of variation for observed monthly wage income."""
    cleaned, _ = deduplicate_transactions(bundle.get("transactions") or [])
    window_tx = [(day, tx) for day, tx in cleaned if WINDOW_START <= day <= WINDOW_END]

    if not window_tx:
        return {
            "customer_id": bundle["customer_id"],
            "income_volatility": float("nan"),
            "months_observed": 0,
            "months_with_wages": 0,
            "mean_monthly_wages_unrounded": float("nan"),
        }

    first_day = window_tx[0][0]
    last_day = window_tx[-1][0]
    first_month = month_id(first_day.year, first_day.month)
    last_month = month_id(last_day.year, last_day.month)
    month_wages = defaultdict(float)

    for day, tx in window_tx:
        category = tx.get("personal_finance_category") or {}
        if (
            category.get("primary") == "INCOME"
            and category.get("detailed") == WAGE_INCOME_DETAIL
            and float(tx["amount"]) < 0
        ):
            month_wages[month_id(day.year, day.month)] -= float(tx["amount"])

    # Zero-wage months count, provided we observe transaction activity across
    # the span; otherwise irregularity would be artificially understated.
    monthly_wages = [month_wages[m] for m in range(first_month, last_month + 1)]
    monthly_mean = mean(monthly_wages)
    volatility = (
        stdev(monthly_wages) / monthly_mean
        if len(monthly_wages) >= 2 and monthly_mean > 0
        else float("nan")
    )

    return {
        "customer_id": bundle["customer_id"],
        "income_volatility": volatility,
        "months_observed": len(monthly_wages),
        "months_with_wages": sum(w > 0 for w in monthly_wages),
        "mean_monthly_wages_unrounded": monthly_mean,
    }


def income_volatility() -> None:
    shards = sorted(DATA_DIR.glob("plaid_synth_shard_*.ndjson.gz"))
    if len(shards) != 50:
        raise ValueError(f"Expected 50 shards in {DATA_DIR}, found {len(shards)}")

    applications = pd.read_csv(DATA_DIR / "applications.csv", usecols=["customer_id", "application_date"])
    if not applications.customer_id.is_unique:
        raise ValueError("Duplicate customer_id in applications.csv")
    if (pd.to_datetime(applications.application_date).dt.date <= WINDOW_END).any():
        raise ValueError("Some applications precede the observation-window end")

    records = []
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(income_volatility_feature(json.loads(line)))

    features = pd.DataFrame(records)
    if not features.customer_id.is_unique or set(features.customer_id) != set(applications.customer_id):
        raise ValueError("Applicant IDs and unique JSON customer IDs do not match")
    if (features.income_volatility.dropna() < 0).any():
        raise ValueError("Volatility must be nonnegative")
    if not features.income_volatility.dropna().map(isfinite).all():
        raise ValueError("Volatility must be finite when defined")
    if not features.loc[features.mean_monthly_wages_unrounded == 0, "income_volatility"].isna().all():
        raise ValueError("Zero wage income must yield missing volatility")

    # Confirm that income-volatility monthly wage mean matches the payroll feature.
    previous = OUTPUT_DIR / "payroll_features.csv"
    if previous.exists():
        payroll = pd.read_csv(previous, usecols=["customer_id", "monthly_payroll_income", "months_observed"])
        joined = features.merge(payroll, on="customer_id", suffixes=("", "_payroll"), validate="one_to_one")
        if len(joined) != len(features):
            raise ValueError("Income-volatility and payroll features have mismatched customer IDs")
        if not (joined.months_observed == joined.months_observed_payroll).all():
            raise ValueError("Income-volatility and payroll features have inconsistent observation windows")
        discrepancy = (joined.mean_monthly_wages_unrounded - joined.monthly_payroll_income).abs()
        if not (discrepancy <= 0.005001).all():
            raise ValueError("Income-volatility and payroll features disagree on mean monthly payroll income")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "income_volatility_features.csv"
    features.to_csv(path, index=False, float_format="%.6f")

    print(f"Shards read: {len(shards)}")
    print(f"Customers processed: {len(features):,}")
    print(f"Customers with zero observed wage income: {(features.mean_monthly_wages_unrounded == 0).sum():,}")
    print(f"Customers with calculated volatility: {features.income_volatility.notna().sum():,}")
    print(f"Missing volatility values: {features.income_volatility.isna().sum():,}")
    print(f"Mean income volatility: {features.income_volatility.mean():.4f}")
    print(f"Median income volatility: {features.income_volatility.median():.4f}")
    print(f"Wrote: {path}")


# ===========================================================================
# SPENDING-TO-INCOME
# ===========================================================================


TRANSFER_CATEGORIES = {"TRANSFER_OUT", "TRANSFER_IN"}


def spending_to_income_feature(bundle: dict) -> dict:
    cleaned, duplicates = deduplicate_transactions(bundle.get("transactions") or [])
    window_tx = [(day, tx) for day, tx in cleaned if WINDOW_START <= day <= WINDOW_END]

    if not window_tx:
        return {
            "customer_id": bundle["customer_id"],
            "spending_to_income_ratio": float("nan"),
            "avg_monthly_spending": float("nan"),
            "monthly_payroll_income_unrounded": float("nan"),
            "months_observed": 0,
            "spending_transactions": 0,
            "excluded_transfer_outflows": 0,
            "duplicate_transactions_removed": duplicates,
        }

    first, last = window_tx[0][0], window_tx[-1][0]
    months_observed = (last.year - first.year) * 12 + last.month - first.month + 1

    spending_total = 0.0
    payroll_total = 0.0
    spending_count = 0
    excluded_transfers = 0
    for _, tx in window_tx:
        category = tx.get("personal_finance_category") or {}
        primary = category.get("primary")
        detailed = category.get("detailed")
        amount = float(tx["amount"])

        if primary == "INCOME" and detailed == WAGE_INCOME_DETAIL and amount < 0:
            payroll_total -= amount

        # Positive Plaid amounts are account outflows. Do not mistake
        # internal transfers into savings for consumption/expenditures.
        if amount > 0:
            if primary in TRANSFER_CATEGORIES:
                excluded_transfers += 1
            else:
                spending_total += amount
                spending_count += 1

    avg_spending = spending_total / months_observed
    avg_payroll = payroll_total / months_observed
    ratio = avg_spending / avg_payroll if avg_payroll > 0 else float("nan")

    return {
        "customer_id": bundle["customer_id"],
        "spending_to_income_ratio": ratio,
        "avg_monthly_spending": avg_spending,
        "monthly_payroll_income_unrounded": avg_payroll,
        "months_observed": months_observed,
        "spending_transactions": spending_count,
        "excluded_transfer_outflows": excluded_transfers,
        "duplicate_transactions_removed": duplicates,
    }


def spending_to_income() -> None:
    shards = sorted(DATA_DIR.glob("plaid_synth_shard_*.ndjson.gz"))
    if len(shards) != 50:
        raise ValueError(f"Expected 50 shards in {DATA_DIR}, found {len(shards)}")

    apps = pd.read_csv(DATA_DIR / "applications.csv", usecols=["customer_id", "application_date"])
    if not apps.customer_id.is_unique:
        raise ValueError("Duplicate customer_id in applications.csv")
    if (pd.to_datetime(apps.application_date).dt.date <= WINDOW_END).any():
        raise ValueError("An application predates the feature observation cutoff")

    records = []
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(spending_to_income_feature(json.loads(line)))

    features = pd.DataFrame(records)
    if not features.customer_id.is_unique or set(features.customer_id) != set(apps.customer_id):
        raise ValueError("Applicant IDs and unique JSON customer IDs do not match")
    if not features.spending_to_income_ratio.dropna().map(isfinite).all():
        raise ValueError("Defined spending-to-income ratios must be finite")
    if (features.spending_to_income_ratio.dropna() < 0).any():
        raise ValueError("Spending-to-income ratios must be nonnegative")
    if not features.loc[features.monthly_payroll_income_unrounded == 0,
                        "spending_to_income_ratio"].isna().all():
        raise ValueError("Zero payroll income must produce a missing ratio")

    # Verify spending-to-income uses the same payroll and observation-window
    # definitions used by the payroll feature.
    previous_path = OUTPUT_DIR / "payroll_features.csv"
    if previous_path.exists():
        payroll = pd.read_csv(previous_path, usecols=["customer_id", "monthly_payroll_income", "months_observed"])
        joined = features.merge(payroll, on="customer_id", suffixes=("", "_payroll"), validate="one_to_one")
        if len(joined) != len(features):
            raise ValueError("Payroll and spending-to-income customer IDs differ")
        if not (joined.months_observed == joined.months_observed_payroll).all():
            raise ValueError("Payroll and spending-to-income observation windows disagree")
        difference = (joined.monthly_payroll_income_unrounded - joined.monthly_payroll_income).abs()
        if not (difference <= 0.005001).all():
            raise ValueError("Payroll and spending-to-income monthly payroll income disagree")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "spending_to_income_features.csv"
    features.to_csv(path, index=False, float_format="%.6f")

    print(f"Shards read: {len(shards)}")
    print(f"Customers processed: {len(features):,}")
    print(f"Customers with a defined ratio: {features.spending_to_income_ratio.notna().sum():,}")
    print(f"Customers with zero payroll income: {(features.monthly_payroll_income_unrounded == 0).sum():,}")
    print(f"Missing ratios: {features.spending_to_income_ratio.isna().sum():,}")
    print(f"Transfer outflows excluded: {features.excluded_transfer_outflows.sum():,}")
    print(f"Mean monthly spending: ${features.avg_monthly_spending.mean():,.2f}")
    print(f"Median monthly spending: ${features.avg_monthly_spending.median():,.2f}")
    print(f"Mean spending-to-income ratio: {features.spending_to_income_ratio.mean():.4f}")
    print(f"Median spending-to-income ratio: {features.spending_to_income_ratio.median():.4f}")
    print(f"Wrote: {path}")


# ===========================================================================
# CASH BUFFER
# ===========================================================================


# Operational definition of 'essential' consistent with the feature's
# hypothesis about fixed/necessary expenses. These categories are
# observable in Group 07's synthetic Plaid transaction records.
ESSENTIAL_PRIMARY = {"RENT_AND_UTILITIES", "LOAN_PAYMENTS", "MEDICAL"}
ESSENTIAL_DETAILED = {
    "FOOD_AND_DRINK_GROCERIES",
    "TRANSPORTATION_GAS",
}


def is_essential(tx: dict) -> bool:
    category = tx.get("personal_finance_category") or {}
    return (category.get("primary") in ESSENTIAL_PRIMARY
            or category.get("detailed") in ESSENTIAL_DETAILED)


def reconstruct_median_checking_balance(
    *,
    ending_balances: dict[str, float],
    cleaned_tx: list[tuple[date, dict]],
    observation_start: date,
    observation_end: date,
) -> tuple[float, int]:
    """Median daily, END-OF-DAY checking balance, pooled across accounts.

    Let a_t be Plaid's signed outflow on day t and B_end be the observed
    2026-07-31 checking balance. Then

        B_day = B_end + sum(a_t for t > day)

    (reversing a +100 expense increases the earlier balance by $100).
    Use calendar days, not just transaction days. No credit-card/savings
    balances enter the feature; their movements are not checking cash.
    """
    if not ending_balances or observation_start > observation_end:
        return float("nan"), 0

    checking_ids = set(ending_balances)
    flows_by_date: defaultdict[date, float] = defaultdict(float)
    for day, tx in cleaned_tx:
        if observation_start <= day <= observation_end and tx.get("account_id") in checking_ids:
            flows_by_date[day] += float(tx["amount"])

    # Build from the latest day backward; the end-of-period current balance
    # is treated as the closing balance after transactions on WINDOW_END.
    current = sum(ending_balances.values())
    balances = []
    cursor = observation_end
    while cursor >= observation_start:
        balances.append(current)
        current += flows_by_date[cursor]   # undo today's net outflow
        cursor -= timedelta(days=1)

    # No numpy dependency needed; sorted median works for <= 181 days.
    balances.sort()
    n = len(balances)
    median = balances[n // 2] if n % 2 else (balances[n // 2 - 1] + balances[n // 2]) / 2
    return median, n


def cash_buffer_feature(bundle: dict) -> dict:
    customer_id = bundle["customer_id"]
    cleaned, n_duplicate = deduplicate_transactions(bundle.get("transactions") or [])
    tx_in_window = [(d, t) for d, t in cleaned if WINDOW_START <= d <= WINDOW_END]

    # Refer only to current (not available) depository/checking balances.
    checking_accounts = [
        a for a in (bundle.get("accounts") or [])
        if a.get("type") == "depository" and a.get("subtype") == "checking"
    ]
    ending_balances = {}
    for account in checking_accounts:
        current = (account.get("balances") or {}).get("current")
        if current is not None and isfinite(float(current)):
            ending_balances[account["account_id"]] = float(current)

    checking_tx = [
        (day, tx) for day, tx in tx_in_window
        if tx.get("account_id") in ending_balances
    ]

    # Reconstruct only a period supported by dated checking transactions.
    # A short-file applicant may have just 3-6 months of transaction history.
    observation_start = checking_tx[0][0] if checking_tx else None
    median_balance, reconstructed_days = (
        reconstruct_median_checking_balance(
            ending_balances=ending_balances,
            cleaned_tx=checking_tx,
            observation_start=observation_start,
            observation_end=WINDOW_END,
        ) if observation_start is not None
        else (float("nan"), 0)
    )

    # Match previous feature's denominator: first to last observed
    # transaction calendar months inside the six-month observation window.
    # Exclude internal transfers, restaurant meals, discretionary spending
    # and overdraft fees from the essential expense definition.
    months_observed = (
        ((tx_in_window[-1][0].year - tx_in_window[0][0].year) * 12
         + tx_in_window[-1][0].month - tx_in_window[0][0].month + 1)
        if tx_in_window else 0
    )
    essential_total = 0.0
    essential_count = 0
    for day, tx in tx_in_window:
        amount = float(tx["amount"])
        if amount > 0 and is_essential(tx):
            essential_total += amount
            essential_count += 1

    avg_essential = essential_total / months_observed if months_observed else float("nan")
    cash_buffer = (
        median_balance / avg_essential
        if (isfinite(median_balance) and isfinite(avg_essential) and avg_essential > 0)
        else float("nan")
    )

    return {
        "customer_id": customer_id,
        "cash_buffer": cash_buffer,
        "median_daily_checking_balance": median_balance,
        "avg_monthly_essential_expenses": avg_essential,
        "essential_expense_transactions": essential_count,
        "months_observed": months_observed,
        "days_reconstructed": reconstructed_days,
        "checking_accounts": len(ending_balances),
        "duplicate_transactions_removed": n_duplicate,
    }


def _unit_test_reconstruction() -> None:
    """Known-balance arithmetic with a missing-transaction day."""
    start, end = date(2026, 7, 29), date(2026, 7, 31)
    # $200 before 7/29. Pay $100 on 7/29, receive $50 on 7/31.
    # EOD balances = [100, 100, 150], median = 100.
    test_tx = [(start, {"account_id": "c1", "amount": 100.0}),
               (end, {"account_id": "c1", "amount": -50.0})]
    median, days = reconstruct_median_checking_balance(
        ending_balances={"c1": 150.0}, cleaned_tx=test_tx,
        observation_start=start, observation_end=end,
    )
    assert median == 100.0 and days == 3, (median, days)


def cash_buffer() -> None:
    _unit_test_reconstruction()
    shards = sorted(DATA_DIR.glob("plaid_synth_shard_*.ndjson.gz"))
    if len(shards) != 50:
        raise ValueError(f"Expected 50 shards in {DATA_DIR}, found {len(shards)}")

    apps = pd.read_csv(DATA_DIR / "applications.csv", usecols=["customer_id", "application_date"])
    if not apps.customer_id.is_unique:
        raise ValueError("Duplicate customer_id in applications.csv")
    if (pd.to_datetime(apps.application_date).dt.date <= WINDOW_END).any():
        raise ValueError("An application predates or falls within the feature observation window")

    rows = []
    for shard_number, shard in enumerate(shards, start=1):
        with gzip.open(shard, "rb") as stream:
            for line in stream:
                if line.strip():
                    rows.append(cash_buffer_feature(
                        orjson.loads(line) if orjson is not None else json.loads(line)
                    ))
        if shard_number % 10 == 0:
            print(f"Shards processed: {shard_number}/{len(shards)}", flush=True)

    df = pd.DataFrame(rows)
    if not df.customer_id.is_unique or set(df.customer_id) != set(apps.customer_id):
        raise ValueError("Duplicate or mismatched customer IDs")
    if not df.cash_buffer.dropna().map(isfinite).all():
        raise ValueError("Cash buffer contains non-finite defined values")
    if (df.avg_monthly_essential_expenses.dropna() < 0).any():
        raise ValueError("Essential expenses cannot be negative")
    if not df.loc[df.avg_monthly_essential_expenses.eq(0), "cash_buffer"].isna().all():
        raise ValueError("Zero essentials must produce an undefined buffer ratio")
    if (df.days_reconstructed > (WINDOW_END - WINDOW_START).days + 1).any():
        raise ValueError("History extended outside six-month feature window")

    previous_path = OUTPUT_DIR / "spending_to_income_features.csv"
    if previous_path.exists():
        prev = pd.read_csv(previous_path, usecols=["customer_id", "months_observed", "duplicate_transactions_removed"])
        checked = df.merge(prev, on="customer_id", suffixes=("", "_spending"), validate="one_to_one")
        if len(checked) != len(df):
            raise ValueError("Spending-to-income features have missing customer IDs")
        if not (checked.months_observed == checked.months_observed_spending).all():
            raise ValueError("Observation months differ from spending-to-income features")
        if not (checked.duplicate_transactions_removed == checked.duplicate_transactions_removed_spending).all():
            raise ValueError("Deduplication differs from spending-to-income features")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = OUTPUT_DIR / "cash_buffer_features.csv"
    df.to_csv(result, index=False, float_format="%.6f")

    print(f"Shards: {len(shards)}")
    print(f"Customers processed: {len(df):,}")
    print(f"Valid cash buffer: {df.cash_buffer.notna().sum():,}")
    print(f"Missing cash buffer: {df.cash_buffer.isna().sum():,}")
    print(f"Customers without checking accounts: {(df.checking_accounts == 0).sum():,}")
    print(f"Median reconstructed checking balance: ${df.median_daily_checking_balance.median():,.2f}")
    print(f"Median essential monthly expenses: ${df.avg_monthly_essential_expenses.median():,.2f}")
    print(f"Mean cash buffer: {df.cash_buffer.mean():.4f}")
    print(f"Median cash buffer: {df.cash_buffer.median():.4f}")
    print(f"Negative cash buffers: {(df.cash_buffer < 0).sum():,}")
    print(f"Wrote: {result}")


# ===========================================================================
# EMPLOYMENT TENURE
# ===========================================================================


def employment_tenure() -> None:
    applications = pd.read_csv(
        DATA_DIR / "applications.csv",
        usecols=["customer_id", "employment_months"],
    )
    if applications["customer_id"].isna().any():
        raise ValueError("applications.csv contains missing customer IDs")
    if not applications["customer_id"].is_unique:
        raise ValueError("applications.csv contains duplicate customer IDs")

    raw = applications["employment_months"]
    tenure = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & tenure.isna()
    if invalid.any():
        raise ValueError(f"Invalid nonnumeric employment_months values: {invalid.sum()}")
    if (~np.isfinite(tenure.dropna())).any():
        raise ValueError("employment_months contains non-finite values")
    if (tenure.dropna() < 0).any():
        raise ValueError("employment_months contains negative values")
    if ((tenure.dropna() % 1) != 0).any():
        raise ValueError("employment_months contains fractional months")

    features = pd.DataFrame({
        "customer_id": applications["customer_id"],
        "employment_tenure_months": tenure.astype("Int64"),
    })

    # Confirm that the same 25,000 applicants appear in the independently
    # produced cash-buffer features, if that output is present.
    previous_path = OUTPUT_DIR / "cash_buffer_features.csv"
    if previous_path.exists():
        prior = pd.read_csv(previous_path, usecols=["customer_id"])
        if (not prior.customer_id.is_unique
                or set(prior.customer_id) != set(features.customer_id)):
            raise ValueError("Applicant ID mismatch with cash-buffer output")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "employment_tenure_features.csv"
    features.to_csv(output_path, index=False)

    s = features.employment_tenure_months
    print(f"Applicants processed: {len(features):,}")
    print(f"Missing employment tenure: {s.isna().sum():,}")
    print(f"Minimum employment tenure: {s.min()} months")
    print(f"Maximum employment tenure: {s.max()} months")
    print(f"Mean employment tenure: {s.mean():.2f} months")
    print(f"Median employment tenure: {s.median():.0f} months")
    print(f"Wrote: {output_path}")


# ===========================================================================
# INCOME DISCREPANCY
# ===========================================================================


def income_discrepancy() -> None:
    applications = pd.read_csv(
        DATA_DIR / "applications.csv",
        usecols=["customer_id", "stated_monthly_income"],
    )

    payroll_path = OUTPUT_DIR / "payroll_features.csv"
    if not payroll_path.is_file():
        raise FileNotFoundError(
            f"Missing {payroll_path}; run this consolidated script from its entry point."
        )
    payroll = pd.read_csv(
        payroll_path,
        usecols=["customer_id", "monthly_payroll_income", "payroll_deposits"],
    )

    for label, frame in (("applications.csv", applications),
                         ("payroll_features.csv", payroll)):
        if frame.customer_id.isna().any() or not frame.customer_id.is_unique:
            raise ValueError(f"Missing or duplicate customer_id in {label}")

    if len(applications) != len(payroll) or set(applications.customer_id) != set(payroll.customer_id):
        raise ValueError("Applicant IDs do not match between applications and payroll files")

    data = applications.merge(payroll, on="customer_id", how="left", validate="one_to_one")

    for col in ("stated_monthly_income", "monthly_payroll_income"):
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all():
            raise ValueError(f"Missing/nonfinite values in {col}")
        if (data[col] < 0).any():
            raise ValueError(f"Negative values in {col}")

    # Ratios with zero payroll have no defined denominator; preserve NaN.
    payroll_amount = data["monthly_payroll_income"].to_numpy(dtype=float)
    stated_amount = data["stated_monthly_income"].to_numpy(dtype=float)
    valid_payroll = payroll_amount > 0

    discrepancy = np.full(len(data), np.nan, dtype=float)
    discrepancy[valid_payroll] = (
        np.maximum(0.0, stated_amount[valid_payroll] - payroll_amount[valid_payroll])
        / payroll_amount[valid_payroll]
    )

    features = data.assign(income_discrepancy=discrepancy)[
        ["customer_id", "income_discrepancy", "stated_monthly_income",
         "monthly_payroll_income", "payroll_deposits"]
    ]

    assert len(features) == len(applications)
    assert features.customer_id.is_unique
    assert features.income_discrepancy.notna().sum() == int(valid_payroll.sum())
    assert (features.income_discrepancy.dropna() >= 0).all()
    assert (features.loc[valid_payroll, "income_discrepancy"] == 0).equals(
        pd.Series(stated_amount[valid_payroll] <= payroll_amount[valid_payroll],
                  index=features.index[valid_payroll])
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "income_discrepancy_features.csv"
    features.to_csv(output_path, index=False, float_format="%.8f")

    observed = features.income_discrepancy.dropna()
    print(f"Applicants processed: {len(features):,}")
    print(f"Applicants with a calculated discrepancy: {len(observed):,}")
    print(f"Applicants without identified payroll (undefined ratio): {(~valid_payroll).sum():,}")
    print(f"Stated income greater than observed payroll: {(observed > 0).sum():,}")
    print(f"Stated income at/below observed payroll: {(observed == 0).sum():,}")
    print(f"Mean income discrepancy: {observed.mean():.4f}")
    print(f"Median income discrepancy: {observed.median():.4f}")
    print(f"Maximum income discrepancy: {observed.max():.4f}")
    print(f"Wrote: {output_path}")


# ===========================================================================
# HOUSING BURDEN
# ===========================================================================


RENT_CATEGORY = "RENT_AND_UTILITIES_RENT"
MORTGAGE_CATEGORY = "LOAN_PAYMENTS_MORTGAGE_PAYMENT"


def observed_months(cleaned: list[tuple]) -> int:
    """Calendar months from first to last observed transaction in window."""
    dates = [day for day, _ in cleaned if WINDOW_START <= day <= WINDOW_END]
    if not dates:
        return 0
    first, last = min(dates), max(dates)
    return (last.year - first.year) * 12 + (last.month - first.month) + 1


def mortgage_liability_monthly_payment(bundle: dict) -> float:
    """Sum listed loan payments without double-counting a loan's fields."""
    liabilities = bundle.get("liabilities") or {}
    mortgages = liabilities.get("mortgage") or []
    total = 0.0
    valid = False
    for mortgage in mortgages:
        next_payment = mortgage.get("next_monthly_payment")
        last_payment = mortgage.get("last_payment_amount")
        for raw in (next_payment, last_payment):
            if raw is not None:
                value = float(raw)
                if isfinite(value) and value > 0:
                    total += value
                    valid = True
                    break  # one monthly payment per mortgage account
    return total if valid else float("nan")


def housing_payment_info(bundle: dict, housing_status: str) -> dict:
    """Calculate housing payment amount from transactions/liability fallback."""
    if housing_status not in {"rent", "mortgage"}:
        raise ValueError(f"Unexpected housing_status {housing_status!r} for {bundle['customer_id']}")

    clean_tx, duplicate_count = deduplicate_transactions(bundle.get("transactions") or [])
    window_tx = [(day, tx) for day, tx in clean_tx if WINDOW_START <= day <= WINDOW_END]
    n_months = observed_months(window_tx)
    housing_detail = RENT_CATEGORY if housing_status == "rent" else MORTGAGE_CATEGORY

    relevant_amounts = []
    opposite_count = 0
    for _, tx in window_tx:
        category = tx.get("personal_finance_category") or {}
        detail = category.get("detailed")
        amount = float(tx["amount"])
        if not isfinite(amount) or amount <= 0:
            continue
        if detail == housing_detail:
            relevant_amounts.append(amount)
        elif detail in {RENT_CATEGORY, MORTGAGE_CATEGORY}:
            opposite_count += 1

    liability_payment = mortgage_liability_monthly_payment(bundle)
    if relevant_amounts and n_months:
        avg_monthly_housing = sum(relevant_amounts) / n_months
        source = "transactions"
    elif housing_status == "mortgage" and isfinite(liability_payment):
        avg_monthly_housing = liability_payment
        source = "mortgage_liabilities"
    else:
        avg_monthly_housing = float("nan")
        source = "missing"

    return {
        "customer_id": bundle["customer_id"],
        "avg_monthly_housing_payment": avg_monthly_housing,
        "housing_payment_source": source,
        "housing_transaction_count": len(relevant_amounts),
        "opposite_housing_transaction_count": opposite_count,
        "mortgage_liability_monthly_payment": liability_payment,
        "months_observed": n_months,
        "duplicate_transactions_removed": duplicate_count,
    }


def test_housing_rules() -> None:
    def transaction(date_str, amount, detailed):
        return {
            "date": date_str, "amount": amount, "account_id": "checking1",
            "name": detailed, "pending": False,
            "personal_finance_category": {"primary": "RENT_AND_UTILITIES", "detailed": detailed},
        }

    data = {
        "customer_id": "test",
        "transactions": [
            transaction("2026-02-02", 1200, RENT_CATEGORY),
            transaction("2026-03-02", 1200, RENT_CATEGORY),
            transaction("2026-03-04", 100, "RENT_AND_UTILITIES_GAS_AND_ELECTRICITY"),
        ],
        "liabilities": {"mortgage": []},
    }
    res = housing_payment_info(data, "rent")
    assert res["months_observed"] == 2
    assert res["avg_monthly_housing_payment"] == 1200
    assert res["housing_transaction_count"] == 2
    assert res["housing_payment_source"] == "transactions"

    data["transactions"] = [transaction("2026-03-02", 1550, MORTGAGE_CATEGORY)]
    data["liabilities"] = {"mortgage": [{"next_monthly_payment": 1800,
                                         "last_payment_amount": 1800}]}
    res = housing_payment_info(data, "mortgage")
    assert res["avg_monthly_housing_payment"] == 1550  # prefer transactions, not both
    assert res["housing_payment_source"] == "transactions"

    data["transactions"] = [transaction("2026-02-02", 50, "FOOD_AND_DRINK_GROCERIES")]
    res = housing_payment_info(data, "mortgage")
    assert res["avg_monthly_housing_payment"] == 1800
    assert res["housing_payment_source"] == "mortgage_liabilities"
    res = housing_payment_info(data, "rent")
    assert np.isnan(res["avg_monthly_housing_payment"])


def housing_burden() -> None:
    test_housing_rules()
    shards = sorted(DATA_DIR.glob("plaid_synth_shard_*.ndjson.gz"))
    if len(shards) != 50:
        raise ValueError(f"Expected 50 shards in {DATA_DIR}; found {len(shards)}")

    apps = pd.read_csv(DATA_DIR / "applications.csv",
                       usecols=["customer_id", "application_date", "housing_status"])
    if not apps.customer_id.is_unique or apps.customer_id.isna().any():
        raise ValueError("Applications contain missing or duplicate customer_id")
    if (pd.to_datetime(apps.application_date).dt.date <= WINDOW_END).any():
        raise ValueError("Housing features would include post-application data")

    payroll_file = OUTPUT_DIR / "payroll_features.csv"
    if not payroll_file.is_file():
        raise FileNotFoundError(f"Missing {payroll_file}; run this consolidated script from its entry point.")
    payroll = pd.read_csv(payroll_file,
                          usecols=["customer_id", "monthly_payroll_income", "months_observed"])
    if not payroll.customer_id.is_unique or payroll.customer_id.isna().any():
        raise ValueError("Payroll file contains missing or duplicate customer_id")

    status_lookup = dict(zip(apps.customer_id, apps.housing_status))
    rows = []
    for shard_number, shard in enumerate(shards, start=1):
        with gzip.open(shard, "rb") as stream:
            for line in stream:
                if line.strip():
                    bundle = orjson.loads(line) if orjson is not None else json.loads(line)
                    customer_id = bundle["customer_id"]
                    if customer_id not in status_lookup:
                        raise ValueError(f"Unknown customer_id in shard: {customer_id}")
                    rows.append(housing_payment_info(bundle, status_lookup[customer_id]))
        if shard_number % 10 == 0:
            print(f"Shards processed: {shard_number}/{len(shards)}", flush=True)

    extracted = pd.DataFrame(rows)
    if not extracted.customer_id.is_unique or set(extracted.customer_id) != set(apps.customer_id):
        raise ValueError("Housing feature customers do not match applications")

    features = (apps[["customer_id", "housing_status"]]
                .merge(extracted, on="customer_id", validate="one_to_one")
                .merge(payroll, on="customer_id", validate="one_to_one",
                       suffixes=("", "_payroll")))
    if len(features) != len(apps):
        raise ValueError("Lost customer IDs during joins")
    if not (features.months_observed == features.months_observed_payroll).all():
        raise ValueError("Payroll and housing-burden observation windows disagree")

    # Inapplicable/undefined ratios are NA, never 0 or infinite.
    denominator = features.monthly_payroll_income.to_numpy(dtype=float)
    housing = features.avg_monthly_housing_payment.to_numpy(dtype=float)
    valid = np.isfinite(denominator) & (denominator > 0) & np.isfinite(housing) & (housing > 0)
    ratio = np.full(len(features), np.nan)
    ratio[valid] = housing[valid] / denominator[valid]
    features.insert(1, "housing_payment_burden", ratio)
    if not np.isfinite(features.housing_payment_burden.dropna().to_numpy()).all():
        raise ValueError("Defined housing ratios are not finite")
    if (features.housing_payment_burden.dropna() < 0).any():
        raise ValueError("Negative housing ratios found")
    if not features.loc[features.monthly_payroll_income.eq(0), "housing_payment_burden"].isna().all():
        raise ValueError("Zero payroll income must have missing housing burden")

    # Cross-check deduplication and window rules against payroll extraction.
    # Already produced by the payroll extraction above; here we check
    # 25,000 rows against the original preprocessing output.
    previous_dedup = pd.read_csv(payroll_file,
                                 usecols=["customer_id", "duplicate_transactions_removed"])
    checked = features.merge(previous_dedup, on="customer_id", validate="one_to_one",
                             suffixes=("", "_payroll"))
    if not (checked.duplicate_transactions_removed == checked.duplicate_transactions_removed_payroll).all():
        raise ValueError("Housing-burden and payroll transaction deduplication results differ")

    output_columns = [
        "customer_id", "housing_payment_burden", "housing_status",
        "avg_monthly_housing_payment", "monthly_payroll_income",
        "housing_payment_source", "housing_transaction_count",
        "opposite_housing_transaction_count", "mortgage_liability_monthly_payment",
        "months_observed", "duplicate_transactions_removed",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_file = OUTPUT_DIR / "housing_payment_burden_features.csv"
    features[output_columns].to_csv(result_file, index=False, float_format="%.8f")

    print(f"Applicants processed: {len(features):,}")
    print(f"Renters: {(features.housing_status == 'rent').sum():,}")
    print(f"Mortgage holders: {(features.housing_status == 'mortgage').sum():,}")
    print(f"Housing payments from transactions: {(features.housing_payment_source == 'transactions').sum():,}")
    print(f"Mortgage liability fallbacks: {(features.housing_payment_source == 'mortgage_liabilities').sum():,}")
    print(f"Missing housing payment amounts: {(features.housing_payment_source == 'missing').sum():,}")
    print(f"Without wage payroll: {(features.monthly_payroll_income == 0).sum():,}")
    print(f"Defined housing burden ratios: {features.housing_payment_burden.notna().sum():,}")
    print(f"Missing housing burden ratios: {features.housing_payment_burden.isna().sum():,}")
    print(f"Mean monthly housing payment: ${features.avg_monthly_housing_payment.mean():,.2f}")
    print(f"Median monthly housing payment: ${features.avg_monthly_housing_payment.median():,.2f}")
    print(f"Mean housing payment burden: {features.housing_payment_burden.mean():.4f}")
    print(f"Median housing payment burden: {features.housing_payment_burden.median():.4f}")
    print(f"Wrote: {result_file}")


# ===========================================================================
# MODEL TABLE
# ===========================================================================


# Paths to adjust for the grader (no other path changes required).

FEATURE_SOURCES = {
    "income_regularity_features.csv": ["income_regularity"],
    "income_volatility_features.csv": ["income_volatility"],
    "spending_to_income_features.csv": ["spending_to_income_ratio"],
    "cash_buffer_features.csv": ["cash_buffer"],
    "employment_tenure_features.csv": ["employment_tenure_months"],
    "income_discrepancy_features.csv": ["income_discrepancy", "monthly_payroll_income"],
    "housing_payment_burden_features.csv": ["housing_payment_burden"],
}

FEATURE_COLUMNS = [
    "income_regularity",
    "income_volatility",
    "spending_to_income_ratio",
    "cash_buffer",
    "employment_tenure_months",
    "income_discrepancy",
    "monthly_payroll_income",
    "housing_payment_burden",
]


def read_unique_ids(path: Path, usecols: list[str]) -> pd.DataFrame:
    """Read a table and reject missing or repeated customer IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    frame = pd.read_csv(path, usecols=usecols, dtype={"customer_id": "string"})
    if frame["customer_id"].isna().any() or frame["customer_id"].duplicated().any():
        raise ValueError(f"Missing/duplicate customer_id in {path.name}")
    return frame


def model_table() -> None:
    applications = read_unique_ids(
        DATA_DIR / "applications.csv",
        ["customer_id", "approved", "took_up"],
    )
    outcomes = read_unique_ids(
        DATA_DIR / "outcomes_train.csv", ["customer_id", "default_12m"]
    )
    holdout = read_unique_ids(DATA_DIR / "holdout_ids.csv", ["customer_id"])

    if len(applications) != 25_000 or len(holdout) != 5_000:
        raise ValueError("Unexpected applicant/holdout count")
    applicants = set(applications["customer_id"])
    labeled_ids = set(outcomes["customer_id"])
    pending_ids = set(holdout["customer_id"])
    if not labeled_ids <= applicants or not pending_ids <= applicants:
        raise ValueError("Outcome or pending customer IDs absent from applications")
    if labeled_ids & pending_ids:
        raise ValueError("Labeled outcomes overlap the pending queue")
    if not outcomes["default_12m"].isin([0, 1]).all():
        raise ValueError("Outcome column must contain only 0/1")

    # Use approved/took_up ONLY to verify outcome-file provenance, not to
    # select the training sample or as predictors, imputers, feature-selection
    # inputs, or segmentation variables. The label file alone defines training
    # eligibility; group is never loaded into any modeling table.
    # Validate that labels originate exclusively from booked historical loans.
    booked = applications.loc[
        (applications["approved"] == 1) & (applications["took_up"] == 1),
        "customer_id",
    ]
    if set(booked) != labeled_ids:
        raise ValueError("Outcome IDs do not exactly match approved and taken-up loans")
    pending_funnel = applications.set_index("customer_id").loc[holdout["customer_id"]]
    if not pending_funnel[["approved", "took_up"]].isna().all().all():
        raise ValueError("Pending applicants unexpectedly have incumbent decisions")

    features = applications[["customer_id"]].copy()
    for filename, columns in FEATURE_SOURCES.items():
        part = read_unique_ids(OUTPUT_DIR / filename, ["customer_id", *columns])
        if len(part) != len(applications) or set(part["customer_id"]) != applicants:
            raise ValueError(f"Feature file has incomplete/different IDs: {filename}")
        features = features.merge(part, on="customer_id", how="left", validate="one_to_one")

    features = features[["customer_id", *FEATURE_COLUMNS]]
    for col in FEATURE_COLUMNS:
        features[col] = pd.to_numeric(features[col], errors="raise")
        finite_or_missing = np.isfinite(features[col]) | features[col].isna()
        if not finite_or_missing.all():
            raise ValueError(f"Infinite values found in {col}")
    if (features["income_regularity"].dropna() > 0).any():
        raise ValueError("Income regularity must not be positive")
    for col in ("income_volatility", "spending_to_income_ratio", "employment_tenure_months",
                "income_discrepancy", "monthly_payroll_income", "housing_payment_burden"):
        if (features[col].dropna() < 0).any():
            raise ValueError(f"Unexpected negative value in {col}")
    # Cash buffers can be negative and are deliberately retained.

    # Cross-check payroll output if present. Do not round or change any values.
    payroll_path = OUTPUT_DIR / "payroll_features.csv"
    if payroll_path.is_file():
        payroll = read_unique_ids(payroll_path, ["customer_id", "monthly_payroll_income"])
        both = features[["customer_id", "monthly_payroll_income"]].merge(
            payroll, on="customer_id", validate="one_to_one", suffixes=("_model", "_extracted")
        )
        if len(both) != len(features) or not np.allclose(
            both["monthly_payroll_income_model"],
            both["monthly_payroll_income_extracted"],
            rtol=0, atol=1e-8, equal_nan=True
        ):
            raise ValueError("Monthly payroll income disagrees with the payroll extraction output")

    labeled = features.merge(outcomes, on="customer_id", how="inner", validate="one_to_one")
    pending = holdout.merge(features, on="customer_id", how="left", validate="one_to_one")
    if len(labeled) != len(outcomes) or len(pending) != len(holdout):
        raise ValueError("Rows lost/duplicated in labeled or pending join")
    if not pending["customer_id"].equals(holdout["customer_id"]):
        raise ValueError("Pending queue changed order")
    if set(labeled["customer_id"]) != labeled_ids:
        raise ValueError("Some labeled observations are missing")
    assert "group" not in features.columns
    assert not {"approved", "offered_apr", "took_up", "default_12m"} & set(features.columns)
    assert "default_12m" not in pending.columns
    assert len(FEATURE_COLUMNS) == 8

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = {
        "customer_features.csv": features,
        "labeled_model_data.csv": labeled,
        "pending_model_data.csv": pending,
    }
    for filename, df in written.items():
        path = OUTPUT_DIR / filename
        df.to_csv(path, index=False, float_format="%.10g")
        print(f"Saved {path}: {len(df):,} rows")

    print(f"\nLabeled default count: {int(labeled['default_12m'].sum()):,}")
    print(f"Labeled default rate: {labeled['default_12m'].mean():.2%}")
    print("\nMissing predictor values (labeled / pending):")
    for col in FEATURE_COLUMNS:
        print(f"  {col}: {labeled[col].isna().sum():,} / {pending[col].isna().sum():,}")
    print("\nAll customer-ID, label-selection, and feature checks: PASSED")
    print("No imputation or model training has been performed.")



# ===========================================================================
# STAGE 3: TWO-MODEL CHAMPION / CHALLENGER COMPARISON
# ===========================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

SEED = 6155
FEATURES = FEATURE_COLUMNS  # Exactly 8 pre-registered economic features.
REFERENCE_APPROVAL_FRACTION = 0.70
REMEDIATION_APPROVAL_FRACTION = 0.75
EXPECTED_LOAN_MARGIN_IF_REPAID = 720.0  # $8,000 principal x 9% margin.
LOSS_IF_DEFAULT = 6800.0               # $8,000 principal x 85% loss.


def make_logistic():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=5000,
            class_weight=None, random_state=SEED)),
    ])


def make_random_forest():
    # Conservative complexity relative to only 176 training defaults.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=25,
            max_features="sqrt", bootstrap=True, class_weight=None,
            n_jobs=-1, random_state=SEED)),
    ])


def discrimination(y, probabilities):
    fpr, tpr, _ = roc_curve(y, probabilities)
    return {
        "auc": float(roc_auc_score(y, probabilities)),
        "ks": float(np.max(tpr - fpr)),
        "brier": float(brier_score_loss(y, probabilities)),
        "mean_predicted_pd": float(np.mean(probabilities)),
        "observed_default_rate": float(np.mean(y)),
    }


def decile_calibration(ids, y, probabilities, model_name):
    d = pd.DataFrame({"customer_id": np.asarray(ids), "default_12m": np.asarray(y),
                      "pd": np.asarray(probabilities)})
    d = d.sort_values(["pd", "customer_id"], kind="mergesort").reset_index(drop=True)
    d["decile"] = np.arange(len(d)) * 10 // len(d) + 1
    d = d.groupby("decile", as_index=False).agg(
        n=("customer_id", "size"), defaults=("default_12m", "sum"),
        avg_predicted_pd=("pd", "mean"), observed_default_rate=("default_12m", "mean"))
    d.insert(0, "model", model_name)
    return d


def policy_approvals(scored: pd.DataFrame, fraction: float) -> np.ndarray:
    """Group-blind global policy: lowest PD first; stable ID tie-break.

    The approval fraction is global and fixed before looking at any person's
    group. It is NOT a group-specific approval threshold or a trained feature.
    """
    order = np.lexsort((scored.customer_id.astype(str).to_numpy(),
                        scored.pd.to_numpy(dtype=float)))
    approve = np.zeros(len(scored), dtype=int)
    approve[order[:int(round(fraction * len(scored)))]] = 1
    return approve


def fairness_table(scored: pd.DataFrame, validation_outcomes: pd.DataFrame,
                   model_name: str, fraction: float) -> pd.DataFrame:
    """Demographic parity on all 25,000; other metrics on untouched val only.

    Reporting only: group is joined after scores and approvals exist.
    Outcomes never get used to choose thresholds, fit, or score.
    """
    d = scored[["customer_id", "pd", "approve", "group"]].merge(
        validation_outcomes[["customer_id", "default_12m"]],
        on="customer_id", how="left", validate="one_to_one")
    rows = []
    for group in ("A", "B"):
        pop = d.loc[d.group.eq(group)]
        val = pop.loc[pop.default_12m.notna()]
        good = val.loc[val.default_12m.eq(0)]
        bad = val.loc[val.default_12m.eq(1)]
        appr = val.loc[val.approve.eq(1)]
        rows.append({
            "model": model_name,
            "global_approval_fraction": fraction,
            "group": group,
            "all_applicants_n": int(len(pop)),
            "all_applicants_approved": int(pop.approve.sum()),
            "demographic_parity_approval_rate": float(pop.approve.mean()),
            "heldout_labeled_n": int(len(val)),
            "heldout_nondefaults_n": int(len(good)),
            "equal_opportunity_approval_among_nondefaults": (
                float(good.approve.mean()) if len(good) else np.nan),
            "heldout_defaults_n": int(len(bad)),
            "equalized_odds_approval_among_defaults": (
                float(bad.approve.mean()) if len(bad) else np.nan),
            "heldout_approved_n": int(len(appr)),
            "predictive_parity_default_rate_among_approved": (
                float(appr.default_12m.mean()) if len(appr) else np.nan),
        })
    return pd.DataFrame(rows)


def policy_economics(scored: pd.DataFrame, model_name: str,
                     policy_label: str, fraction: float):
    # Expected profit is MODEL-IMPLIED, not observable on unlabeled applicants.
    # E[profit booked] = (1-PD)*720 - PD*6800 = 720 - 7520*PD.
    approved = scored.loc[scored.approve.eq(1), "pd"].to_numpy(dtype=float)
    return {
        "model": model_name, "policy": policy_label,
        "approval_fraction": fraction, "n_approved": int(len(approved)),
        "mean_predicted_pd_approved": float(approved.mean()),
        "expected_profit_per_booked_loan": float((720 - 7520 * approved).mean()),
        "expected_total_booked_profit": float((720 - 7520 * approved).sum()),
        "assumption": "Model-implied; $720 on repay, -$6,800 on default; ignores take-up and selection",
    }


def run_stage3():
    # Basic audit of labeled training population and independently held-out queue.
    labeled = pd.read_csv(OUTPUT_DIR / "labeled_model_data.csv",
                          dtype={"customer_id": "string"})
    features = pd.read_csv(OUTPUT_DIR / "customer_features.csv",
                           dtype={"customer_id": "string"})
    holdout = pd.read_csv(DATA_DIR / "holdout_ids.csv",
                          dtype={"customer_id": "string"})
    assert set(features.columns) == {"customer_id", *FEATURES}
    assert set(labeled.columns) == {"customer_id", "default_12m", *FEATURES}
    assert len(labeled) == 8141 and len(features) == 25000 and len(holdout) == 5000
    assert labeled.customer_id.is_unique and features.customer_id.is_unique
    assert holdout.customer_id.is_unique
    assert not (set(labeled.customer_id) & set(holdout.customer_id))
    assert labeled.default_12m.isin([0, 1]).all()

    y = labeled.default_12m.astype(int)
    train_idx, val_idx = train_test_split(
        np.arange(len(labeled)), test_size=.20, stratify=y, random_state=SEED)
    X_train = labeled.iloc[train_idx][FEATURES]
    y_train = y.iloc[train_idx]
    X_val = labeled.iloc[val_idx][FEATURES]
    y_val = y.iloc[val_idx]
    val_ids = labeled.iloc[val_idx].customer_id
    validation_outcomes = labeled.iloc[val_idx][["customer_id", "default_12m"]]

    # No protected group or incumbent approval/price/take-up may enter inputs.
    assert "group" not in FEATURES
    assert not set(FEATURES) & {"approved", "offered_apr", "took_up", "default_12m"}

    models = {"Logistic regression": make_logistic(),
              "Random forest": make_random_forest()}
    summary = []
    calibration = []
    scores = []
    val_predict = pd.DataFrame({"customer_id": val_ids.to_numpy(),
                                "default_12m": y_val.to_numpy()})
    fitted_val = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict_proba(X_val)[:, 1]
        if not (np.isfinite(pred).all() and ((pred > 0) & (pred < 1)).all()):
            raise ValueError(f"Invalid validation predictions for {name}")
        score = discrimination(y_val, pred)
        summary.append({"model": name, "n_train": len(train_idx),
                        "n_train_defaults": int(y_train.sum()),
                        "n_validation": len(val_idx),
                        "n_validation_defaults": int(y_val.sum()), **score})
        calibration.append(decile_calibration(val_ids, y_val, pred, name))
        col = "logistic_pd" if name == "Logistic regression" else "random_forest_pd"
        val_predict[col] = pred
        fitted_val[name] = model
        print(f"{name}: AUC={score['auc']:.4f}, KS={score['ks']:.4f}, "
              f"Brier={score['brier']:.5f}", flush=True)
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "stage3_model_comparison.csv",
                                  index=False, float_format="%.12g")
    pd.concat(calibration, ignore_index=True).to_csv(
        OUTPUT_DIR / "stage3_validation_calibration.csv", index=False,
        float_format="%.12g")
    val_predict.to_csv(OUTPUT_DIR / "stage3_validation_predictions.csv",
                       index=False, float_format="%.12g")

    # Permutation feature importance is descriptive and not used for selecting
    # inputs, hyperparameters, or the approval threshold.
    perm = permutation_importance(fitted_val["Random forest"], X_val, y_val,
                                  scoring="roc_auc", n_repeats=5,
                                  random_state=SEED, n_jobs=1)
    importance = pd.DataFrame({
        "feature": FEATURES,
        "validation_auc_loss_if_permuted": perm.importances_mean,
        "permutation_sd": perm.importances_std,
    }).sort_values("validation_auc_loss_if_permuted", ascending=False)
    importance.to_csv(OUTPUT_DIR / "stage3_random_forest_importance.csv",
                      index=False, float_format="%.12g")

    # Only now attach protected data for auditing. The numeric predictions and
    # group-blind policy are both finalized BEFORE this merge.
    monitoring = pd.read_csv(DATA_DIR / "applications.csv",
                             usecols=["customer_id", "group"],
                             dtype={"customer_id": "string"})
    if not monitoring.customer_id.is_unique or set(monitoring.group) != {"A", "B"}:
        raise ValueError("Invalid or repeated monitoring data")
    audit = []
    econ = []
    for name in models:
        scored = features[["customer_id"]].copy()
        scored["pd"] = fitted_val[name].predict_proba(features[FEATURES])[:, 1]
        for fraction in ([.70] if name == "Logistic regression" else [.70, .75]):
            temp = scored.assign(approve=policy_approvals(scored, fraction))
            # Note: neither group nor outcomes exist when 'approve' is built.
            audited = temp.merge(monitoring, on="customer_id", validate="one_to_one")
            audit.append(fairness_table(audited, validation_outcomes, name, fraction))
            policy_name = "Reference 70%" if fraction == .70 else "Remediation 75%"
            econ.append(policy_economics(temp, name, policy_name, fraction))
    pd.concat(audit, ignore_index=True).to_csv(
        OUTPUT_DIR / "stage3_fairness_audit_validation_fit.csv", index=False,
        float_format="%.12g")
    pd.DataFrame(econ).to_csv(OUTPUT_DIR / "stage3_remediation_economics.csv",
                              index=False, float_format="%.12g")

    # CHOICE: Random forest is champion for Stage 3. It slightly improves AUC
    # and Brier and has less 70%-policy approval-rate disparity than logistic.
    # Interpretability is weaker than logistic; defer deployment in practice
    # until individual adverse-action explanations are validated (Stage 4).
    champion = make_random_forest()
    champion.fit(labeled[FEATURES], y)
    pending = holdout[["customer_id"]].merge(
        features, on="customer_id", how="left", validate="one_to_one")
    if len(pending) != len(holdout) or pending[FEATURES].isna().all(axis=0).any():
        raise ValueError("Incomplete pending feature join")
    pending_pd = champion.predict_proba(pending[FEATURES])[:, 1]
    submission = pd.DataFrame({"customer_id": pending.customer_id.to_numpy(),
                               "pd": pending_pd})
    if not (np.isfinite(submission.pd).all() and
            ((submission.pd > 0) & (submission.pd < 1)).all() and
            submission.customer_id.is_unique and
            submission.customer_id.tolist() == holdout.customer_id.tolist()):
        raise ValueError("Submission violates Stage 3 contract")
    dest = OUTPUT_DIR / SUBMISSION_FILENAME
    submission.to_csv(dest, index=False, float_format="%.15g", lineterminator="\n")
    disk = pd.read_csv(dest, dtype={"customer_id": "string"})
    assert list(disk.columns) == ["customer_id", "pd"]
    assert len(disk) == 5000 and disk.customer_id.is_unique
    assert disk.customer_id.tolist() == holdout.customer_id.tolist()
    assert np.isfinite(disk.pd).all() and ((disk.pd > 0) & (disk.pd < 1)).all()

    # Refit model audit: all-applicant approval parity is directly estimable,
    # but labels used to fit the champion CANNOT be used for an honest
    # outcome-conditioned audit here. Keep separate from validation metrics.
    full = features[["customer_id"]].copy()
    full["pd"] = champion.predict_proba(features[FEATURES])[:, 1]
    full_audits = []
    for f in [REFERENCE_APPROVAL_FRACTION, REMEDIATION_APPROVAL_FRACTION]:
        applied = full.assign(approve=policy_approvals(full, f)).merge(
            monitoring, on="customer_id", validate="one_to_one")
        for g, frame in applied.groupby("group", sort=True):
            full_audits.append({
                "model": "Random forest (full refit)", "approval_fraction": f,
                "group": g, "n": len(frame), "approved": int(frame.approve.sum()),
                "approval_rate": float(frame.approve.mean())})
    pd.DataFrame(full_audits).to_csv(
        OUTPUT_DIR / "stage3_champion_full_refit_demographic_parity.csv",
        index=False, float_format="%.12g")
    joblib.dump({"model": champion, "features": FEATURES,
                 "training_population": "8,141 historically taken-up loans"},
                OUTPUT_DIR / "stage3_champion_random_forest.joblib")
    print(f"Submission saved: {dest}", flush=True)
    print(f"Full-fit pending mean PD: {pending_pd.mean():.4%}", flush=True)
    print("Fairness outcome metrics are limited to held-out, historically taken-up loans.")
    return dest


def main() -> None:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Missing unzipped data release: {DATA_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Full reproduction from original data: no external feature CSV required.
    for name, fn in [
        ("Payroll", payroll), ("Income regularity", income_regularity),
        ("Income volatility", income_volatility),
        ("Spending / income", spending_to_income),
        ("Cash buffer", cash_buffer),
        ("Employment tenure", employment_tenure),
        ("Income discrepancy", income_discrepancy),
        ("Housing burden", housing_burden),
        ("Modeling table", model_table),
        ("Challenger + fairness + pending queue", run_stage3),
    ]:
        print(f"\n--- {name} ---", flush=True)
        fn()
    print("\nComplete. Only submit the Python program, write-up PDF, and Stage 3 CSV.")

if __name__ == "__main__":
    main()

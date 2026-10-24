#!/usr/bin/env python3
"""FIN 6155 — WageWave (Group 07) | Stage 2 baseline PD model.

SINGLE-FILE Python submission equivalent to the instructor's Stage 2 .Rmd.

Runs the full build from the original, unzipped synthetic data release:
    python WageWave_group_07_stage2.py

Input layout (the only external input is data_release/):
    project/WageWave_group_07_stage2.py
    project/data_release/applications.csv
    project/data_release/outcomes_train.csv
    project/data_release/holdout_ids.csv
    project/data_release/plaid_synth_shard_*.ndjson.gz   (50 files)

Produces outputs/WageWave_group_07_stage2.csv, and supporting feature,
validation, coefficient, and pre-registration-scorecard artifacts in outputs/.
Requires Python 3.10+ and numpy, pandas, scipy, scikit-learn, joblib.
Optional orjson makes JSON shard reading faster.

Implementation: one self-contained script performs feature engineering,
validation, full-sample refitting, and submission scoring. It outputs separate
validation and final-fit coefficient tables/scorecards so estimates are not conflated. The data and model inputs exclude 'group'
and the incumbent's approved/offered_apr/took_up decision fields.
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
SUBMISSION_FILENAME = "WageWave_group_07_stage2.csv"


# ===========================================================================
# PAYROLL
# ===========================================================================


# Six full calendar months before the release's 2026-07-31 data cutoff.
# This fixed window is earlier than every recorded application date.
WINDOW_START = date(2026, 2, 1)
WINDOW_END = date(2026, 7, 31)


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
        # Other INCOME_* categories (e.g., gig work) are not wage deposits;
        # TRANSFER_IN_* deposits are excluded too.
        if (pfc.get("primary") == "INCOME"
                and pfc.get("detailed") == "INCOME_WAGES"
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
            and pfc.get("detailed") == "INCOME_WAGES"
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
            and category.get("detailed") == "INCOME_WAGES"
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

        if primary == "INCOME" and detailed == "INCOME_WAGES" and amount < 0:
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
# LOGISTIC TRAIN/VALIDATE
# ===========================================================================


# Two paths the grader may change; keep all other paths relative to these.

RANDOM_SEED = 6155
VALIDATION_FRACTION = 0.20
TARGET = "default_12m"
FEATURES = [
    "income_regularity",
    "income_volatility",
    "spending_to_income_ratio",
    "cash_buffer",
    "employment_tenure_months",
    "income_discrepancy",
    "monthly_payroll_income",
    "housing_payment_burden",
]
EXPECTED_SIGNS = {
    "income_regularity": "-",
    "income_volatility": "+",
    "spending_to_income_ratio": "+",
    "cash_buffer": "-",
    "employment_tenure_months": "-",
    "income_discrepancy": "+",
    "monthly_payroll_income": "-",
    "housing_payment_burden": "+",
}
ECONOMIC_RATIONALES = {
    "income_regularity": "Regular paycheck timing makes funds available when repayment is due.",
    "income_volatility": "Fluctuating wages make repayment less predictable.",
    "spending_to_income_ratio": "Spending relative to wages reduces cash available for repayment.",
    "cash_buffer": "Liquid cash reserves absorb unexpected financial shocks.",
    "employment_tenure_months": "Longer tenure may indicate more stable future wages.",
    "income_discrepancy": "Stated income above observed wages may overstate repayment capacity.",
    "monthly_payroll_income": "Higher recurring wage income increases repayment capacity.",
    "housing_payment_burden": "Larger housing obligations reduce cash available for repayment.",
}


def make_pipeline() -> Pipeline:
    """All learned preprocessing happens *after* the train/validation split."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=5000,
            class_weight=None, random_state=RANDOM_SEED,
        )),
    ])


def load_labeled() -> pd.DataFrame:
    path = OUTPUT_DIR / "labeled_model_data.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run this consolidated script from its entry point."
        )
    df = pd.read_csv(path, dtype={"customer_id": "string"})
    required = set(FEATURES) | {"customer_id", TARGET}
    absent = required - set(df.columns)
    if absent:
        raise ValueError(f"Missing columns: {sorted(absent)}")
    # Fail rather than accidentally incorporating a protected or incumbent field.
    forbidden = {"group", "approved", "offered_apr", "took_up"}
    if forbidden & set(df.columns):
        raise ValueError(f"Forbidden columns in modeling input: {forbidden & set(df.columns)}")
    if df["customer_id"].isna().any() or df["customer_id"].duplicated().any():
        raise ValueError("Missing or duplicated labeled customer IDs")
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Labels must all be 0/1")
    X = df[FEATURES]
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in FEATURES):
        raise ValueError("All features must be numeric")
    if not np.isfinite(X.to_numpy(dtype=float)[~X.isna().to_numpy()]).all():
        raise ValueError("Inf/-Inf in feature table")
    if (X.notna().sum(axis=0) == 0).any():
        raise ValueError("At least one predictor is entirely missing")
    return df


def make_calibration_table(ids: pd.Series, y: pd.Series, pd_hat: np.ndarray):
    cal = pd.DataFrame({
        "customer_id": ids.to_numpy(),
        "default_12m": y.to_numpy(dtype=int),
        "pd": pd_hat,
    }).sort_values(["pd", "customer_id"], kind="mergesort").reset_index(drop=True)
    n = len(cal)
    cal["risk_decile"] = (np.arange(n) * 10 // n) + 1
    table = (cal.groupby("risk_decile", as_index=False)
             .agg(n=("pd", "size"),
                  defaults=("default_12m", "sum"),
                  mean_predicted_pd=("pd", "mean"),
                  observed_default_rate=("default_12m", "mean"),
                  min_pd=("pd", "min"),
                  max_pd=("pd", "max")))
    if len(table) != 10 or table["n"].sum() != n:
        raise AssertionError("Calibration table does not contain ten complete deciles")
    return table, cal


def coefficient_tables(pipe: Pipeline) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputer = pipe.named_steps["imputer"]
    model = pipe.named_steps["logistic"]
    scaler = pipe.named_steps["scaler"]
    names = list(imputer.get_feature_names_out(FEATURES))
    if len(names) != len(model.coef_[0]):
        raise AssertionError("Coefficient/column names do not align")

    coef = pd.DataFrame({
        "term": names,
        "standardized_coefficient": model.coef_[0],
        "odds_ratio_per_1sd": np.exp(model.coef_[0]),
        "training_column_sd": scaler.scale_,
    })
    coef["kind"] = np.where(coef["term"].str.startswith("missingindicator_"),
                            "missingness indicator", "pre-registered feature")
    coef["expected_sign"] = coef["term"].map(EXPECTED_SIGNS)
    coef["observed_sign"] = np.where(coef["standardized_coefficient"] > 0, "+",
                                     np.where(coef["standardized_coefficient"] < 0, "-", "0"))
    coef["economic_rationale"] = coef["term"].map(ECONOMIC_RATIONALES)
    coef["estimated_economic_interpretation"] = np.where(
        coef["kind"] == "pre-registered feature",
        np.where(coef["standardized_coefficient"] >= 0,
                 "Higher values are associated with higher modeled odds of default, holding the other features fixed.",
                 "Higher values are associated with lower modeled odds of default, holding the other features fixed."),
        "This coefficient describes whether the feature is missing; it is not a change in the feature's value."
    )
    # For standardized missingness indicators, a one-SD OR is not the
    # 0-to-1 contrast. This extra field makes the interpretation possible.
    coef["indicator_odds_ratio_missing_vs_present"] = np.where(
        coef["kind"] == "missingness indicator",
        np.exp(coef["standardized_coefficient"] / coef["training_column_sd"]),
        np.nan,
    )

    prereg = coef.loc[coef["term"].isin(FEATURES),
                     ["term", "standardized_coefficient", "expected_sign",
                      "observed_sign", "economic_rationale"]].copy()
    prereg = prereg.set_index("term").loc[FEATURES].reset_index().rename(columns={"term": "feature"})
    prereg.insert(1, "built", "Yes")
    prereg["hypothesis"] = np.where(prereg["observed_sign"] == prereg["expected_sign"],
                                    "Confirmed (sign)", "Contradicted (sign)")
    contradiction_notes = {
        "income_volatility": (
            "Income volatility has a small negative conditional coefficient rather than the "
            "hypothesized positive sign; shared information with other cash-flow measures "
            "and selection into booked loans may contribute."
        ),
        "employment_tenure_months": (
            "In the 80%-training fit, employment tenure has a small positive coefficient "
            "rather than the hypothesized negative sign; correlated borrower factors "
            "and labeled-loan selection may contribute."
        ),
        "spending_to_income_ratio": (
            "In the full-sample fit, spending-to-income has a very small negative "
            "conditional coefficient rather than the hypothesized positive sign; "
            "multicollinearity or sampling variation may contribute."
        ),
    }
    prereg["contradiction_note"] = prereg.apply(
        lambda row: contradiction_notes.get(row["feature"],
                                            "The estimated conditional sign differed from the hypothesis; "
                                            "the reason is not identified from this model.")
        if row["hypothesis"] == "Contradicted (sign)" else "", axis=1
    )
    return coef, prereg


def preprocessing_metadata(pipe: Pipeline) -> dict:
    """Report learned indicator terms and why they were added; do not confuse
    them with the eight pre-registered *substantive* credit-risk features.
    """
    imputer = pipe.named_steps["imputer"]
    indicator_terms = [
        name for name in imputer.get_feature_names_out(FEATURES)
        if name.startswith("missingindicator_")
    ]
    return {
        "pre_registered_substantive_features": list(FEATURES),
        "added_substantive_features_after_preregistration": [],
        "added_preprocessing_terms": indicator_terms,
        "added_preprocessing_term_count": len(indicator_terms),
        "added_preprocessing_term_rationale": (
            "An indicator records when a pre-registered variable is unavailable "
            "(e.g., no observed wage payroll or insufficient payroll dates). "
            "Its coefficient describes missingness, not a change in that feature's value."
        ),
        "indicator_redundancy_note": (
            "Four payroll-dependent ratio/volatility indicators are identical "
            "on this cohort; L2-regularized logistic regression retains them "
            "without independently interpretable coefficients."
        ),
    }


def train_validate() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_labeled()
    if set(df.columns) != set(["customer_id", *FEATURES, TARGET]):
        raise ValueError("Model input must contain only customer_id, eight features, and label")
    y = df[TARGET].astype(int)
    all_indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        all_indices, test_size=VALIDATION_FRACTION,
        stratify=y, random_state=RANDOM_SEED,
    )
    if set(train_idx) & set(val_idx):
        raise AssertionError("Train and validation overlap")
    X_train = df.iloc[train_idx][FEATURES]
    X_val = df.iloc[val_idx][FEATURES]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)
    predicted_pd = pipeline.predict_proba(X_val)[:, 1]
    if not np.isfinite(predicted_pd).all() or not ((0 < predicted_pd) & (predicted_pd < 1)).all():
        raise ValueError("Invalid model predictions")

    auc = roc_auc_score(y_val, predicted_pd)
    fpr, tpr, thresholds = roc_curve(y_val, predicted_pd)
    ks = float(np.max(tpr - fpr))
    brier = brier_score_loss(y_val, predicted_pd)

    calibration, val_predictions = make_calibration_table(
        df.iloc[val_idx]["customer_id"], y_val, predicted_pd
    )
    coefficient_table, scorecard = coefficient_tables(pipeline)
    metrics = {
        "method": "Stratified 80/20, fixed-seed; median imputation + missingness "
                  "indicators + standardization, all fitted on training only; "
                  "L2 logistic regression, C=1, no class reweighting",
        "random_seed": RANDOM_SEED,
        "n_labeled": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "defaults_train": int(y_train.sum()),
        "defaults_validation": int(y_val.sum()),
        "observed_default_rate_validation": float(y_val.mean()),
        "mean_predicted_pd_validation": float(predicted_pd.mean()),
        "auc_validation": float(auc),
        "ks_validation": float(ks),
        "brier_validation": float(brier),
        "feature_count": len(FEATURES),
        **preprocessing_metadata(pipeline),
        "comment": "Metrics are for the selected, labeled taken-up-loan validation sample, "
                   "not the full pending applicant population.",
    }
    (OUTPUT_DIR / "stage2_validation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    val_predictions = val_predictions[["customer_id", "default_12m", "pd", "risk_decile"]]
    val_predictions.to_csv(OUTPUT_DIR / "stage2_validation_predictions.csv", index=False, float_format="%.12g")
    calibration.to_csv(OUTPUT_DIR / "stage2_calibration_deciles.csv", index=False, float_format="%.12g")
    coefficient_table.to_csv(OUTPUT_DIR / "stage2_coefficients.csv", index=False, float_format="%.12g")
    scorecard.to_csv(OUTPUT_DIR / "stage2_preregistration_scorecard.csv", index=False, float_format="%.12g")
    joblib.dump({"model": pipeline, "features": FEATURES, "train_indices": train_idx,
                 "validation_indices": val_idx, "random_seed": RANDOM_SEED},
                OUTPUT_DIR / "stage2_validation_model.joblib")

    print("Stage 2 — VALIDATION results")
    print(f"Labeled: {len(df):,}; training: {len(train_idx):,} "
          f"({int(y_train.sum())} defaults); validation: {len(val_idx):,} "
          f"({int(y_val.sum())} defaults)")
    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation KS: {ks:.4f}")
    print(f"Validation Brier: {brier:.5f}")
    print(f"Validation observed default rate: {y_val.mean():.2%}")
    print(f"Validation mean predicted PD: {predicted_pd.mean():.2%}")
    print("\n10-decile calibration (1 = lowest predicted risk):")
    print(calibration[["risk_decile", "n", "defaults", "mean_predicted_pd",
                       "observed_default_rate"]].to_string(
        index=False, formatters={
            "mean_predicted_pd": lambda x: f"{x:.2%}",
            "observed_default_rate": lambda x: f"{x:.2%}",
        }
    ))
    print("\nPre-registration scorecard:")
    print(scorecard[["feature", "expected_sign", "observed_sign", "hypothesis"]].to_string(index=False))
    if auc >= 0.89:
        print("\nWARNING: AUC approaches the stated data-generating ceiling; investigate leakage.")
    print("\nSaved validation metrics, predictions, calibration, coefficient table, scorecard, and model in outputs/.")
    print("Pending-queue scoring follows validation.")


# ===========================================================================
# FULL TRAIN + QUEUE SCORING
# ===========================================================================


def read_model_table(path: Path, *, labels: bool) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run this consolidated script from its entry point.")
    df = pd.read_csv(path, dtype={"customer_id": "string"})
    required = {"customer_id", *FEATURES}
    if labels:
        required.add(TARGET)
    if set(df.columns) != required:
        raise ValueError(
            f"{path.name}: expected only {sorted(required)}; got {sorted(df.columns)}"
        )
    if df.customer_id.isna().any() or df.customer_id.duplicated().any():
        raise ValueError(f"{path.name}: missing or duplicate customer IDs")
    for feature in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise ValueError(f"{path.name}: {feature} is not numeric")
    array = df[FEATURES].to_numpy(dtype=float)
    if np.isinf(array).any():
        raise ValueError(f"{path.name}: inf/-inf feature values")
    if labels and (df[TARGET].isna().any() or not df[TARGET].isin((0, 1)).all()):
        raise ValueError("Labeled data must have observed 0/1 default_12m")
    return df


def holdout_validation() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labeled = read_model_table(OUTPUT_DIR / "labeled_model_data.csv", labels=True)
    pending = read_model_table(OUTPUT_DIR / "pending_model_data.csv", labels=False)

    holdout_path = DATA_DIR / "holdout_ids.csv"
    if not holdout_path.is_file():
        raise FileNotFoundError(holdout_path)
    holdout = pd.read_csv(holdout_path, dtype={"customer_id": "string"})
    if "customer_id" not in holdout or holdout.customer_id.isna().any():
        raise ValueError("holdout_ids.csv has missing customer_id values")
    if holdout.customer_id.duplicated().any():
        raise ValueError("holdout_ids.csv contains duplicate customer IDs")
    if len(holdout) != 5000:
        raise ValueError(f"Expected 5,000 pending IDs, found {len(holdout)}")
    if len(labeled) != 8141:
        raise ValueError(f"Expected 8,141 Group 07 labeled loans, found {len(labeled)}")

    labeled_ids = set(labeled.customer_id)
    pending_ids = set(pending.customer_id)
    holdout_ids = set(holdout.customer_id)
    if labeled_ids & pending_ids:
        raise ValueError("Data leakage: labeled and pending customers overlap")
    if pending_ids != holdout_ids:
        raise ValueError(
            "Pending features and holdout IDs disagree: "
            f"missing {len(holdout_ids - pending_ids)}, extra {len(pending_ids - holdout_ids)}"
        )
    if labeled[FEATURES].isna().all(axis=0).any():
        raise ValueError("One or more training predictors are entirely missing")

    # Exactly the same preprocessing and model as the validation fit,
    # but refit on every labeled loan. The holdout never participates in fit().
    pipeline = make_pipeline()
    pipeline.fit(labeled[FEATURES], labeled[TARGET].astype(int))
    probabilities = pipeline.predict_proba(pending[FEATURES])[:, 1]
    if not np.isfinite(probabilities).all() or not np.all(
        (probabilities > 0) & (probabilities < 1)
    ):
        raise ValueError("All predictions must be finite probabilities strictly in (0, 1)")

    # Keep the exact ordering of holdout_ids.csv; do not create duplicate rows.
    scores = pd.DataFrame({"customer_id": pending.customer_id, "pd": probabilities})
    submission = holdout[["customer_id"]].merge(
        scores, on="customer_id", how="left", validate="one_to_one", sort=False
    )
    if (list(submission.columns) != ["customer_id", "pd"]
            or len(submission) != 5000
            or submission.customer_id.tolist() != holdout.customer_id.tolist()
            or submission.pd.isna().any()):
        raise AssertionError("Submission contract verification failed")

    csv_path = OUTPUT_DIR / SUBMISSION_FILENAME
    submission.to_csv(csv_path, index=False, float_format="%.15g", lineterminator="\n")

    # Read the actual disk output back: verify serialization has not produced
    # endpoints 0/1, dropped IDs, or changed row order.
    check = pd.read_csv(csv_path, dtype={"customer_id": "string"})
    assert list(check.columns) == ["customer_id", "pd"]
    assert len(check) == len(holdout) == 5000
    assert check.customer_id.tolist() == holdout.customer_id.tolist()
    assert check.customer_id.is_unique and check.pd.notna().all()
    assert np.isfinite(check.pd).all() and ((check.pd > 0) & (check.pd < 1)).all()

    # Report the final-fit coefficients separately: the validation-model signs
    # need not match the full-sample fit used for pending-queue scoring.
    final_coef, final_scorecard = coefficient_tables(pipeline)
    final_coef.to_csv(OUTPUT_DIR / "stage2_final_coefficients.csv", index=False,
                      float_format="%.12g")
    final_scorecard.to_csv(OUTPUT_DIR / "stage2_final_preregistration_scorecard.csv",
                           index=False, float_format="%.12g")
    validation_scorecard = pd.read_csv(
        OUTPUT_DIR / "stage2_preregistration_scorecard.csv"
    )
    comparison = validation_scorecard[[
        "feature", "expected_sign", "observed_sign", "standardized_coefficient"
    ]].rename(columns={
        "observed_sign": "validation_sign",
        "standardized_coefficient": "validation_standardized_coefficient",
    }).merge(final_scorecard[[
        "feature", "observed_sign", "standardized_coefficient"
    ]].rename(columns={
        "observed_sign": "final_sign",
        "standardized_coefficient": "final_standardized_coefficient",
    }), on="feature", validate="one_to_one")
    comparison["sign_changed_after_refit"] = (
        comparison["validation_sign"] != comparison["final_sign"]
    )
    comparison.to_csv(OUTPUT_DIR / "stage2_coefficient_sign_comparison.csv",
                      index=False, float_format="%.12g")

    # Additional artifacts are local diagnostics; only the CSV is required.
    joblib.dump({"model": pipeline, "features": FEATURES,
                 "trained_on": "all labeled historical loans (Group 07)",
                 "n_labeled": len(labeled)},
                OUTPUT_DIR / "stage2_full_model.joblib")
    summary = {
        "n_labeled": int(len(labeled)),
        "defaults_labeled": int(labeled[TARGET].sum()),
        "n_pending": int(len(submission)),
        "pending_pd_mean": float(check.pd.mean()),
        "pending_pd_median": float(check.pd.median()),
        "pending_pd_min": float(check.pd.min()),
        "pending_pd_max": float(check.pd.max()),
        **preprocessing_metadata(pipeline),
        "validation_to_final_sign_changes": comparison.loc[
            comparison["sign_changed_after_refit"], "feature"
        ].tolist(),
        "note": "Pending predictions are uncalibrated to any known full-applicant default rate; "
                "labels exist only for historically taken-up loans.",
    }
    (OUTPUT_DIR / "stage2_pending_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("Stage 2 — FINAL PENDING-QUEUE PREDICTIONS")
    print(f"Trained on: {len(labeled):,} labeled historical loans "
          f"({int(labeled[TARGET].sum())} defaults)")
    print(f"Scored: {len(check):,} pending applicants")
    print(f"Mean PD: {check.pd.mean():.2%}; Median PD: {check.pd.median():.2%}")
    print(f"PD range: {check.pd.min():.4%} – {check.pd.max():.4%}")
    print("All customer-ID, probability, column, and row-count checks: PASSED")
    print("Sign changes after full-sample refit:",
          ", ".join(summary["validation_to_final_sign_changes"]) or "none")
    print("Final coefficient/scorecard and sign comparison saved in outputs/.")
    print(f"Submission CSV: {csv_path}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================


def main() -> None:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Missing data directory: {DATA_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for task_name, runner in [
        ("Payroll", payroll), ("Income Regularity", income_regularity), ("Income Volatility", income_volatility),
        ("Spending to Income", spending_to_income), ("Cash Buffer", cash_buffer), ("Employment Tenure", employment_tenure),
        ("Income Discrepancy", income_discrepancy), ("Housing Payment Burden", housing_burden), ("Model Table", model_table),
        ("Training and Validation", train_validate), ("Holdout Validation", holdout_validation),
    ]:
        print(f"\n{'=' * 72}\nRunning {task_name}", flush=True)
        runner()
    path = OUTPUT_DIR / "WageWave_group_07_stage2.csv"
    if not path.is_file():
        raise RuntimeError("Final submission CSV was not created")
    print(f"\nCOMPLETE: {path}", flush=True)


if __name__ == "__main__":
    main()

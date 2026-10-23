
'''
Used AI to change formatting from R to python
'''

# FIN 6155 · Digital Credit Build · Stage 2 Starter
# Synthetic Data Loader + Baseline Model
# Author: (your team)

"""
You receive a historical loan book from First Ute Bank:
25,000 applicants (20,000 historical + 5,000 pending).

Data:
- 50 Plaid-shaped JSON shards, 500 customers each
- applications.csv
- outcomes_train.csv
- holdout_ids.csv

IMPORTANT:
Performance (default_12m) is observed ONLY for taken-up loans.
The PD model is trained on a doubly-selected sample and will be
miscalibrated on the full applicant population.

offered_apr was never randomized, so the price-default correlation
reflects selection, NOT the causal effect of price.

The transaction schema matches Stage 1.
"""

# ============================================================
# SETUP
# ============================================================

# ===== SET YOUR WORKING DIRECTORY FIRST =====
# Leave "." if running from the project folder.
# Otherwise, specify the folder's full path.

import os
import json
import gzip
from pathlib import Path

import pandas as pd
import numpy as np

# Modeling stack (Stage 2 baseline uses logistic regression)
# pip install pandas numpy statsmodels scikit-learn python-dotenv

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# WORKING_DIR = "."
# os.chdir(WORKING_DIR)

# Load PLAID_CLIENT_ID / PLAID_SECRET if .env exists.
# Python's dotenv is the closest equivalent to R's .Renviron.
from dotenv import load_dotenv

if Path(".env").exists():
    load_dotenv(".env")

DATA_DIR = "data_release"  # Adjust to your release location


# ============================================================
# 1. LOAD ONE SHARD (WORKED EXAMPLE — THEN SCALE UP)
# ============================================================

# Each shard is gzipped NDJSON:
# one customer bundle per line.

def read_shard(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        lines = f.readlines()

    return [json.loads(line) for line in lines]


shard0 = read_shard(
    os.path.join(DATA_DIR, "plaid_synth_shard_00.ndjson.gz")
)

# print(len(shard0))  # 500 customers

b = shard0[0]       # One customer bundle

# Equivalent to str(b, max.level = 1)
# print(b.keys())

# customer_id, item, identity, accounts,
# transactions, liabilities

# Equivalent to head(as_tibble(b$transactions))
# print("IDENTITY", pd.DataFrame(b["liabilities"]).head())


'''
Customer Data
dict_keys(['customer_id', 'item', 'identity', 'accounts',
           'transactions', 'liabilities'])

Transactions
['transaction_id', 'account_id', 'amount', 'iso_currency_code',
 'date', 'name', 'merchant_name', 'payment_channel',
 'pending', 'personal_finance_category']

    *payment_channel
        ['in store', 'other', 'online']

    *personal_finance_category (original nested dictionary)
        ['primary', 'detailed']

    *personal_finance_category (after extracting into tx)
        ['pfc_primary', 'pfc_detailed']

Identity
['names', 'emails']

Accounts
['account_id', 'balances', 'mask', 'name',
 'official_name', 'subtype', 'type']

    *mask = Last few digits of the account number

Liabilities
['credit', 'student', 'mortgage']

Item
['item_id', 'institution_id', 'institution_name']


NEW DATAFRAME: customer_features(b)

['customer_id', 'inc_mean', 'nsf', 'inc_cv']

    *customer_id = Unique customer identifier
    *inc_mean = Average monthly income
    *nsf = Number of overdraft fee transactions
    *inc_cv = Income volatility (if added)

    One row per customer.
'''


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# Remember the Plaid sign convention from Stage 1.
#
# Negative amount = money entering the account
# Positive amount = money leaving the account

#collectivvely takes the 
def customer_features(b):

    tx = pd.DataFrame(b["transactions"]).copy()

    tx["date"] = pd.to_datetime(tx["date"])

    # Extract nested personal_finance_category fields
    tx["pfc_primary"] = tx[
        "personal_finance_category"
    ].apply(
        lambda x: x.get("primary") if isinstance(x, dict) else None
    )

    tx["pfc_detailed"] = tx[
        "personal_finance_category"
    ].apply(
        lambda x: x.get("detailed") if isinstance(x, dict) else None
    )

    # --------------------------------------------------------
    # Example feature 1:
    # Mean monthly income, identified BY CATEGORY
    # --------------------------------------------------------

    #inc_m consists of |month   amount|
    inc_m = tx[
        (tx["pfc_primary"] == "INCOME") &
        (tx["amount"] < 0)
    ].copy()

    inc_m["month"] = inc_m["date"].dt.strftime("%Y-%m")

    inc_m = (
        inc_m.groupby("month", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "inc"})
    )

    inc_m["inc"] = -inc_m["inc"]

    inc_mean = inc_m["inc"].mean()


    # --------------------------------------------------------
    # Example feature 2:
    # Overdraft (NSF) events
    # --------------------------------------------------------

    nsf = (
        tx["pfc_detailed"] == "BANK_FEES_OVERDRAFT_FEES"
    ).sum()

    # --------------------------------------------------------
    # Return one row per customer
    # --------------------------------------------------------

    #returns customer we referring to, their mean income, and their average income on all qualifying months.
    return pd.DataFrame({
        "customer_id": [b["customer_id"]],
        "inc_mean": [inc_mean],
        "nsf": [nsf]
    })

#good for variability of how much their income is likely to change given the months that are given in the data set.
#For example, can mutiply this output to total monthly income to determine variability
    def income_volatility(inc_m):
        #takes all rows from inc_m["inc"] and puts it in a mean
        inc_mean = inc_m["inc"].mean()
        #takes the standard deviation for all  months for customer
        inc_std = inc_m["inc"].std(ddof=0)
        #In case there is no mean or variability
        if pd.isna(inc_mean) or inc_mean == 0:
            return None
        #calculates the coeffiecient of variation
        return inc_std / inc_mean

    # TODO — ideas you should pursue:
    def payroll_regularity():
        pass
    #   and payroll regularity
    #
    # * Balance path:
    #   You observe the END balance in b["accounts"]
    #   and every dated flow.
    #   Reconstruct daily balance BACKWARD from end balance.
    #   Median balance / income ("buffer") is a strong
    #   cash-flow feature.
    #
    # * Spending composition
    #   Debt payments (LOAN_PAYMENTS) / income
    #
    # * Credit card utilization from credit-type account:
    #   balance / limit
    #
    # * Liabilities:
    #   minimum_payment_amount
    #   is_overdue
    #   (watch missing cases!)
    #
    # * Data quality:
    #   Are transactions duplicated?
    #   Do all income-looking inflows belong in income?
    #   TRANSFER_IN is your own money moving.
    #
    # * Application file:
    #   requested_amount vs income
    #   STATED income vs observed bank income
    #   (W2: "Observing Unobservables")


print("RUNNING CUSTOMER_FEATURES",customer_features(b))


# ============================================================
# 3. SCALE TO ALL SHARDS + JOIN LABELS
# ============================================================

# TODO: Loop through all shards 00..49.
# This takes a few minutes; cache the result!

# files = sorted(
#     Path(DATA_DIR).glob("plaid_synth_shard_*.ndjson.gz")
# )
#
# X = pd.concat(
#     [
#         customer_features(b)
#         for f in files
#         for b in read_shard(f)
#     ],
#     ignore_index=True
# )
#
# X.to_pickle("features_cache.pkl")


# Load application and training outcome files

apps = pd.read_csv(
    os.path.join(DATA_DIR, "applications.csv")
)

ytrain = pd.read_csv(
    os.path.join(DATA_DIR, "outcomes_train.csv")
)


# Join engineered features, applications, and labels

# df = (
#     X.merge(apps, on="customer_id", how="left")
#      .merge(ytrain, on="customer_id", how="left")
# )


"""
IMPORTANT:

Labels in outcomes_train.csv exist ONLY for taken-up loans
(approved AND accepted).

A PD model fit on these loans is miscalibrated on all applicants.

The project asks for THREE models:

1. Underwriting model:
   Approve / deny, on all applicants

2. Pricing model:
   APR for those you approve

3. PD model:
   Probability of default, on taken-up loans

Apply them to the new-consumer holdout.

IMPORTANT — FAIR LENDING:

"group" in applications.csv is fair-lending MONITORING data.

It must NOT be used as a model input at any stage.

Use it in Stage 3 to audit your model.

Document this exclusion in your write-up.
"""


# ============================================================
# 4. BASELINE LOGISTIC REGRESSION
# STAGE 2 DELIVERABLE
# ============================================================

# Split LABELED taken-up customers into train/validation.
#
# The rows of YOUR outcomes_train.csv contain roughly
# 7,200–9,300 labeled loans, depending on cohort.
#
# The pending ids in holdout_ids.csv are scored
# by the instructor.

# Equivalent to R: set.seed(6155)

np.random.seed(6155)


# TODO: Split labeled customers into train/validation

# train, validation = train_test_split(
#     labeled_df,
#     test_size=0.2,
#     random_state=6155
# )


# TODO: Fit logistic regression
#
# R equivalent:
# glm(default_12m ~ ..., family = binomial(), data = train)
#
# Python equivalent:
#
# import statsmodels.formula.api as smf
#
# model = smf.glm(
#     formula="default_12m ~ inc_mean + nsf",
#     data=train,
#     family=sm.families.Binomial()
# ).fit()
#
# print(model.summary())


# TODO: Validation AUC + KS
#
# Equivalent to yardstick::roc_auc
# and KS from cumulative separation
#
# validation["pd"] = model.predict(validation)
#
# auc = roc_auc_score(
#     validation["default_12m"],
#     validation["pd"]
# )
#
# fpr, tpr, thresholds = roc_curve(
#     validation["default_12m"],
#     validation["pd"]
# )
#
# ks = np.max(tpr - fpr)
#
# print("Validation AUC:", auc)
# print("Validation KS:", ks)


# TODO: Calibration table
#
# 10 deciles of predicted PD vs observed default rate
#
# validation["pd_decile"] = pd.qcut(
#     validation["pd"],
#     q=10,
#     labels=False,
#     duplicates="drop"
# )
#
# calibration = (
#     validation.groupby("pd_decile")
#     .agg(
#         predicted_pd=("pd", "mean"),
#         observed_default=("default_12m", "mean"),
#         count=("default_12m", "size")
#     )
#     .reset_index()
# )
#
# print(calibration)


# TODO: Coefficient table
#
# Include a one-sentence ECONOMIC interpretation
# per feature.
#
# Why SHOULD this feature predict default?
#
# "It helps AUC" is NOT an economic explanation.


# ============================================================
# 5. HOLDOUT SUBMISSION
# REQUIRED WITH EVERY MODEL STAGE
# ============================================================

# TODO: Predict PD for every id in holdout_ids.csv.
#
# Output file:
# TEAMNAME_group_NN_stage2.csv
#
# NN = your data-group number


# holdout_ids = pd.read_csv(
#     os.path.join(DATA_DIR, "holdout_ids.csv")
# )
#
# TODO: Join holdout IDs with engineered features
# and apply the fitted PD model.
#
# holdout_predictions = model.predict(holdout_features)
#
# submission = pd.DataFrame({
#     "customer_id": holdout_features["customer_id"],
#     "pd": holdout_predictions
# })
#
# submission.to_csv(
#     "TEAMNAME_group_NN_stage2.csv",
#     index=False
# )


"""
SUBMISSION CONSTRAINTS:

- Every pending ID must appear exactly once.
- PD must be strictly between 0 and 1.

The lowest-PD 70% of applicants would be approved
under the default policy.

Cutoff and pricing analysis are yours to design
in Stages 3 and 5.
"""

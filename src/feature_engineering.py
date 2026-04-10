import numpy as np

def create_features(df):
    # Claim flag
    df["claim_flag"] = (df["claim_amount"] > 0).astype(int)

    # Claim per year
    df["claim_per_year"] = df["claim_amount"] / (df["vintage"] + 1)

    # Policy mapping
    df["num_policies"] = df["num_policies"].replace({
        "More than 1": 2,
        "1": 1
    })

    return df
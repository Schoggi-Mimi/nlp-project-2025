import pandas as pd


def annotate_entities(path: str) -> pd.DataFrame:
    """
    Annotate entities in the given DataFrame by prompting the user for validation.

    Args:
        df (pd.DataFrame): DataFrame containing 'entity' and 'label' columns.
    Returns:
        pd.DataFrame: DataFrame with an additional 'correct' column indicating user validation.
    """
    df = pd.read_csv(path)

    df["correct"] = None
    for i, row in df.iterrows():
        print("\n--------------------------------------------")
        print(f"Entity: {row['entity']}")
        print(f"Label : {row['label']}")
        ans = input("Correct? (y/n): ")

        df.at[i, "correct"] = 1 if ans.lower() == "y" else 0

    df.to_csv(path, index=False)
    return df
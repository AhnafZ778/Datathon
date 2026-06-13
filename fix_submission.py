import pandas as pd

def main():
    print("Reading test probabilities...")
    df = pd.read_csv("./predictions/test_probabilities.csv")
    
    # Rename column
    df = df.rename(columns={"churn_probability": "CHURN_PROB"})
    
    # Reorder columns just to be safe
    df = df[["ACCOUNT_ID", "CHURN_PROB"]]
    
    print("Saving to predictions.csv...")
    df.to_csv("./predictions.csv", index=False)
    print("Done! Formatted file saved successfully.")

if __name__ == "__main__":
    main()

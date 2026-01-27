import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import resample
import os


def perform_mcnemar(df_a, df_b, model_a_name, model_b_name):
    """Calculates McNemar's test p-value between two models"""
    # Ensure they are comparing the same samples
    merged = pd.merge(df_a, df_b, on="image_path", suffixes=("_a", "_b"))

    # REQUIRED FIX #4: Validate label consistency
    assert (merged["true_label_a"] == merged["true_label_b"]).all(), \
        "Error: true labels do not match between models. Data integrity check failed."
    print(f"DEBUG: Comparing {len(merged)} matching samples for {model_b_name}")
    y_true = merged["true_label_a"]
    pred_a = merged["prediction_a"]
    pred_b = merged["prediction_b"]

    # Correctness arrays
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    # Contingency Table
    a = sum((correct_a == True) & (correct_b == True))
    b = sum((correct_a == True) & (correct_b == False))
    c = sum((correct_a == False) & (correct_b == True))
    d = sum((correct_a == False) & (correct_b == False))

    table = [[a, b],
            [c, d]]
    exact = (b + c) < 25
    result = mcnemar(table, exact=exact, correction=not exact)

    return result.pvalue, b, c


def bootstrap_metrics(df, n_iterations=1000):
    """Calculates 95% Confidence Intervals for F1 and Accuracy"""
    f1_stats = []
    acc_stats = []

    y_true = df["true_label"].values
    y_pred = df["prediction"].values

    for _ in range(n_iterations):
        # Resample with replacement
        indices = np.random.randint(0, len(y_true), len(y_true))
        f1_stats.append(f1_score(y_true[indices], y_pred[indices], average="macro"))
        acc_stats.append(accuracy_score(y_true[indices], y_pred[indices]))

    return {
        "f1_ci": (np.percentile(f1_stats, 2.5), np.percentile(f1_stats, 97.5)),
        "acc_ci": (np.percentile(acc_stats, 2.5), np.percentile(acc_stats, 97.5)),
    }


def main():
    # 1. Load your detailed CSV files (Replace with your actual filenames)
    # Ensure these files exist in your working directory
    try:
        df_cafnet = pd.read_csv(
            "detailed_results_C_AFNet_MITBIH_20260119_212835.csv"
        )
        df_rafnet = pd.read_csv(
            "detailed_results_R-AFNet_MIT_BIH_20260119_204725.csv"
        )
        df_resnet = pd.read_csv("detailed_results_resnet18_MITBIH_20260119_231717.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure CSV files are named correctly.")
        return

    summary_results = []

    # 2. Perform McNemar's Tests
    for other_df, name in [(df_rafnet, "R-AFNet"), (df_resnet, "ResNet-18")]:
        p_val, wins, losses = perform_mcnemar(df_cafnet, other_df, "C-AFNet", name)
        significance = "Significant" if p_val < 0.05 else "Not Significant"

        # REQUIRED FIX #3: Safer interpretation wording
        summary_results.append(
            {
                "Comparison": f"C-AFNet vs {name}",
                "Metric": "McNemar p-value",
                "Value": f"{p_val:.4f}",
                "Discordant_b": wins,
                "Discordant_c": losses,
                "Note": f"C-AFNet was correct on {wins} samples where {name} was incorrect; {significance} (p < 0.05)",
            }
        )

    # 3. Perform Bootstrapping
    for df, name in [
        (df_cafnet, "C-AFNet"),
        (df_rafnet, "R-AFNet"),
        (df_resnet, "ResNet-18"),
    ]:
        ci_results = bootstrap_metrics(df)

        # REQUIRED FIX #2: Correct metric label
        summary_results.append(
            {
                "Comparison": name,
                "Metric": "95% CI Macro F1",
                "Value": f"[{ci_results['f1_ci'][0]:.4f}, {ci_results['f1_ci'][1]:.4f}]",
                "Discordant_b": None,
                "Discordant_c": None,
                "Note": "Bootstrap n=1000",
            }
        )

        # RECOMMENDED: Also report accuracy CI
        summary_results.append(
            {
                "Comparison": name,
                "Metric": "95% CI Accuracy",
                "Value": f"[{ci_results['acc_ci'][0]:.4f}, {ci_results['acc_ci'][1]:.4f}]",
                "Discordant_b": None,
                "Discordant_c": None,
                "Note": "Bootstrap n=1000",
            }
        )

    # 4. Save and Print Results
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv("statistical_significance_summary_MIMIC.csv", index=False)
    print(
        "\nStatistical Analysis Complete. Results saved to statistical_significance_summary.csv"
    )
    print(summary_df[["Comparison", "Metric", "Value", "Note"]])


if __name__ == "__main__":
    main()

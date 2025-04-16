import pandas as pd
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score

# Paths
MERGED_PATH = Path("E:/Bumblebee_Recognizer/Data/Output_annotations/predictions_fully_merged.csv")
LABELS_PATH = Path("E:/Bumblebee_Recognizer/Data/Output_annotations/clip_labels_2.csv")
PREDICTIONS_PATH = Path("E:/Bumblebee_Recognizer/Scores")


def load_data(force_reload=False):
    """
    Load predictions and labels, merge, and save merged dataset.
    Set force_reload=True to recompute even if merged file exists.
    """
    # Check if the merged file exists and force_reload is False
    if MERGED_PATH.exists() and not force_reload:
        print("Merged file exists. Loading from disk...")
        # Load the pre-saved merged file
        return pd.read_csv(MERGED_PATH, index_col=[0, 1, 2])

    else:
        # If file doesn't exist or force_reload is True, load raw data
        print("Loading prediction files...")
        prediction_files = glob.glob(str(PREDICTIONS_PATH / "*.csv"))
        all_predictions = pd.concat(
            [pd.read_csv(f, index_col=[0, 1, 2]) for f in prediction_files],
            axis=0
        )
        print(f"Loaded {len(all_predictions):,} predictions")

        clip_labels = pd.read_csv(LABELS_PATH, index_col=[0, 1, 2])
        print(f"Loaded {len(clip_labels):,} clip labels")

        # Clean columns by dropping 'file' column if it exists
        clip_labels = clip_labels.drop(columns=['file'], errors='ignore')
        all_predictions = all_predictions.drop(columns=['file'], errors='ignore')

        # Merge the datasets based on the index
        merged_df = pd.merge(
            clip_labels,
            all_predictions,
            left_index=True,
            right_index=True,
            how='inner',
            suffixes=('_label', '_score')
        )

        merged_df.rename(columns={'BUZZ_label': 'buzz_present', 'BUZZ_score': 'predict_score'}, inplace=True)

        # Save the merged dataset for future use
        merged_df.to_csv(MERGED_PATH)
        print(f"Merged predictions saved to {MERGED_PATH}")
        
        return merged_df


def evaluate_threshold(merged_df, threshold=8):
    """
    Evaluate predictions at a given threshold.
    """
    merged_df['binary_predictions'] = (merged_df['predict_score'] > threshold).astype(int)

    actual = merged_df['buzz_present'].astype(int)
    predicted = merged_df['binary_predictions']
    score = merged_df['predict_score']

    # Confusion matrix
    cm = confusion_matrix(actual, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['absent', 'present'])
    disp.plot()
    plt.title(f"Confusion Matrix (Threshold = {threshold})")
    plt.show()

    print(f"\nEvaluation (Threshold = {threshold}):")
    print(f"F1 Score    : {f1_score(actual, predicted):.3f}")
    print(f"Precision   : {precision_score(actual, predicted):.3f}")
    print(f"Recall      : {recall_score(actual, predicted):.3f}")

    # Histogram
    positives = actual == 1
    negatives = actual == 0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(score[positives], bins=50, alpha=0.5, color="red", label="Buzz present")
    ax.hist(score[negatives], bins=50, alpha=0.5, color="blue", label="Buzz absent")
    ax.set_yscale("log")
    ax.set_ylabel("Number of audio segments")
    ax.set_xlabel("Prediction score")
    ax.legend()
    ax.set_title("Prediction Score Distribution")
    plt.tight_layout()
    plt.show()


def main():
    # Set this to True if you want to recompute from scratch
    merged_df = load_data(force_reload=False)

    # You can now evaluate at different thresholds easily
    evaluate_threshold(merged_df, threshold=4)
    # evaluate_threshold(merged_df, threshold=3.5)
    # evaluate_threshold(merged_df, threshold=5)


if __name__ == "__main__":
    main()
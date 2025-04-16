import pandas as pd
from pathlib import Path
import glob

# Path to the prediction files
predictions_path = Path("E:/Bumblebee_Recognizer/Scores")
prediction_files = glob.glob(str(predictions_path / "*.csv"))

# Load and concatenate all prediction CSVs
all_predictions = pd.concat(
    [pd.read_csv(f, index_col=[0, 1, 2]) for f in prediction_files],
    axis=0
)
print(f"Loaded {len(all_predictions)} predictions")

# Load existing labels from test_set_2
clip_labels = pd.read_csv("E:/Bumblebee_Recognizer/Data/Output_annotations/clip_labels.csv", index_col=[0, 1, 2])
print(f"Loaded {len(clip_labels)} clip labels")

# Align datasets, drop 'file' column if present
clip_labels = clip_labels.drop(columns=['file'], errors='ignore')
all_predictions = all_predictions.drop(columns=['file'], errors='ignore')

# Merge labels and predictions
merged_df = pd.merge(clip_labels, all_predictions, left_index=True, right_index=True, how='inner', suffixes=('_label', '_score'))

# Binarize predictions
threshold = 4  # adjust as needed
merged_df['binary_predictions'] = (merged_df['BUZZ'] > threshold).astype(int)

# Rename for clarity
merged_df.rename(columns={'BUZZ_label': 'buzz_present', 'BUZZ_score': 'predict_score'}, inplace=True)

# Save dataframe
output_path = "E:/Bumblebee_Recognizer/Data/Output_annotations/predictions_fully_merged.csv"
merged_df.to_csv(output_path)
print(f"Merged predictions saved to {output_path}")

# ----------------------
# Evaluation metrics
# ----------------------
actual = merged_df['buzz_present'].astype(int)
predicted = merged_df['binary_predictions']
score = merged_df['predict_score']

# Confusion matrix
cm = confusion_matrix(actual, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['absent', 'present'])
disp.plot()
plt.show()

# Precision, Recall, F1
print("f1:", f1_score(actual, predicted))
print("precision:", precision_score(actual, predicted))
print("recall:", recall_score(actual, predicted))

# ----------------------
# Score distribution histogram
# ----------------------
positives = actual == 1
negatives = actual == 0

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(score[positives], alpha=0.5, color="red", label="Positives")
ax.hist(score[negatives], alpha=0.5, color="blue", label="Negatives")
ax.set_yscale("log")
ax.set_ylabel("Number of audio segments")
ax.set_xlabel("Score")
ax.legend()
plt.tight_layout()
plt.show()

# Load necessary libraries (if you don't have them installed, install with install.packages("dplyr"))
library(dplyr)

# Assuming 'test_set_2' and 'predictions' are your data frames:
test_set_2 <- read.csv("E:\\Bumblebee_Recognizer\\Data\\Output_annotations\\clip_labels_test_4.csv", stringsAsFactors = T)
predictions <- read.csv("E:\\Bumblebee_Recognizer\\Data\\TundraBUZZ_RawAudio\\testing_data\\predict_score_test.csv", stringsAsFactors = T)

# Check the structure of the data frames before modification
str(test_set_2)
str(predictions)

# Convert 'file' columns to character (string) type explicitly
test_set_2$file <- as.character(test_set_2$file)
predictions$file <- as.character(predictions$file)

# Check again the structure after conversion
str(test_set_2)
str(predictions)

# Remove the specified paths using gsub()
test_set_2$file <- basename(test_set_2$file)
predictions$file <- basename(predictions$file)


# Check the results to see the modified file paths
head(test_set_2[, c("file")])
head(predictions[, c("file")])



# Merge by file name, start_time, and end_time
merged_df <- merge(
  test_set_2,
  predictions,
  by = c("file", "start_time", "end_time"),
  all = TRUE  # or use all.x = TRUE for a left join
)

write.csv(merged_df, "E:\\Bumblebee_Recognizer\\Data\\Output_annotations\\merged_predictions_clips_test.csv")

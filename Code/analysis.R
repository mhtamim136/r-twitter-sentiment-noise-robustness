# STEP 1: Setup
# install.packages(c("tidyverse","tm","caret","e1071","SnowballC"))
library(tidyverse)
library(tm)
library(caret)
library(SnowballC)

# STEP 2: Load Data (no header in CSV)
setwd("F:/IDS-Project/Data_Science_final_Project")
data_raw <- read.csv("twitter_training.csv", header = FALSE,
                     col.names = c("tweet_id", "entity", "sentiment", "text"),
                     stringsAsFactors = FALSE)
str(data_raw)
table(data_raw$sentiment)

# STEP 3: Cleaning
data_clean <- data_raw %>%
  filter(!is.na(text), text != "") %>%
  distinct()
nrow(data_clean)
table(data_clean$sentiment)

# STEP 4: Stratified Sampling
set.seed(123)
sample_size <- 5000
data_sample <- data_clean %>%
  group_by(sentiment) %>%
  slice_sample(prop = sample_size / nrow(data_clean)) %>%
  ungroup()
table(data_sample$sentiment)

ggplot(data_sample, aes(x = sentiment, fill = sentiment)) +
  geom_bar() +
  labs(title = "Class Distribution", x = "Sentiment", y = "Count")

# STEP 5: EDA
data_sample$text_length <- sapply(strsplit(data_sample$text, "\\s+"), length)

hist(data_sample$text_length, main = "Distribution of Tweet Length",
     xlab = "Word Count", col = "steelblue")

boxplot(text_length ~ sentiment, data = data_sample,
        main = "Tweet Length by Sentiment", col = "lightgreen")

# STEP 6: Preprocessing
corpus <- Corpus(VectorSource(data_sample$text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)
corpus <- tm_map(corpus, stripWhitespace)

inspect(corpus[1:3])

# STEP 7: Feature Representation
dtm_bow <- DocumentTermMatrix(corpus)
dtm_bow <- removeSparseTerms(dtm_bow, 0.99)
dim(dtm_bow)

dtm_tfidf <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))
dtm_tfidf <- removeSparseTerms(dtm_tfidf, 0.99)
dim(dtm_tfidf)

# --- ADDITION: run this after your existing Step 7 (dim(dtm_tfidf)) ---
# Uses objects already in your session: corpus, dtm_bow, data_sample
# install.packages(c("wordcloud","RColorBrewer"))
library(wordcloud)
library(RColorBrewer)

# Wordcloud (overall, cleaned corpus)
wordcloud(corpus, max.words = 80, colors = brewer.pal(8, "Dark2"))

# Most Frequent Words per Sentiment Class (from BoW matrix)
bow_matrix <- as.matrix(dtm_bow)

for (cls in unique(data_sample$sentiment)) {
  class_freq <- colSums(bow_matrix[data_sample$sentiment == cls, ])
  top10 <- sort(class_freq, decreasing = TRUE)[1:10]
  print(paste("Top words -", cls))
  print(top10)
  
  barplot(top10, las = 2, col = "coral",
          main = paste("Top 10 Words -", cls))
}




# STEP 8: Extra library needed for this part
library(e1071)

# STEP 9: Build model-ready data frames from DTMs
bow_df <- as.data.frame(as.matrix(dtm_bow))
tfidf_df <- as.data.frame(as.matrix(dtm_tfidf))
bow_df$sentiment <- as.factor(data_sample$sentiment)
tfidf_df$sentiment <- as.factor(data_sample$sentiment)

# STEP 10: Train-test split (same index used for both representations)
set.seed(123)
trainIndex <- createDataPartition(bow_df$sentiment, p = 0.8, list = FALSE)

train_bow <- bow_df[trainIndex, ]
test_bow  <- bow_df[-trainIndex, ]
train_tfidf <- tfidf_df[trainIndex, ]
test_tfidf  <- tfidf_df[-trainIndex, ]

# STEP 11: Train classifiers on clean data (BoW)
nb_bow <- naiveBayes(sentiment ~ ., data = train_bow)
pred_nb_bow <- predict(nb_bow, test_bow)
confusionMatrix(pred_nb_bow, test_bow$sentiment)

svm_bow <- svm(sentiment ~ ., data = train_bow, kernel = "linear")
pred_svm_bow <- predict(svm_bow, test_bow)
confusionMatrix(pred_svm_bow, test_bow$sentiment)

# STEP 12: Train classifiers on clean data (TF-IDF)
nb_tfidf <- naiveBayes(sentiment ~ ., data = train_tfidf)
pred_nb_tfidf <- predict(nb_tfidf, test_tfidf)
confusionMatrix(pred_nb_tfidf, test_tfidf$sentiment)

svm_tfidf <- svm(sentiment ~ ., data = train_tfidf, kernel = "linear")
pred_svm_tfidf <- predict(svm_tfidf, test_tfidf)
confusionMatrix(pred_svm_tfidf, test_tfidf$sentiment)

# STEP 13: Noise generation function (word deletion + character swap)
add_noise <- function(text, word_del_prob = 0.15, char_swap_prob = 0.1) {
  words <- unlist(strsplit(text, "\\s+"))
  words <- words[runif(length(words)) > word_del_prob]
  for (i in seq_along(words)) {
    if (nchar(words[i]) > 3 && runif(1) < char_swap_prob) {
      chars <- strsplit(words[i], "")[[1]]
      pos <- sample(1:(length(chars) - 1), 1)
      chars[c(pos, pos + 1)] <- chars[c(pos + 1, pos)]
      words[i] <- paste(chars, collapse = "")
    }
  }
  paste(words, collapse = " ")
}

set.seed(123)
test_text_raw <- data_sample$text[-trainIndex]
test_text_noisy <- sapply(test_text_raw, add_noise)

# STEP 14: Preprocess noisy text (same steps as STEP 6)
corpus_noisy <- Corpus(VectorSource(test_text_noisy))
corpus_noisy <- tm_map(corpus_noisy, content_transformer(tolower))
corpus_noisy <- tm_map(corpus_noisy, removePunctuation)
corpus_noisy <- tm_map(corpus_noisy, removeNumbers)
corpus_noisy <- tm_map(corpus_noisy, removeWords, stopwords("english"))
corpus_noisy <- tm_map(corpus_noisy, stemDocument)
corpus_noisy <- tm_map(corpus_noisy, stripWhitespace)

# STEP 15: Build noisy DTMs using the SAME vocabulary as training
dtm_bow_noisy <- DocumentTermMatrix(corpus_noisy, control = list(dictionary = Terms(dtm_bow)))
dtm_tfidf_noisy <- DocumentTermMatrix(corpus_noisy, control = list(weighting = weightTfIdf,
                                                                   dictionary = Terms(dtm_tfidf)))

test_bow_noisy <- as.data.frame(as.matrix(dtm_bow_noisy))
test_tfidf_noisy <- as.data.frame(as.matrix(dtm_tfidf_noisy))
test_bow_noisy$sentiment <- test_bow$sentiment
test_tfidf_noisy$sentiment <- test_tfidf$sentiment

# STEP 16: Predict on noisy test data
pred_nb_bow_noisy <- predict(nb_bow, test_bow_noisy)
pred_svm_bow_noisy <- predict(svm_bow, test_bow_noisy)
pred_nb_tfidf_noisy <- predict(nb_tfidf, test_tfidf_noisy)
pred_svm_tfidf_noisy <- predict(svm_tfidf, test_tfidf_noisy)

# STEP 17: Collect accuracy for all 8 combinations
results <- data.frame(
  Representation = rep(c("BoW", "TF-IDF"), each = 4),
  Model = rep(c("Naive Bayes", "Naive Bayes", "SVM", "SVM"), 2),
  Condition = rep(c("Clean", "Noisy"), 4),
  Accuracy = c(
    confusionMatrix(pred_nb_bow, test_bow$sentiment)$overall["Accuracy"],
    confusionMatrix(pred_nb_bow_noisy, test_bow_noisy$sentiment)$overall["Accuracy"],
    confusionMatrix(pred_svm_bow, test_bow$sentiment)$overall["Accuracy"],
    confusionMatrix(pred_svm_bow_noisy, test_bow_noisy$sentiment)$overall["Accuracy"],
    confusionMatrix(pred_nb_tfidf, test_tfidf$sentiment)$overall["Accuracy"],
    confusionMatrix(pred_nb_tfidf_noisy, test_tfidf_noisy$sentiment)$overall["Accuracy"],
    confusionMatrix(pred_svm_tfidf, test_tfidf$sentiment)$overall["Accuracy"],
    confusionMatrix(pred_svm_tfidf_noisy, test_tfidf_noisy$sentiment)$overall["Accuracy"]
  )
)
print(results)

# STEP 18: Visualize comparison
ggplot(results, aes(x = Model, y = Accuracy, fill = Condition)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Representation) +
  labs(title = "Clean vs Noisy Accuracy by Representation and Model", y = "Accuracy")


#.............................................................................................................................

# Run this AFTER Part 1 script, in the SAME R session
# (needs: data_sample, corpus, dtm_bow, dtm_tfidf from Part 1)

# STEP 10: Extra libraries
library(e1071)
library(nnet)

# STEP 11: Build model-ready data frames from DTMs
bow_df <- as.data.frame(as.matrix(dtm_bow))
tfidf_df <- as.data.frame(as.matrix(dtm_tfidf))
bow_df$sentiment <- as.factor(data_sample$sentiment)
tfidf_df$sentiment <- as.factor(data_sample$sentiment)

# STEP 12: Train-test split (same index used for both representations)
set.seed(123)
trainIndex <- createDataPartition(bow_df$sentiment, p = 0.8, list = FALSE)

train_bow <- bow_df[trainIndex, ];  test_bow  <- bow_df[-trainIndex, ]
train_tfidf <- tfidf_df[trainIndex, ];  test_tfidf  <- tfidf_df[-trainIndex, ]

# STEP 13: Train classifiers on clean data (BoW)
nb_bow  <- naiveBayes(sentiment ~ ., data = train_bow)
svm_bow <- svm(sentiment ~ ., data = train_bow, kernel = "linear")
mlr_bow <- multinom(sentiment ~ ., data = train_bow, trace = FALSE)

# STEP 14: Train classifiers on clean data (TF-IDF)
nb_tfidf  <- naiveBayes(sentiment ~ ., data = train_tfidf)
svm_tfidf <- svm(sentiment ~ ., data = train_tfidf, kernel = "linear")
mlr_tfidf <- multinom(sentiment ~ ., data = train_tfidf, trace = FALSE)

# STEP 15: Noise generation function (word deletion + character swap)
add_noise <- function(text, word_del_prob = 0.15, char_swap_prob = 0.1) {
  words <- unlist(strsplit(text, "\\s+"))
  words <- words[runif(length(words)) > word_del_prob]
  for (i in seq_along(words)) {
    if (nchar(words[i]) > 3 && runif(1) < char_swap_prob) {
      chars <- strsplit(words[i], "")[[1]]
      pos <- sample(1:(length(chars) - 1), 1)
      chars[c(pos, pos + 1)] <- chars[c(pos + 1, pos)]
      words[i] <- paste(chars, collapse = "")
    }
  }
  paste(words, collapse = " ")
}

set.seed(123)
test_text_raw   <- data_sample$text[-trainIndex]
test_text_noisy <- sapply(test_text_raw, add_noise)

# STEP 16: Preprocess noisy text (same steps as Part 1 - Step 6)
corpus_noisy <- Corpus(VectorSource(test_text_noisy))
corpus_noisy <- tm_map(corpus_noisy, content_transformer(tolower))
corpus_noisy <- tm_map(corpus_noisy, removePunctuation)
corpus_noisy <- tm_map(corpus_noisy, removeNumbers)
corpus_noisy <- tm_map(corpus_noisy, removeWords, stopwords("english"))
corpus_noisy <- tm_map(corpus_noisy, stemDocument)
corpus_noisy <- tm_map(corpus_noisy, stripWhitespace)

# STEP 17: Build noisy DTMs using the SAME vocabulary as training
dtm_bow_noisy   <- DocumentTermMatrix(corpus_noisy, control = list(dictionary = Terms(dtm_bow)))
dtm_tfidf_noisy <- DocumentTermMatrix(corpus_noisy, control = list(weighting = weightTfIdf,
                                                                   dictionary = Terms(dtm_tfidf)))
test_bow_noisy   <- as.data.frame(as.matrix(dtm_bow_noisy))
test_tfidf_noisy <- as.data.frame(as.matrix(dtm_tfidf_noisy))
test_bow_noisy$sentiment   <- test_bow$sentiment
test_tfidf_noisy$sentiment <- test_tfidf$sentiment

# STEP 18: Helper - Accuracy + macro Precision/Recall/F1 from confusionMatrix
get_metrics <- function(pred, actual) {
  cm <- confusionMatrix(pred, actual)
  c(Accuracy  = cm$overall["Accuracy"],
    Precision = mean(cm$byClass[, "Precision"], na.rm = TRUE),
    Recall    = mean(cm$byClass[, "Recall"], na.rm = TRUE),
    F1        = mean(cm$byClass[, "F1"], na.rm = TRUE))
}

# STEP 19: Predict clean + noisy for all 3 models x 2 representations
results <- data.frame()

combos <- list(
  list(rep = "BoW",    model = "Naive Bayes", fit = nb_bow,  clean = test_bow,   noisy = test_bow_noisy),
  list(rep = "BoW",    model = "SVM",         fit = svm_bow, clean = test_bow,   noisy = test_bow_noisy),
  list(rep = "BoW",    model = "Multinom LR", fit = mlr_bow, clean = test_bow,   noisy = test_bow_noisy),
  list(rep = "TF-IDF", model = "Naive Bayes", fit = nb_tfidf,  clean = test_tfidf, noisy = test_tfidf_noisy),
  list(rep = "TF-IDF", model = "SVM",         fit = svm_tfidf, clean = test_tfidf, noisy = test_tfidf_noisy),
  list(rep = "TF-IDF", model = "Multinom LR", fit = mlr_tfidf, clean = test_tfidf, noisy = test_tfidf_noisy)
)

for (c in combos) {
  pred_clean <- predict(c$fit, c$clean)
  pred_noisy <- predict(c$fit, c$noisy)
  m_clean <- get_metrics(pred_clean, c$clean$sentiment)
  m_noisy <- get_metrics(pred_noisy, c$noisy$sentiment)
  results <- rbind(results,
                   data.frame(Representation = c$rep, Model = c$model, Condition = "Clean", t(m_clean)),
                   data.frame(Representation = c$rep, Model = c$model, Condition = "Noisy", t(m_noisy)))
}

names(results) <- c("Representation", "Model", "Condition", "Accuracy", "Precision", "Recall", "F1")
print(results)

# STEP 20: Visualize comparison
ggplot(results, aes(x = Model, y = Accuracy, fill = Condition)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Representation) +
  labs(title = "Clean vs Noisy Accuracy by Representation and Model", y = "Accuracy") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))




#.....................................................................................

# Run this AFTER Part 2, in the SAME R session
# (needs: nb_bow, svm_bow, mlr_bow, nb_tfidf, svm_tfidf, mlr_tfidf,
#          test_bow, test_tfidf, trainIndex, dtm_bow, dtm_tfidf, data_sample, add_noise)

# STEP 21: Noise-level sweep settings
noise_levels <- c(0.10, 0.20, 0.30)   # word-deletion probability
n_repeats <- 5                         # repeat each level 5x for stability

test_text_raw <- data_sample$text[-trainIndex]

models_list <- list(
  list(rep = "BoW",    model = "Naive Bayes", fit = nb_bow,    sentiment_ref = test_bow$sentiment,   tfidf = FALSE),
  list(rep = "BoW",    model = "SVM",         fit = svm_bow,   sentiment_ref = test_bow$sentiment,   tfidf = FALSE),
  list(rep = "BoW",    model = "Multinom LR", fit = mlr_bow,   sentiment_ref = test_bow$sentiment,   tfidf = FALSE),
  list(rep = "TF-IDF", model = "Naive Bayes", fit = nb_tfidf,  sentiment_ref = test_tfidf$sentiment, tfidf = TRUE),
  list(rep = "TF-IDF", model = "SVM",         fit = svm_tfidf, sentiment_ref = test_tfidf$sentiment, tfidf = TRUE),
  list(rep = "TF-IDF", model = "Multinom LR", fit = mlr_tfidf, sentiment_ref = test_tfidf$sentiment, tfidf = TRUE)
)

# STEP 22: Run sweep (each noise level x each repeat x each model)
library(tm)
library(e1071)    # naiveBayes + svm  → provides predict.naiveBayes / predict.svm
library(nnet)     # multinom          → provides predict.multinom
library(dplyr)    # %>%, group_by, summarise  (STEP 23)
library(ggplot2)  # ggplot                     (STEP 24)
sweep_results <- data.frame()

for (lvl in noise_levels) {
  for (r in 1:n_repeats) {
    set.seed(1000 + r)   # same seed across models within a repeat = fair comparison
    noisy_text <- sapply(test_text_raw, add_noise, word_del_prob = lvl, char_swap_prob = 0.10)
    
    corpus_n <- Corpus(VectorSource(noisy_text))
    corpus_n <- tm_map(corpus_n, content_transformer(tolower))
    corpus_n <- tm_map(corpus_n, removePunctuation)
    corpus_n <- tm_map(corpus_n, removeNumbers)
    corpus_n <- tm_map(corpus_n, removeWords, stopwords("english"))
    corpus_n <- tm_map(corpus_n, stemDocument)
    corpus_n <- tm_map(corpus_n, stripWhitespace)
    
    dtm_bow_n   <- DocumentTermMatrix(corpus_n, control = list(dictionary = Terms(dtm_bow)))
    dtm_tfidf_n <- DocumentTermMatrix(corpus_n, control = list(weighting = weightTfIdf, dictionary = Terms(dtm_tfidf)))
    test_bow_n   <- as.data.frame(as.matrix(dtm_bow_n))
    test_tfidf_n <- as.data.frame(as.matrix(dtm_tfidf_n))
    
    for (m in models_list) {
      newdata <- if (m$tfidf) test_tfidf_n else test_bow_n
      pred <- predict(m$fit, newdata)
      acc <- mean(pred == m$sentiment_ref)
      sweep_results <- rbind(sweep_results, data.frame(
        Representation = m$rep, Model = m$model,
        NoiseLevel = lvl, Repeat = r, Accuracy = acc))
    }
  }
  cat("Finished noise level:", lvl, "\n")
}

# STEP 23: Summarize - mean and SD across the 5 repeats
sweep_summary <- sweep_results %>%
  group_by(Representation, Model, NoiseLevel) %>%
  summarise(MeanAccuracy = mean(Accuracy), SD = sd(Accuracy), .groups = "drop")

print(sweep_summary)

# STEP 24: Plot degradation trend (mean +/- 1 SD)
ggplot(sweep_summary, aes(x = NoiseLevel, y = MeanAccuracy, color = Model, group = Model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = MeanAccuracy - SD, ymax = MeanAccuracy + SD), width = 0.01) +
  facet_wrap(~ Representation) +
  labs(title = "Accuracy vs Noise Level (mean of 5 repeats, error bars = SD)",
       x = "Word Deletion Probability", y = "Mean Accuracy")



#.....................................................................................


# Run this AFTER Part 2 (needs: results, bow_df, tfidf_df from Part 2)
# Independent of Part 3 (noise sweep) - can run before or after it

# STEP 25: Bubble Chart - Precision vs Recall, bubble size = Accuracy, color = Model
ggplot(results, aes(x = Recall, y = Precision, size = Accuracy, color = Model, shape = Condition)) +
  geom_point(alpha = 0.7) +
  facet_wrap(~ Representation) +
  scale_size(range = c(3, 12)) +
  labs(title = "Model Performance Bubble Chart (bubble size = Accuracy)",
       x = "Recall", y = "Precision")

# STEP 26: 5-fold Cross-Validation (clean data, stability check)
set.seed(123)
folds <- createFolds(bow_df$sentiment, k = 5, list = TRUE)

cv_results <- data.frame()

run_cv <- function(df, repname) {
  for (i in 1:5) {
    test_idx   <- folds[[i]]
    train_fold <- df[-test_idx, ]
    test_fold  <- df[test_idx, ]
    
    nb_fit  <- naiveBayes(sentiment ~ ., data = train_fold)
    svm_fit <- svm(sentiment ~ ., data = train_fold, kernel = "linear")
    mlr_fit <- multinom(sentiment ~ ., data = train_fold, trace = FALSE)
    
    accs <- c(
      "Naive Bayes" = mean(predict(nb_fit, test_fold) == test_fold$sentiment),
      "SVM"         = mean(predict(svm_fit, test_fold) == test_fold$sentiment),
      "Multinom LR" = mean(predict(mlr_fit, test_fold) == test_fold$sentiment)
    )
    for (mod in names(accs)) {
      cv_results <<- rbind(cv_results, data.frame(
        Representation = repname, Model = mod, Fold = i, Accuracy = accs[[mod]]))
    }
  }
  cat("Finished CV for:", repname, "\n")
}

run_cv(bow_df, "BoW")
run_cv(tfidf_df, "TF-IDF")

# STEP 27: Summarize CV results (mean +/- SD across 5 folds)
cv_summary <- cv_results %>%
  group_by(Representation, Model) %>%
  summarise(MeanAccuracy = mean(Accuracy), SD = sd(Accuracy), .groups = "drop")

print(cv_summary)

ggplot(cv_summary, aes(x = Model, y = MeanAccuracy, fill = Representation)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_errorbar(aes(ymin = MeanAccuracy - SD, ymax = MeanAccuracy + SD),
                position = position_dodge(0.9), width = 0.2) +
  labs(title = "5-Fold Cross-Validation Accuracy (mean +/- SD)", y = "Accuracy")





#...................................................................................................


# ==================================================================
# PART 5 (MINIMAL) : only the two outputs the paper is actually missing
#   A. Confusion matrix + per-class Precision/Recall/F1  -> fixes problem 5
#   B. Per-fold 5-fold CV table (5 folds visible)        -> fixes problem 7
# Run AFTER your Part 4, in the SAME R session.
# ==================================================================

## ---- libraries FIRST (this is what caused your two earlier errors) ----
library(caret); library(e1071); library(nnet); library(ggplot2)

OUT <- "F:/IDS-Project/paper_outputs"   # new folder; change if you want
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)


# ==================================================================
# A.  CONFUSION MATRIX + PER-CLASS METRICS   (problem 5)
#     Only predicts with models you already trained - no refitting.
# ==================================================================
combo_list <- list(
  list(rep = "BoW",    model = "Naive Bayes", fit = nb_bow,    cl = test_bow,   ns = test_bow_noisy),
  list(rep = "BoW",    model = "SVM",         fit = svm_bow,   cl = test_bow,   ns = test_bow_noisy),
  list(rep = "BoW",    model = "Multinom LR", fit = mlr_bow,   cl = test_bow,   ns = test_bow_noisy),
  list(rep = "TF-IDF", model = "Naive Bayes", fit = nb_tfidf,  cl = test_tfidf, ns = test_tfidf_noisy),
  list(rep = "TF-IDF", model = "SVM",         fit = svm_tfidf, cl = test_tfidf, ns = test_tfidf_noisy),
  list(rep = "TF-IDF", model = "Multinom LR", fit = mlr_tfidf, cl = test_tfidf, ns = test_tfidf_noisy)
)

perclass <- data.frame(); overall <- data.frame()

for (cb in combo_list) {
  for (cond in c("Clean", "Noisy")) {
    nd <- if (cond == "Clean") cb$cl else cb$ns
    cm <- confusionMatrix(predict(cb$fit, nd), nd$sentiment)
    
    cat("\n---------------------------------------------------------\n")
    cat(sprintf("%s | %s | %s\n", cb$rep, cb$model, cond))
    cat("---------------------------------------------------------\n")
    print(cm$table)
    cat(sprintf("Accuracy = %.4f   Kappa = %.4f   Baseline(NIR) = %.4f\n",
                cm$overall["Accuracy"], cm$overall["Kappa"], cm$overall["AccuracyNull"]))
    print(round(cm$byClass[, c("Precision", "Recall", "F1")], 4))
    
    bc <- as.data.frame(cm$byClass[, c("Precision", "Recall", "F1")])
    bc$Class   <- sub("Class: ", "", rownames(bc))
    bc$Support <- as.numeric(table(nd$sentiment)[bc$Class])
    bc$Representation <- cb$rep; bc$Model <- cb$model; bc$Condition <- cond
    perclass <- rbind(perclass, bc)
    
    overall <- rbind(overall, data.frame(
      Representation = cb$rep, Model = cb$model, Condition = cond,
      Accuracy = cm$overall["Accuracy"], Kappa = cm$overall["Kappa"],
      Baseline = cm$overall["AccuracyNull"]))
  }
}
rownames(perclass) <- NULL; rownames(overall) <- NULL
write.csv(perclass, file.path(OUT, "table_perclass_metrics.csv"), row.names = FALSE)
write.csv(overall,  file.path(OUT, "table_overall_metrics.csv"),  row.names = FALSE)

cat("\n\n### TABLE for paper: overall accuracy + Kappa ###\n")
print(overall, digits = 4, row.names = FALSE)

# The single confusion matrix the paper will print (best model = MLR + BoW, clean)
cm_best <- confusionMatrix(predict(mlr_bow, test_bow), test_bow$sentiment)
cat("\n\n### CONFUSION MATRIX for the paper (Multinom LR + BoW, clean) ###\n")
print(cm_best$table)
write.csv(as.data.frame.matrix(cm_best$table),
          file.path(OUT, "cm_best_MLR_BoW_clean.csv"))

# Optional figure of the same matrix, for your images folder
cm_df <- as.data.frame(cm_best$table)
p_cm <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = Freq), size = 4) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix: Multinomial LR + BoW (clean test set)",
       subtitle = sprintf("Accuracy = %.3f, Kappa = %.3f",
                          cm_best$overall["Accuracy"], cm_best$overall["Kappa"])) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
ggsave(file.path(OUT, "fig_confusion_matrix.png"), p_cm, width = 6, height = 5, dpi = 300)


# ==================================================================
# B.  PER-FOLD 5-FOLD CV TABLE   (problem 7)
#     Reuses cv_results from your Part 4 if it is still in the session.
#     Only refits if it is missing (e.g. you restarted R).
# ==================================================================
if (exists("cv_results") && nrow(cv_results) == 30) {
  cat("\n\nUsing cv_results already in your session - no refitting needed.\n")
} else {
  cat("\n\ncv_results not found; recomputing 5-fold CV (takes a few minutes)...\n")
  set.seed(123)
  folds <- createFolds(bow_df$sentiment, k = 5, list = TRUE)
  cv_results <- data.frame()
  for (repname in c("BoW", "TF-IDF")) {
    df <- if (repname == "BoW") bow_df else tfidf_df
    for (i in 1:5) {
      tr <- df[-folds[[i]], ]; te <- df[folds[[i]], ]
      accs <- c(
        "Naive Bayes" = mean(predict(naiveBayes(sentiment ~ ., data = tr), te) == te$sentiment),
        "SVM"         = mean(predict(svm(sentiment ~ ., data = tr, kernel = "linear"), te) == te$sentiment),
        "Multinom LR" = mean(predict(multinom(sentiment ~ ., data = tr, trace = FALSE), te) == te$sentiment))
      for (mod in names(accs))
        cv_results <- rbind(cv_results, data.frame(
          Representation = repname, Model = mod, Fold = i, Accuracy = accs[[mod]]))
      cat("  ", repname, "fold", i, "done\n")
    }
  }
}

cat("\n### ALL 30 PER-FOLD RESULTS (long form) ###\n")
print(cv_results, digits = 4, row.names = FALSE)

cat("\n\n### TABLE for paper: 5 folds side by side ###\n")
wide <- reshape(cv_results[, c("Representation", "Model", "Fold", "Accuracy")],
                idvar = c("Representation", "Model"), timevar = "Fold", direction = "wide")
names(wide) <- c("Representation", "Model", paste0("Fold", 1:5))
wide$Mean <- rowMeans(wide[, 3:7])
wide$SD   <- apply(wide[, 3:7], 1, sd)
print(wide, digits = 4, row.names = FALSE)
write.csv(wide, file.path(OUT, "table_cv_perfold_wide.csv"), row.names = FALSE)

# Figure showing all 5 folds (replaces fig8, which only showed the means)
p_cv <- ggplot(cv_results, aes(x = factor(Fold), y = Accuracy, fill = Model)) +
  geom_col(position = "dodge") +
  facet_wrap(~ Representation) +
  geom_hline(yintercept = max(table(bow_df$sentiment)) / nrow(bow_df), linetype = "dashed") +
  labs(title = "Per-Fold 5-Fold Cross-Validation Accuracy",
       subtitle = "dashed line = majority-class baseline", x = "Fold", y = "Accuracy")
ggsave(file.path(OUT, "fig_cv_perfold.png"), p_cv, width = 9, height = 4.5, dpi = 300)

cat("\n\nDONE. Written to:", OUT, "\n")
cat("  2 images: fig_confusion_matrix.png, fig_cv_perfold.png\n")
cat("  CSVs: table_perclass_metrics, table_overall_metrics, cm_best_MLR_BoW_clean, table_cv_perfold_wide\n")



#.............................................................................
# STEP 7-ADD: Wordcloud from the actual 177-term BoW vocabulary (matches Fig. 2 caption)
library(wordcloud)
library(RColorBrewer)

# ============================================

# ============================================

library(wordcloud)
library(RColorBrewer)

make_wordcloud <- function(n_words = 24) {
  library(wordcloud)
  library(RColorBrewer)
  
  term_freq <- sort(colSums(as.matrix(dtm_bow)), decreasing = TRUE)
  freq_df <- data.frame(word = names(term_freq), freq = as.numeric(term_freq))
  
  exclude_words <- c("fuck", "shit", "damn", "hell", "ass", "bitch")
  freq_df <- freq_df[!(freq_df$word %in% exclude_words), ]
  freq_df <- freq_df[1:n_words, ]
  
  set.seed(123)
  png("fig2_wordcloud.png", width = 1400, height = 1000, res = 300)
  par(mar = c(0, 0, 0, 0))
  wordcloud(words = freq_df$word, freq = freq_df$freq,
            scale = c(4, 0.8),
            random.order = FALSE,
            rot.per = 0.15,
            colors = brewer.pal(8, "Dark2"))
  dev.off()
  
  cat("Done. Fig. 2 caption should say:", n_words, "most frequent terms\n")
}
t
make_wordcloud(24)
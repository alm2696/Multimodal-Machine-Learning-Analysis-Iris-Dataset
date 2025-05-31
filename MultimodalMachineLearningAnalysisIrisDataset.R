# Install necessary packages
# install.packages("randomForest")
# install.packages("ggplot2")
# install.packages("e1071")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("caret")
# install.packages("cluster")
# install.packages("rgl")

# Load necessary libraries
library(randomForest)
library(datasets)
library(ggplot2)
library(e1071)
library(rpart)
library(rpart.plot)
library(caret)
library(cluster)
library(rgl)

# 1. Classification using Random Forest

# View the first few rows of the dataset
head(iris)

# Split the dataset into training and testing sets (70% training, 30% testing)
set.seed(123)  # For reproducibility
trainIndex <- sample(1:nrow(iris), 0.7 * nrow(iris))

trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

# Train a Random Forest model
rf_model <- randomForest(Species ~ ., data = trainData, ntree = 100)

# Print the model summary
print(rf_model)

# Predict using the trained model
predictions <- predict(rf_model, testData)

# Create a confusion matrix to evaluate the model
confMatrix <- table(Predicted = predictions, Actual = testData$Species)

# Print the confusion matrix
print(confMatrix)

# Calculate accuracy
accuracy <- sum(diag(confMatrix)) / sum(confMatrix)
cat("Accuracy:", accuracy, "\n")


# 2. Regression using Linear Regression

# Fit a linear regression model to predict Sepal.Length based on the other variables
lm_model <- lm(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, data = iris)

# Print the model summary
summary(lm_model)

# Make predictions
predict_lm <- predict(lm_model, iris)

# Plot actual vs predicted values
plot(iris$Sepal.Length, predict_lm, main = "Actual vs Predicted Sepal Length",
     xlab = "Actual Sepal Length", ylab = "Predicted Sepal Length", col = "blue")
abline(a = 0, b = 1, col = "red")


# 3. Clustering using k-means

# Use k-means clustering to group the data
kmeans_model <- kmeans(iris[, -5], centers = 4)

# Print the clustering results
print(kmeans_model)

# Add the clustering results to the Iris dataset
iris$Cluster <- as.factor(kmeans_model$cluster)

# Plot the clustering results
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Cluster)) +
  geom_point() +
  labs(title = "k-means Clustering of Iris Dataset")


# 4. Principal Component Analysis for dimensionality reduction

# Perform PCA on the numeric columns
pca_model <- prcomp(iris[, 1:4], scale. = TRUE)

# Print summary of PCA results
summary(pca_model)

# Plot the first two principal components
plot(pca_model$x[, 1], pca_model$x[, 2], col = as.numeric(iris$Species),
     main = "PCA of Iris Dataset", xlab = "PCA1", ylab = "PCA2")
legend("topright", legend = levels(iris$Species), col = 1:3, pch = 1)


# 5. Support Vector Machine for Classification

# Split the dataset into training and testing sets (70% training, 30% testing)
trainIndex <- sample(1:nrow(iris), 0.7 * nrow(iris))

trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

# Train an SVM model
svm_model <- svm(Species ~ ., data = trainData)

# Predict using the trained model
predict_svm <- predict(svm_model, testData)

# Create a confusion matrix to evaluate the model
confMatrix_svm <- table(Predicted = predict_svm, Actual = testData$Species)

# Print the confusion matrix
print(confMatrix_svm)

# Calculate accuracy
accur_svm <- sum(diag(confMatrix_svm)) / sum(confMatrix_svm)
cat("Accuracy (SVM):", accur_svm, "\n")


# 6. Decision Tree for Classification

# Train a decision tree model
tree_model <- rpart(Species ~ ., data = iris, method = "class")

# Print the model summary
summary(tree_model)

# Visualize the decision tree
rpart.plot(tree_model)


# 7. Combined analysis techniques

# Step 1: Preprocessing with PCA
iris_data <- iris[, 1:4]

# Apply PCA to reduce the dimensionality
pca_model <- prcomp(iris_data, scale. = TRUE)

# Print PCA summary
summary(pca_model)

# Project data into 2D for visualization
pca_data <- pca_model$x[, 1:2]

# Add PCA components to the original data for classification and clustering
iris$pca1 <- pca_data[, 1]
iris$pca2 <- pca_data[, 2]

# Step 2: Classification using Random Forest
# Split the data into training and testing sets (70% training, 30% testing)
trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)

trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

# Train a Random Forest classifier using the PCA components
rf_model <- randomForest(Species ~ pca1 + pca2, data = trainData, ntree = 100)

# Predict the species in the test set
rf_predict <- predict(rf_model, testData)

# Evaluate the model with a confusion matrix
confMatrix_rf <- table(Predicted = rf_predict, Actual = testData$Species)
print(confMatrix_rf)

# Step 3: Clustering using k-means
# Apply k-means clustering on the PCA components
kmeans_model <- kmeans(iris[, c("pca1", "pca2")], centers = 3)

# Add the cluster assignments to the data
iris$Cluster <- as.factor(kmeans_model$cluster)

# Step 4: Visualize the PCA components and clustering results
ggplot(iris, aes(x = pca1, y = pca2, color = Cluster)) +
  geom_point() +
  labs(title = "PCA of Iris Dataset with K-means Clustering") +
  theme_minimal()

# Visualize the classification results
ggplot(iris, aes(x = pca1, y = pca2, color = Species)) +
  geom_point() +
  labs(title = "PCA of Iris Dataset with Random Forest Classification") +
  theme_minimal()


# 8. Hyper-parameter Tuning for Random Forest

# Tuning the Random Forest model
tune_rf <- train(Species ~ pca1 + pca2, data = trainData, 
                 method = "rf", 
                 trControl = trainControl(method = "cv", number = 5), 
                 tuneGrid = expand.grid(mtry = 1:2),
                 ntree = 100)

# Best model after tuning
print(tune_rf$bestTune)

# Predictions with the tuned model
rf_predict_tuned <- predict(tune_rf, testData)

# Confusion matrix for the tuned model
confMatrix_rf_tuned <- table(Predicted = rf_predict_tuned, Actual = testData$Species)
print(confMatrix_rf_tuned)


# 9. K-means Clustering with Silhouette Score

# Silhouette Score
silho_scores <- sapply(2:10, function(k) {
  kmeans_model <- kmeans(iris[, c("pca1", "pca2")], centers = k)
  sil <- silhouette(kmeans_model$cluster, dist(iris[, c("pca1", "pca2")]))
  return(mean(sil[, 3]))  # Extract the average silhouette width
})

# Plot silhouette scores to choose the best k
plot(2:10, silho_scores, type = "b", main = "Silhouette Scores", 
     xlab = "Number of Clusters", ylab = "Silhouette Score")

# Optimal number of clusters
best_k <- which.max(silho_scores)
cat("Optimal number of clusters:", best_k, "\n")

# Re-run k-means with the optimal k
final_kmeans_model <- kmeans(iris[, c("pca1", "pca2")], centers = best_k)
iris$Cluster <- as.factor(final_kmeans_model$cluster)

# Visualize the final clustering
ggplot(iris, aes(x = pca1, y = pca2, color = Cluster)) +
  geom_point() +
  labs(title = paste("PCA of Iris Dataset with K-means Clustering (k=", best_k, ")", sep = "")) +
  theme_minimal()


# 10. Evaluate Model Performance Using Cross-Validation

# Cross-validation for Random Forest
cv_rf <- train(Species ~ pca1 + pca2, data = iris, 
               method = "rf", 
               trControl = trainControl(method = "cv", number = 10))

# Print cross-validation results
print(cv_rf)

# Predictions with cross-validation model on the test set
rf_predict_cv <- predict(cv_rf, testData)

# Confusion matrix for the cross-validation model
confMatrix_cv <- table(Predicted = rf_predict_cv, Actual = testData$Species)
print(confMatrix_cv)


# 11. Visualize Classification and Clustering Results in 3D

# Convert Species to colors
species_colors <- as.factor(iris$Species)
colors <- rainbow(length(unique(species_colors)))[species_colors]

# Plot for K-means clustering
plot3d(iris$pca1, iris$pca2, iris$pca3, col = as.factor(iris$Cluster), 
       size = 3, main = "PCA 3D - K-means Clustering")

# Plot for Random Forest classification
plot3d(iris$pca1, iris$pca2, iris$pca3, col = colors, 
       size = 3, main = "PCA 3D - Random Forest Classification")


# 12. Model interpretability feature importance with Random Forest

# Get feature importance from the Random Forest model
importance(rf_model)

# Visualize the importance of features
varImpPlot(rf_model)

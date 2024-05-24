library(quantmod)
library(xts)
library(TTR)
library(PerformanceAnalytics)
library(caret)
library(Metrics)
library(ggplot2)
library(stats)
library(psych)

#Get Stock data
getSymbols("AAPL", src = "yahoo")

#Exploratory Data Analysis
head(AAPL)
tail(AAPL)
summary(AAPL)

#Some Indicators
AAPL$SMA20 <- SMA(AAPL$AAPL.Close, n = 20) # 20-day Simple Moving Average; SMA determines the arithmetic average of the series based on the last n observation
AAPL$RSI <- RSI(AAPL$AAPL.Close) # Relative Strength Index; Relative Strength Index (RSI) calculates a ratio of the recent upward price movements to the absolute price movement
AAPL$MACD <- MACD(AAPL$AAPL.Close) # Moving Average (MA) Convergence Divergence;he rate of change between the fast MA and the slow MA
AAPL$OBV <- OBV(AAPL$AAPL.Close,AAPL$AAPL.Volume)
AAPL$ATR <- ATR(AAPL[, c("AAPL.High", "AAPL.Low", "AAPL.Close")])

#Visualization
ggplot(data = AAPL, aes(x = index(AAPL))) +
  geom_line(aes(y = SMA20, color = "SMA20")) +
  geom_line(aes(y = RSI, color = "RSI")) +
  geom_line(aes(y = MACD, color = "MACD")) +
  labs(title = "Indicators for AAPL") +
  scale_color_manual(values = c("SMA20" = "blue", "RSI" = "red", "MACD" = "green")) +
  theme_minimal()

ggplot(data = AAPL, aes(x = index(AAPL), y = AAPL.Close)) +
geom_line() +
labs(title = "Apple Inc. Stock Prices")

# Convert RSI to returns
AAPL$RSI_returns <- ROC(AAPL$RSI, type = "discrete", na.pad = FALSE) 

# Plot performance summary
charts.PerformanceSummary(AAPL$RSI_returns, main = "RSI Performance Summary")

# Remove missing values
data <- na.omit(AAPL)

# Perform PCA
pca_result <- prcomp(data, scale. = TRUE)

# Print summary of PCA
summary(pca_result)

# Plot PCA results (Scree plot)
plot(pca_result, type = "l", main = "Scree Plot for PCA")

# Plot PCA results (Biplot)
biplot(pca_result, scale = 0, cex = 0.8)

#Split data into training and testing sets
set.seed(123) # for reproducibility
train_index <- createDataPartition(data$RSI_returns, p = 0.8, list = FALSE)
train_subset <- data[train_index, ]
test_subset <- data[-train_index, ]

# Train linear regression model
model <- lm(RSI_returns ~ AAPL.Open + SMA20 + RSI + MACD + ATR + macd + tr , data = train_subset)
summary(model)
# Make predictions on test data
predictions <- predict(model, newdata = test_subset)
r_squared <- R2(predictions, test_subset$RSI_returns)
mae <- MAE(predictions, test_subset$RSI_returns)
mse <- mse(predictions, test_subset$RSI_returns)
rmse <- RMSE(predictions, test_subset$RSI_returns)

cat("R-squared:", r_squared, "\n")
cat("Mean Absolute Error (MAE):", mae, "\n")
cat("Mean Squared Error (MSE):", mse, "\n")
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

# Analyzing the Residual
residuals <- test_subset$RSI_returns - predictions
plot(predictions, residuals, main = "Residuals vs Fitted", xlab = "Fitted values", ylab = "Residuals")
abline(h = 0, col = "red")

# Conducting a Cross-Validation
cv_results <- train(
  RSI_returns ~ AAPL.Open + SMA20 + RSI + MACD + ATR + macd + tr,
  data = train_data,
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)
print(cv_results)

baseline_model <- lm(RSI_returns ~ 1, data = train_subset) # Simple baseline model (predicting mean)
baseline_predictions <- predict(baseline_model, newdata = test_subset)
baseline_mae <- MAE(baseline_predictions, test_subset$RSI_returns)

cat("Baseline MAE:", baseline_mae, "\n")

#Visualize
test_data_pred<-cbind(predictions, test_subset)
test_data_pred<-na.omit(test_data_pred)
ggplot(data = test_data_pred, aes(x = 1:nrow(test_data_pred))) +
  geom_line(aes(y = predictions, color = "Predicted")) +
  geom_line(aes(y = RSI_returns, color = "Actual")) +
  labs(title = "Actual vs Predicted RSI Return for AAPL") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()

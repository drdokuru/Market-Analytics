### Introduction

This project aims to analyze historical stock data for Apple Inc. (AAPL) using various technical indicators and machine learning models. The objective is to predict future returns based on the Relative Strength Index (RSI) and other key indicators derived from the stock price and volume data.

### Data Collection and Exploration

To begin, we collected stock data for AAPL from Yahoo Finance using the quantmod package. The exploratory data analysis phase involved examining the structure and characteristics of the dataset. We inspected the head, tail, and summary statistics of the data to gain insights into its distribution and variability.

```
# Load necessary libraries
library(quantmod)
library(xts)
library(TTR)
library(PerformanceAnalytics)
library(caret)
library(Metrics)
library(ggplot2)
library(stats)
library(psych)

# Get Stock data
getSymbols("AAPL", src = "yahoo")

# Exploratory Data Analysis
head(AAPL)
tail(AAPL)
summary(AAPL)
```

### Technical Indicators Calculation

We calculated several technical indicators commonly used in financial analysis, including the 20-day Simple Moving Average (SMA20), RSI, Moving Average Convergence Divergence (MACD), On-Balance Volume (OBV), and Average True Range (ATR). These indicators provide valuable insights into price trends, momentum, volatility, and trading volume dynamics in the stock market.

```
# Some Indicators
AAPL$SMA20 <- SMA(AAPL$AAPL.Close, n = 20)
AAPL$RSI <- RSI(AAPL$AAPL.Close)
AAPL$MACD <- MACD(AAPL$AAPL.Close)
AAPL$OBV <- OBV(AAPL$AAPL.Close, AAPL$AAPL.Volume)
AAPL$ATR <- ATR(AAPL[, c("AAPL.High", "AAPL.Low", "AAPL.Close")])
```

### Visualization

Using the ggplot2 package, we visualized the calculated technical indicators to gain a deeper understanding of AAPL's historical performance. Line charts were created to visualize the trends in SMA20, RSI, and MACD over time. Additionally, we plotted AAPL's stock prices to visualize its historical price movements and identify potential patterns or trends.

```
# Visualization
library(ggplot2)

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

```

### Conversion of RSI to Returns

One of the key transformations performed in this analysis was the conversion of RSI values to returns. This was achieved using the ROC function from the TTR package. By calculating the discrete rate of change in RSI values over time, we obtained a measure of the percentage change in market momentum, which can be indicative of potential shifts in stock returns.

```
# Convert RSI to returns
library(TTR)
AAPL$RSI_returns <- ROC(AAPL$RSI, type = "discrete", na.pad = FALSE)

```

### Performance Evaluation and Model Development

We evaluated the predictive performance of the RSI returns using the PerformanceAnalytics package, generating a performance summary chart to assess the effectiveness of the indicator in predicting future returns. Additionally, we developed a linear regression model trained on the RSI returns and other technical indicators to forecast future stock returns. Model evaluation metrics such as R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) were computed to assess the model's accuracy and reliability.

```
# Plot performance summary
library(PerformanceAnalytics)
charts.PerformanceSummary(AAPL$RSI_returns, main = "RSI Performance Summary")

# Remove missing values
data <- na.omit(AAPL)

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

```

### Residual Analysis and Cross-Validation

I analyzed the residuals of the regression model to assess its predictive performance and conducted cross-validation to validate the model's robustness. Furthermore, I compared the model's performance against a baseline model predicting the mean returns.

```
# Analyzing the Residual
residuals <- test_subset$RSI_returns - predictions
plot(predictions, residuals, main = "Residuals vs Fitted", xlab = "Fitted values", ylab = "Residuals")
abline(h = 0, col = "red")

# Conducting a Cross-Validation
cv_results <- train(
  RSI_returns ~ AAPL.Open + SMA20 + RSI + MACD + ATR + macd + tr,
  data = train_subset,
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
  geom_line(aes(y = RSI_returns, color = "Actual")) +
  geom_line(aes(y = predictions, color = "Predicted")) +
  labs(title = "Actual vs Predicted RSI Return for AAPL") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()

```

### Conclusion

In conclusion, this project demonstrates the application of technical analysis and machine learning techniques for analyzing historical stock data and predicting future returns. By leveraging RSI and other key indicators, we gain valuable insights into market trends and potential trading opportunities in AAPL stock. The developed regression model provides a valuable tool for forecasting future returns, aiding investors and traders in making informed investment decisions. Further refinement and optimization of the model could enhance its predictive performance and applicability in real-world trading scenarios.

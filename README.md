# 📈 Google Stock Price Prediction Using LSTM

This project aims to predict Google's stock price (GOOGL) using Long Short-Term Memory (LSTM), a type of Recurrent Neural Network (RNN) that is well-suited for processing time series data. The model is trained on historical stock price data and used to forecast future prices.

---

## 🧠 Technologies and Libraries

- Python
- TensorFlow / Keras
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn

---

## 📁 Project Structure

```
📆google-stock-lstm
 ├📊 data/
 ┃ ├ Google_Stock_Price_Train.csv
 ┃ └ Google_Stock_Price_Test.csv
 ├🔢 RNN.ipynb
 ├🔢 README.md
```

---

## 🔍 Project Workflow

1. **Import Libraries**\
   Uses essential libraries such as TensorFlow, Pandas, and Matplotlib to build and visualize the model.

2. **Load Dataset**\
   Loads Google stock price data from CSV files for training and testing.

3. **Preprocessing**

   - Normalize data using `MinMaxScaler`
   - Shape data into time series format with a specified window size

4. **Build LSTM Model**

   - LSTM layers with dropout for regularization
   - Dense output layer

5. **Train the Model**\
   Trains the model using the Adam optimizer and MSE loss function.

6. **Evaluate the Model**\
   Uses metrics like MAE, MSE, and R² score to assess model performance.

7. **Visualize Results**\
   Plots predicted vs actual stock prices to evaluate prediction quality.

---

## 🧠 Model Architecture

The model is built using the Keras Sequential API and consists of the following layers:

1. **LSTM Layer 1:**

   - 50 units (neurons)
   - `return_sequences=True` to pass the output to the next LSTM layer
   - Input shape defined as (timesteps, features)

2. **Dropout Layer 1:**

   - Dropout rate of 20% (`rate=0.2`) to reduce overfitting

3. **LSTM Layer 2:**

   - 50 units
   - `return_sequences=False` since this is the last LSTM layer

4. **Dropout Layer 2:**

   - Another 20% dropout rate

5. **Dense Output Layer:**

   - Single neuron with linear activation to predict the stock price

### Summary of Architecture

- LSTM (50 units, return\_sequences=True)
- Dropout (0.2)
- LSTM (50 units, return\_sequences=False)
- Dropout (0.2)
- Dense (1 unit)

The model is compiled using the **Adam** optimizer and the **Mean Squared Error (MSE)** loss function, which is standard for regression tasks.

## 📊 Model Evaluation

The model was evaluated using the following metrics:

- **Mean Absolute Error (MAE):** Measures the average magnitude of the errors in a set of predictions, without considering their direction.
- **Mean Squared Error (MSE):** Measures the average of the squares of the errors, penalizing larger errors more significantly.
- **R² Score (Coefficient of Determination):** Indicates how well the predictions approximate the actual values. A value closer to 1.0 means a better fit.

Model Prediction Chart
![image](https://github.com/user-attachments/assets/140253df-077f-4e75-b069-05cc7b8c4e73)

Evaluation Results:

```
Mean Squared Error: 133.24
R² Score: 0.3870
```

These results show that the LSTM model performs well in predicting the stock prices with minimal error.

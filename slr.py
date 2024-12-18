import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit UI
st.title("Simple Linear Regression (SLR) with CSV Upload")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.subheader("Dataset")
    st.write(df.head())

    # Select the feature and label from the dataframe
    features = st.selectbox("Select the feature (X)", df.columns)
    label = st.selectbox("Select the label (Y)", df.columns)

    # Ensure the feature and label are not the same
    if features != label:
        if st.button("Train Model"):
            # Split the dataset into train and test
            X = df[[features]]  # Feature should be a 2D array
            y = df[label]       # Label is a 1D array

            # Check if there are missing values in the selected columns
            if X.isnull().sum().any() or y.isnull().sum() > 0:
                st.error("The dataset contains missing values. Please clean the data.")
            else:
                # Split into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Create a Linear Regression model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Display results
                st.subheader("Model Results")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
                st.write(f"R-squared: {r2_score(y_test, y_pred)}")

                # Plot the regression line
                plt.figure(figsize=(10, 6))
                plt.scatter(X_test, y_test, color="blue", label="Actual")
                plt.plot(X_test, y_pred, color="red", label="Predicted")
                plt.xlabel(features)
                plt.ylabel(label)
                plt.title(f"{features} vs {label}")
                plt.legend()
                st.pyplot(plt)
    else:
        st.error("Feature and label cannot be the same. Please select different columns.")

else:
    st.warning("Please upload a CSV file to get started.")

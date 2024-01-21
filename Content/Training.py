import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import time, re

def train_and_visualize_model(epochs, hidden_layers, X, y, is_regression):
    # Create MLP model
    if is_regression:
        model = MLPRegressor(hidden_layer_sizes=hidden_layers,max_iter=epochs, random_state=int(time.time()))
    else:
        model = MLPClassifier(hidden_layer_sizes=hidden_layers,max_iter=epochs, random_state=int(time.time()))

    figure = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if is_regression:
        plt.scatter(X, y, label='Original Data', color='blue')
        plt.title('Regression - Before Training')
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', marker='o', label='Original Data')
        plt.title('Classification - Before Training')

    # Train the model
    model.fit(X, y)

    # Visualization after training
    if is_regression:
        y_pred = model.predict(X)
        plt.subplot(1, 2, 2)
        plt.scatter(X, y, label='Original Data', color='blue')
        plt.plot(X, y_pred, label='Model Prediction', color='red', linewidth=2)
        plt.title('Regression - After Training')
    else:
        plt.subplot(1, 2, 2)
        preds = model.predict(X)==y
        colours = ["g" if preds[i] else "r" for i in range(len(preds))]
        plt.scatter(X[:, 0], X[:, 1], c=colours, edgecolors='k', marker='o', label='Model Prediction')
        plt.title('Classification - After Training')

    plt.legend()
    plt.tight_layout()
    st.pyplot(figure)

    # Display performance metrics
    st.write("Performance Metrics:")
    if is_regression:
        st.write(f"Mean Squared Error: {mean_squared_error(y, y_pred):.2f}")
    else:
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")

def draw(st, is_regression, X,y):
    st.title("Model Training and Visualization")
    st.write("The problems we'll deal with in this section have exactly 2 inputs and 1 output. Your goal is to produce the best results on the selected dataset")
    collect_numbers = lambda x : [int(i) for i in re.split("[^0-9]", x) if i != ""]
    input_text = st.text_input("What hidden layer architecture do you want to use = [2,2]")
    if(input_text):
        hidden_layers = collect_numbers(input_text)
    else:
        hidden_layers = [2,2]

    st.write("Epochs are the number of iterations the model goes over the data trying to learn from it.")
    epochs = st.slider("Epochs",min_value=200,max_value=1000)
    button = st.button("Train Model")
    # Button to train and visualize the model
    if button:
        train_and_visualize_model(epochs, hidden_layers, X, y, is_regression)


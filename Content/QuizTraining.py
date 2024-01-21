import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time, re
import numpy as np
from sklearn.model_selection import train_test_split
from Utils.DrawNN import DrawNN

SEED = 53
SEED_1 = 32
SEED_2 = 512

def generate_data():

    X1, y1 = make_classification(n_samples=2000, n_features=2, n_informative=2, n_redundant=0, class_sep=2, random_state=SEED_1)
    X2, y2 = make_classification(n_samples=2000, n_features=2, n_informative=2, n_redundant=0, class_sep=1, random_state=SEED_2)
    X3, y3 = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, class_sep=0, random_state=SEED_2)
    X = np.concatenate([X1,X2,X3])
    y = np.concatenate([y1,y2,y3])

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.8, random_state=SEED)
    np.random.seed(SEED)
    val = 1
    noise = np.random.uniform(-val,val, X_test.shape)
    X_test +=noise
    return X_train, X_test, y_train, y_test

def plot_data(X, y, Title):
    figure = plt.figure(figsize=(4, 3))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', marker='o')
    plt.title(Title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    return figure


def train_and_visualize_model(epochs, hidden_layers, X_train, y_train, X_test, y_test):
    # Create MLP model
    model = MLPClassifier(hidden_layer_sizes=hidden_layers,max_iter=epochs, random_state=SEED)

    figure = plt.figure(figsize=(12, 6))

    # Train the model
    model.fit(X_train, y_train)

    ytrain_pred = model.predict(X_train)
    ytest_pred = model.predict(X_test)

    plt.subplot(1, 2, 1)
    preds = ytrain_pred==y_train
    colours = ["g" if preds[i] else "r" for i in range(len(preds))]
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colours, edgecolors='k', marker='o')
    plt.title('Classification - On Training Data')


    plt.subplot(1, 2, 2)
    preds = ytest_pred==y_test
    colours = ["g" if preds[i] else "r" for i in range(len(preds))]
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colours, edgecolors='k', marker='o', label='Model Prediction')
    plt.title('Classification - On Test Data')

    plt.legend()
    plt.tight_layout()
    st.pyplot(figure)

    # Display performance metrics
    st.header("Performance Metrics:")
    st.write(f"Training Accuracy: {accuracy_score(y_train, ytrain_pred):.2f}")
    st.write(f"Testing Accuracy: {accuracy_score(y_test, ytest_pred):.2f}")

def draw(st):
    st.write("The problem here again only has 2 inputs and 1 output. The architecture input below only takes values for the hidden layers")
    collect_numbers = lambda x : [min(32,int(i)) for i in re.split("[^0-9]", x) if i != ""]
    input_text = st.text_input("What hidden layer architecture do you want to use = [2,2]")
    if(input_text):
        hidden_layers = collect_numbers(input_text)
    else:
        hidden_layers = [2,2]
    
    Architecture = [2, *hidden_layers, 1]
    if(sum(Architecture)<64):
        network = DrawNN(Architecture)
        figure = network.draw()
        st.pyplot(figure,use_container_width=False)
    else:
        st.write("Architecture too large to plot")

    X_train, X_test, y_train, y_test = generate_data()
    st.pyplot(plot_data(X_train, y_train, "Training Data visualization"))
    st.pyplot(plot_data(X_test, y_test, "Testing Data visualization"))

    st.write("Epochs are the number of iterations the model goes over the data trying to learn from it.")
    epochs = st.slider("Epochs",min_value=50,max_value=1000)
    button = st.button("Train Model")

    if button:
        train_and_visualize_model(epochs, hidden_layers,X_train,y_train,X_test,y_test)


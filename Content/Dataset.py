import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
import numpy as np
import time

def generate_data(dataset_type,random_seed, separation):
    np.random.seed(random_seed)

    if dataset_type == 'regression':
        X, y = make_regression(n_samples=np.random.randint(100,500), n_features=1, noise=np.random.randint(1,10))
        is_regression = True
    elif dataset_type == 'classification':
        if(separation is None):
            X, y = make_classification(n_samples=np.random.randint(100,500), n_features=2, n_informative=2, n_redundant=0,class_sep=np.random.randint(0,2))
        else:
            X, y = make_classification(n_samples=np.random.randint(100,500), n_features=2, n_informative=2, n_redundant=0,class_sep=separation)
        is_regression = False
    else:
        raise ValueError("Invalid dataset type")

    return X, y, is_regression

def plot_data(X, y, dataset_type):
    figure = plt.figure(figsize=(4, 3))
    if dataset_type == 'regression':
        plt.scatter(X, y, label='Generated Regression Data', color='blue')
    elif dataset_type == 'classification':
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', marker='o', label='Generated Classification Data')
    plt.title(f"Generated {dataset_type.capitalize()} Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2" if dataset_type == 'classification' else "Target")
    plt.legend()
    plt.grid(True)
    return figure

def draw(st, separation=None, seed=None):
    st.header("Dataset Generator")
    st.write("Now let's look at some dataset options, as we explained there are two types of problems we can look at")
    dataset_type = st.selectbox("Select dataset type", ['regression', 'classification'])

    button = st.button("Generate Data")
    # Button to generate data
    if button:
        if(dataset_type=="regression"):
            st.write("Regression is a machine learning problem where the target is continous data points where the goal is to estimate a continous value to input data")
        else:
            st.write("Classification is a machine learning problem where the target is discrete data points where the goal is to assign predefined categories to input data")
        st.write()
        if(seed is not None):
            random_seed = seed
        else:
            random_seed = int(time.time())
        X, y, is_regression = generate_data(dataset_type,random_seed, separation)
        figure = plot_data(X, y, dataset_type)
        st.pyplot(figure)
        return  X, y, is_regression
    return generate_data(dataset_type, int(time.time()), separation )


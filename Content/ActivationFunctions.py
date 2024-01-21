import numpy as np
import matplotlib.pyplot as plt

def plot_function(func_type, values):
    
    x = np.linspace(values[0], values[1], 400)  # Generate x values

    if func_type == 'constant':
        y = np.full_like(x, 5)  # Constant function
    elif func_type == 'linear':
        y = 2 * x + 3  # Linear function
    elif func_type == 'absolute':
        y = np.abs(x)  # Absolute value function
    elif func_type == 'quadratic':
        y = x**2  # Quadratic function
    elif func_type == 'root':
        y = np.sqrt(np.abs(x))  # Square root function
    elif func_type == 'cubic':
        y = x**3  # Cubic function
    elif func_type == 'logarithmic':
        y = np.log(np.abs(x) + 1)  # Logarithmic function
    elif func_type == 'exponential':
        y = np.exp(x)  # Exponential function
    elif func_type == 'sin':
        y = np.sin(x)  # Sine function
    elif func_type == 'cos':
        y = np.cos(x)  # Cosine function
    elif func_type == 'tan':
        y = np.tan(x)  # Tangent function
    else:
        raise ValueError("Invalid function type")

    # Plotting
    figure = plt.figure(figsize=(4, 3))
    plt.plot(x, y, label=f"{func_type} function")
    plt.title(f"Simulated {func_type} Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    return figure

def draw(st):
    st.header("Activation Functions")
    st.write("In this section let's look at some of the activation functions we explained")

    values = st.slider( 'Select a range of values for the X-axis',-200.0, 200.0, (-50.0, 50.0))

    # Dropdown menu for function selection
    func_type = st.selectbox("Select a function type", ['constant', 'linear', 'absolute', 'quadratic', 'root',
                                                        'cubic', 'logarithmic', 'exponential', 'sin', 'cos', 'tan'])

    # Call the plot_function function based on the selected option
    figure = plot_function(func_type, values)
    st.pyplot(figure,use_container_width=False)

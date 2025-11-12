import matplotlib.pyplot as plt

def plot_normalized(series1, series2, label1='A', label2='B'):
    s1 = series1 / series1.iloc[0]
    s2 = series2 / series2.iloc[0]

    plt.figure(figsize=(12,6))
    plt.plot(s1, label=label1, color="darkblue")
    plt.plot(s2, label=label2, color="darkred")
    plt.title("Series Normalizadas")
    plt.legend()
    plt.grid(True)
    plt.show()
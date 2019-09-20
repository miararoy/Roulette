import pandas as pd
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt


from bliz import RegressionBuilder

boston = load_boston()
boston_df = pd.DataFrame(data=boston.data, columns=boston.feature_names)

def min_max_norm(y):
    return (y - y.min()) / (y.max() - y.min())

boston_df["target_norm"] = min_max_norm(boston.target)

def main():
    builder = RegressionBuilder("./", train_data=boston_df)
    builder.build(n_experiments=100, target="target_norm")
    results = builder.evaluate()
    print(results)
    plot = builder.get_plot(title={k: round(v,2) for k,v in results.items()})
    plt.show()

if __name__ == "__main__":
    main()    
from dtreeviz.trees import dtreeviz
import matplotlib.pyplot as plt

def visualize_decision_tree(model, X, y, feature_names, target_names):
    viz = dtreeviz(
        model.estimators_[0, 0],
        X,
        y,
        target_name="Target",
        feature_names=feature_names,
        class_names=target_names,
    )
    viz.save("tree_visualization.svg")
    plt.show()

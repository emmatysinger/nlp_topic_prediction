import matplotlib.pyplot as plt
import numpy as np

def explain_prediction(sample, clf, feature_names):
    node_indicator = clf.decision_path(sample)
    end_leaf = clf.apply(sample)
    sample_id = 0

    # get node indices of path traversed by sample
    node_indices = np.where(node_indicator.toarray()[sample_id] != 0)[0]

    feature_importances = []
    for node_index in node_indices:
        if end_leaf[sample_id] == node_index:
            continue
        feature = clf.tree_.feature[node_index]
        left_child = clf.tree_.children_left[node_index]
        right_child = clf.tree_.children_right[node_index]

        # importance of the feature at the current node as the decrease in the measure of disorder (how many positive versus negative samples at the node).
        # subtract the weighted average measure of disorder of the child nodes from the measure of disorder of the current node.
        importance = clf.tree_.impurity[node_index] - (
            (clf.tree_.n_node_samples[left_child] * clf.tree_.impurity[left_child] +
             clf.tree_.n_node_samples[right_child] * clf.tree_.impurity[right_child]) /
            clf.tree_.n_node_samples[node_index]
        )
        feature_importances.append((feature_names[feature], importance))

    feature_importances.sort(key=lambda x: x[1], reverse=True)
    return feature_importances

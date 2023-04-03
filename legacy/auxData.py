import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def Load_Data(a, b, dataset_list_train, dataset_list_test):
    dataset_train = pd.concat([dataset_list_train[a-1], dataset_list_train[b-1]])
    dataset_test = pd.concat([dataset_list_test[a-1], dataset_list_test[b-1]])
    label = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    n_obj = len(dataset_train.values)
    training_features = np.zeros((n_obj, 2))
    training_labels = np.zeros(n_obj)
    for i in range(n_obj):
        training_features[i][0] = dataset_train.values[i][0]
        training_features[i][1] = dataset_train.values[i][1]
        #training_features[i][2] = dataset_train.values[i][2]
        #training_features[i][3] = dataset_train.values[i][3]
        if(dataset_train.values[i][4]==label[a-1]):
             training_labels[i] = 1
        if(dataset_train.values[i][4]==label[b-1]):
             training_labels[i] = -1

    n_obj_test = len(dataset_test.values)
    test_features = np.zeros((n_obj_test, 2))
    test_labels = np.zeros(n_obj_test)
    for i in range(n_obj_test):
        test_features[i][0] = dataset_test.values[i][0]
        test_features[i][1] = dataset_test.values[i][1]
        #test_features[i][2] = dataset_test.values[i][2]
        #test_features[i][3] = dataset_test.values[i][3]
        if(dataset_test.values[i][4]==label[a-1]):
             test_labels[i] = 1
        if(dataset_test.values[i][4]==label[b-1]):
             test_labels[i] = -1
    print(dataset_train.value_counts("class"))
    print(dataset_test.value_counts("class"))
    return [training_features, training_labels, test_features, test_labels]

def Load_DataFull(a, b, dataset_list_train, dataset_list_test):
    dataset_train = pd.concat([dataset_list_train[a-1], dataset_list_train[b-1]])
    dataset_test = pd.concat([dataset_list_test[a-1], dataset_list_test[b-1]])
    label = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    n_obj = len(dataset_train.values)
    training_features = np.zeros((n_obj, 4))
    training_labels = np.zeros(n_obj)
    for i in range(n_obj):
        training_features[i][0] = dataset_train.values[i][0]
        training_features[i][1] = dataset_train.values[i][1]
        training_features[i][2] = dataset_train.values[i][2]
        training_features[i][3] = dataset_train.values[i][3]
        if(dataset_train.values[i][4]==label[a-1]):
             training_labels[i] = 1
        if(dataset_train.values[i][4]==label[b-1]):
             training_labels[i] = -1

    n_obj_test = len(dataset_test.values)
    test_features = np.zeros((n_obj_test, 4))
    test_labels = np.zeros(n_obj_test)
    for i in range(n_obj_test):
        test_features[i][0] = dataset_test.values[i][0]
        test_features[i][1] = dataset_test.values[i][1]
        test_features[i][2] = dataset_test.values[i][2]
        test_features[i][3] = dataset_test.values[i][3]
        if(dataset_test.values[i][4]==label[a-1]):
             test_labels[i] = 1
        if(dataset_test.values[i][4]==label[b-1]):
             test_labels[i] = -1
    print(dataset_train.value_counts("class"))
    print(dataset_test.value_counts("class"))
    return [training_features, training_labels, test_features, test_labels]

def Load_DataMultiClass(dataset_train, dataset_test):
    label = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    n_obj = len(dataset_train.values)
    training_features = np.zeros((n_obj, 4))
    training_labels = np.zeros(n_obj)
    for i in range(n_obj):
        training_features[i][0] = dataset_train.values[i][0]
        training_features[i][1] = dataset_train.values[i][1]
        training_features[i][2] = dataset_train.values[i][2]
        training_features[i][3] = dataset_train.values[i][3]
        if(dataset_train.values[i][4]==label[0]):
             training_labels[i] = 0
        if(dataset_train.values[i][4]==label[1]):
             training_labels[i] = 1
        if(dataset_train.values[i][4]==label[2]):
             training_labels[i] = 2

    n_obj_test = len(dataset_test.values)
    test_features = np.zeros((n_obj_test, 4))
    test_labels = np.zeros(n_obj_test)
    for i in range(n_obj_test):
        test_features[i][0] = dataset_test.values[i][0]
        test_features[i][1] = dataset_test.values[i][1]
        test_features[i][2] = dataset_test.values[i][2]
        test_features[i][3] = dataset_test.values[i][3]
        if(dataset_test.values[i][4]==label[0]):
             test_labels[i] = 0
        if(dataset_test.values[i][4]==label[1]):
             test_labels[i] = 1
        if(dataset_test.values[i][4]==label[2]):
             test_labels[i] = 2
    print(dataset_train.value_counts("class"))
    print(dataset_test.value_counts("class"))
    return [training_features, training_labels, test_features, test_labels]



def plot_sampled_data_sepal(training_features, training_labels, test_features, test_labels):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,6))

    for feature, label in zip(training_features, training_labels):
        marker = 'o' 
        color = 'tab:green' if label == -1 else 'tab:blue'
        plt.scatter(feature[0], feature[1], marker=marker, s=100, color=color)
    
    for feature, label in zip(test_features, test_labels):
        marker = 's' 
        color = 'tab:green' if label == -1 else 'tab:blue'
        plt.scatter(feature[0], feature[1], marker=marker, s=100, color=color)

    legend_elements = [
        Line2D([0], [0], marker='o', c='w', mfc='tab:blue', label='Iris-setosa', ms=15),
        Line2D([0], [0], marker='o', c='w', mfc='tab:green', label='Iris-versicolor', ms=15),

    ]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1, 0.6))
    
    plt.title('Training & test data - Sepal')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')

def plot_sampled_data_petal(training_features, training_labels, test_features, test_labels):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12,6))

    for feature, label in zip(training_features, training_labels):
        marker = 'o' 
        color = 'tab:green' if label == -1 else 'tab:blue'
        plt.scatter(feature[2], feature[3], marker=marker, s=100, color=color)
    
    for feature, label in zip(test_features, test_labels):
        marker = 's' 
        color = 'tab:green' if label == -1 else 'tab:blue'
        plt.scatter(feature[2], feature[3], marker=marker, s=100, color=color)

    legend_elements = [
        Line2D([0], [0], marker='o', c='w', mfc='tab:blue', label='Iris-setosa', ms=15),
        Line2D([0], [0], marker='o', c='w', mfc='tab:green', label='Iris-versicolor', ms=15),

    ]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1, 0.6))
    
    plt.title('Training & test data - Petal')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')

def plot_predicted(a,b, training_features, training_labels, test_features, test_labels, predicted):
    from matplotlib.lines import Line2D
    wrong_predicted = 0
    plt.figure(figsize=(12, 6))
    label_train = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    label_test = ['predict Iris-setosa', 'predict Iris-versicolor', 'predict Iris-virginica']
    for feature, label in zip(training_features, training_labels):
        marker = 'o' 
        color = 'tab:green' if label == -1 else 'tab:blue'
        plt.scatter(feature[0], feature[1], marker=marker, s=100, color=color)
    
    for feature, label, pred in zip(test_features, test_labels, predicted):
        marker = 's' 
        color = 'tab:green' if pred == -1 else 'tab:blue'
        if label != pred:  # mark wrongly classified
            wrong_predicted+=1
            plt.scatter(feature[0], feature[1], marker='o', s=500, linewidths=2.5,
                        facecolor='none', edgecolor='tab:red')

        plt.scatter(feature[0], feature[1], marker=marker, s=100, color=color)
    
    legend_elements = [
        Line2D([0], [0], marker='o', c='w', mfc='tab:green', label=label_train[b-1], ms=15),
        Line2D([0], [0], marker='o', c='w', mfc='tab:blue', label=label_train[a-1], ms=15),
        Line2D([0], [0], marker='s', c='w', mfc='tab:green', label=label_test[b-1], ms=10),
        Line2D([0], [0], marker='s', c='w', mfc='tab:blue', label=label_test[a-1], ms=10),
        Line2D([0], [0], marker='o', c='w', mfc='none', mec='tab:red', label='wrongly classified', mew=2, ms=15)
    ]

    plt.legend(loc= "lower left", handles=legend_elements, bbox_to_anchor=(1.0, 0.4))
    plt.title('Training & test data - Sepal')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    print("Wrong predicted:", wrong_predicted)
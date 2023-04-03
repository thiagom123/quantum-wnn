import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np



def get_binary_dataset(X_all, Y_all, class_0, class_1, shuffle= True):
    #Get the dataset of the classes
    X_0 = np.array(X_all[Y_all == class_0])
    X_1 = np.array(X_all[Y_all == class_1])
    Y_0 = np.array(Y_all[Y_all == class_0])
    Y_1 = np.array(Y_all[Y_all == class_1])
    X = np.concatenate((X_0, X_1), axis=0)
    Y = np.concatenate((Y_0, Y_1), axis=0)

    # Randomly shuffle data and labels 
    rnd = np.random.RandomState(42)
    if(shuffle): idx = rnd.permutation(len(Y))
    X, Y = X[idx], Y[idx]
    # Scale to the range (0, +1)
    y01 = (Y - min(Y))
    y01 = y01 // max(y01)
    y = 2 * y01 -1
    return X, y

def plot_sampled_features(features, labels, axis_x_name='x' , axis_y_name = 'y', class_name_minus='-1', class_name_plus='1' ):
    '''
    training_features: Features dos dados de treinamento
    '''
    plt.figure(figsize=(8,8))

    for feature, label in zip(features, labels):
        marker = 'o' 
        color = 'tab:green' if label == -1 else 'tab:blue'
        plt.scatter(feature[0], feature[1], marker=marker, s=100, color=color)
    
    legend_elements = [
        Line2D([0], [0], marker='o', c='w', mfc='tab:blue', label=class_name_plus, ms=15),
        Line2D([0], [0], marker='o', c='w', mfc='tab:green', label=class_name_minus, ms=15),

    ]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1, 0.6))
    
    plt.title('Dataset')
    plt.xlabel(axis_x_name)
    plt.ylabel(axis_y_name)
    #plt.savefig("iris.jpg", format="jpg", dpi=600)

def plot_predicted(training_features, training_labels, test_features, test_labels, predicted,
    axis_x_name='x' , axis_y_name = 'y', class_name_minus='-1', class_name_plus='1'):
    wrong_predicted = 0
    plt.figure(figsize=(12, 6))
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
        Line2D([0], [0], marker='o', c='w', mfc='tab:green', label=class_name_minus, ms=15),
        Line2D([0], [0], marker='o', c='w', mfc='tab:blue', label=class_name_plus, ms=15),
        Line2D([0], [0], marker='s', c='w', mfc='tab:green', label=class_name_minus, ms=10),
        Line2D([0], [0], marker='s', c='w', mfc='tab:blue', label=class_name_plus, ms=10),
        Line2D([0], [0], marker='o', c='w', mfc='none', mec='tab:red', label='wrongly classified', mew=2, ms=15)
    ]

    plt.legend(loc= "lower left", handles=legend_elements, bbox_to_anchor=(1.0, 0.4))
    plt.title('Training & test data')
    plt.xlabel(axis_x_name)
    plt.ylabel(axis_y_name)
    print("Wrong predicted:", wrong_predicted)




def plot_sampled_data_sepal(training_features, training_labels, test_features, test_labels):
    '''
    training_features: Features dos dados de treinamento
    '''
    plt.figure(figsize=(8,8))

    for feature, label in zip(training_features, training_labels):
        marker = 'o' 
        color = 'tab:green' if label == -1 else 'tab:blue'
        plt.scatter(feature[0], feature[1], marker=marker, s=100, color=color)
    
    for feature, label in zip(test_features, test_labels):
        marker = 'o' 
        color = 'tab:green' if label == -1 else 'tab:blue'
        plt.scatter(feature[0], feature[1], marker=marker, s=100, color=color)

    legend_elements = [
        Line2D([0], [0], marker='o', c='w', mfc='tab:blue', label='Iris-setosa', ms=15),
        Line2D([0], [0], marker='o', c='w', mfc='tab:green', label='Iris-versicolor', ms=15),

    ]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1, 0.6))
    
    plt.title('Iris dataset- Sepal')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    #plt.savefig("iris.jpg", format="jpg", dpi=600)

def plot_sampled_data_petal(training_features, training_labels, test_features, test_labels):
    

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
    
    plt.title('Training & test data - Petal')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')


def plot_area(a,b, test_features, predicted):
    wrong_predicted = 0
    plt.figure(figsize=(12, 6))
    label_train = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    label_test = ['predict Iris-setosa', 'predict Iris-versicolor', 'predict Iris-virginica']
    
    for feature, pred in zip(test_features, predicted):
        marker = 's' 
        color = 'tab:green' if pred == -1 else 'tab:blue'
        plt.scatter(feature[0], feature[1], marker=marker, s=100, color=color)
    
    legend_elements = [

        Line2D([0], [0], marker='s', c='w', mfc='tab:green', label=label_test[b-1], ms=10),
        Line2D([0], [0], marker='s', c='w', mfc='tab:blue', label=label_test[a-1], ms=10),

    ]

    plt.legend(loc= "lower left", handles=legend_elements, bbox_to_anchor=(1.0, 0.4))
    plt.title('Training & test data - Sepal')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')

def plot_bloch_sphere(X, y, fixed_points=[]):
    '''
    training_features: Features dos dados de treinamento
    '''
    plt.figure(figsize=(8,8))

    for feature, label in zip(X, y):
        marker = 'o' 
        color = 'tab:green' if label == -1 else 'tab:blue'
        plt.scatter(feature[0], feature[1], marker=marker, s=100, color=color)
    
    for i in range(len(fixed_points)):
        point = fixed_points[i]
        if(i==0): label = f"00"
        if(i==1): label = f"01"
        if(i==2): label = f"10"
        if(i==3): label = f"11"
        marker = 's'
        color = 'tab:red'
        plt.scatter(point[0], point[1], marker=marker, s=100, color=color)
        #para os outros pontos usei xytext = 10,10
        plt.annotate(label, (point[0], point[1]), xytext=(10,10), textcoords = 'offset points',ha='center' )
    #for point in fixed_points:
    #    color = 'tab:red'
    #    
    #    plt.scatter(point[0], point[1], marker=marker, s=100, color=color)
        
    legend_elements = [
        Line2D([0], [0], marker='o', c='w', mfc='tab:blue', label='+1', ms=15),
        Line2D([0], [0], marker='o', c='w', mfc='tab:green', label='-1', ms=15),
        Line2D([0], [0], marker='s', c='w', mfc='tab:red', label='Base Points', ms=15)
        #Line2D([0], [0], marker='<', c='w', mfc='tab:red', label='00', ms=15),
        #Line2D([0], [0], marker='v', c='w', mfc='tab:red', label='01', ms=15),
        #Line2D([0], [0], marker='>', c='w', mfc='tab:red', label='10', ms=15),
        #Line2D([0], [0], marker='^', c='w', mfc='tab:red', label='11', ms=15),

    ]
    angle = np.linspace( 0 , 2 * np.pi , 100)
    x = np.cos(angle)
    y = np.sin(angle)
    plt.plot(x,y, linestyle = '--', color='k')
    x2 = np.linspace(-1, 1, 100 )
    y2 = np.zeros(100)
    plt.plot(x2, y2, linestyle = '--', color ='k')

    #bbox_to_anchor=(1, 0.6)
    #plt.Circle((0,0), radius=1, fill=False, linestyle = '--')
    plt.legend(handles=legend_elements )
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.title('Bloch Sphere Representation')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.savefig("figure.jpg", format="jpg", dpi=300)
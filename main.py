import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn import metrics

def confMatrix(truth: list, predictions: list, log = False):
    '''
    predictions, truth as lists of values.
    Class names and quantity calculated automatically from unique values across both lists.
    Set log to True for a logarithmic colour map - good for high performing classifiers.
    '''
   
    classnames = sorted(list(set(list(truth) + list(predictions)))) #Produces sorted list of unique classnames
    if log: norm = matplotlib.colors.LogNorm()
    else:   norm = matplotlib.colors.Normalize()
    matrix = pd.DataFrame(metrics.confusion_matrix(truth, predictions),
                      index = classnames,
                      columns = classnames)
    
    fig = plt.figure(figsize = (10,8))
    heatmap = sns.heatmap(matrix, annot = True, fmt = "d",
                          cmap = "rocket_r", square = True,
                          vmin = 0.001,# Zero value catcher for LogNorm
                          norm = norm)
 
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                             rotation = 0, ha = 'right', fontsize = 14)
   
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                             rotation = 0, ha = 'right', fontsize = 14)
  
    # Corrects ylim issues under matplotlib 3.1.1. Comment this out if not required.
    if matplotlib.__version__ == '3.1.1':
        plt.ylim(len(classnames),0)
        print("Note: Matplotlib version 3.1.1 has some compatability issues with Seaborn.")
   
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

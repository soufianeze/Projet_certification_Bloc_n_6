import matplotlib.pyplot as plt
import numpy
import itertools

def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
  
    '''
  Cette fonction permet de génerer un tableau de confusion avec les caractéristiques souhaitées
  '''

  
  
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = numpy.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = 'd' 
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",
    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

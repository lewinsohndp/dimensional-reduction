import numpy as np 

"""Class for calculating contrastive loss"""

class ContrastiveLoss():
    def __init__(self, margin=1):
        self.margin = margin
    
    def get_loss(self, coords, labels):
        """
        Takes in coordinates and binary label for coordinate and returns loss value

        Parameters
        ------------
        coords : 2d array
            of arrays representing 2d coordinates
        labels : array

        Returns
        ---------
        loss : float
        """
        sum = 0.0

        #go through all pairs
        for i in range(len(coords)):
            for x in range(i + 1, len(coords)):
                #0 if they are similar, 1 if they are not
                if labels[i] == labels[x]: Y = 0
                else: Y = 1

                #get euclidean distance between points
                dist = euclidean(coords[i], coords[x])
                
                #what should we define m as ?
                sum += ((1-Y) * .5 * (dist**2)) + (Y * .5 * (max(0, self.margin -dist)**2))

        return sum

def euclidean(x, y):
    """Standard euclidean distance.
    Adapted from UMAP implementation
    """
    result = 0.0
    for i in range(len(x)):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)
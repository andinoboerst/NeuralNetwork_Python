import numpy as np


def sample_prop(self, x: np.array([]), y: np.array([]), update: bool=True) -> list:
    '''
    Backpropagation after every sample
    '''
    updated_weights = [] # This will be 2 dimensional! Each row represent the weights for each layer.


    if update:
        self.apply_prop(updated_weights)
    return updated_weights



def mini_prop(self, X: np.array([[]]), Y: np.array([[]]), num_batches: int=5) -> None:
    '''
    Backpropagation after a mini-batch
    batch_ratio is the percentage of samples that should be used for each mini batch
    '''
    if type(num_batches) != int or num_batches < 1:
        raise ValueError("Given number of batches is not permitted.")
    entries_per_batch = [len(X)//num_batches for i in range(num_batches)]
    for i in range(len(X)%num_batches):
        entries_per_batch[i] += 1
    # Divide the sets and call batch_prop() on them
    for batch_entries in range(num_batches):
        updated_weights = self.batch_prop(X.splice(0,batch_entries), Y.splice(0, batch_entries), update=False)
        self.apply_prop(updated_weights)



def batch_prop(self, X: np.array([[]]), Y: np.array([[]]), update: bool=True) -> list:
    '''
    Backpropagation for the entire batch
    '''
    # call sample_prop() on each sample and average the results
    updated_weights = []
    for x, y in zip(X,Y):
        updated_weights.append(self.sample_prop(x,y,update=False)) # This will be 3 dimensional!!
    #final_updated_weights = np.mean(updated_weights, axis=0) # Not sure this will work
    final_updated_weights = updated_weights[0]
    for entry in updated_weights[1:]:
        for index, l in enumerate(entry):
            final_updated_weights[index] = [sum(i) for i in zip(final_updated_weights[index],l)]
    for i, entry in enumerate(final_updated_weights):
        for j, item in enumerate(entry):
            final_updated_weights[i][j] = item / len(updated_weights)
    if update:
        self.apply_prop(final_updated_weights)
    return final_updated_weights

def apply_prop(self, updated_weights: np.array([[]])):
    '''
    Apply the defined weights and biases to the neural network
    '''



def mean_squared_error(self, y_pred: np.array([]), y_real: np.array([])):
    return ((y_pred-y_real)**2).sum() / (2*len(y_pred))
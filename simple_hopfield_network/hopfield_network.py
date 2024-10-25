import numpy as np

def step_activation(threshold):
    def activation(x):
        return np.where(x > threshold, 1, 0)
    return activation

class HopfieldNetwork:
    def __init__(self, pattern_size, activation=step_activation(0)):
        self.pattern_size = pattern_size
        self.weights = np.zeros((pattern_size, pattern_size))
        self.activation = activation
        self.istrained = False
        
    def predict(self, x):
        for _ in range(100):
            x = self.activation(self.weights @ x)
        return x
    
    def train(self, patterns):
        if isinstance(patterns, np.ndarray) and len(patterns.shape) == 1:
            patterns = patterns[np.newaxis, :]
        if isinstance(patterns, np.ndarray) and patterns.shape[0] != 1:
            raise NotImplementedError("Only a single pattern supported for now")
        
        self.weights = patterns.T @ patterns
        np.fill_diagonal(self.weights, 0)

if __name__=='__main__':
    pattern_size = 4
    pattern = np.array([1, 0, 1, 0])
    print(f"Pattern: {pattern}")
    hopfield = HopfieldNetwork(pattern_size)
    hopfield.train(pattern)
    all_possible_patterns = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1]
    ]
    for input_pattern in all_possible_patterns:
        output_pattern = hopfield.predict(input_pattern)
        print(f"Input: {input_pattern} -> Output: {output_pattern}")
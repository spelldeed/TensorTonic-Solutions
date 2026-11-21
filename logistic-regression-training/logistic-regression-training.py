import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))


# def _differentiate()

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    c,r = X.shape
    w = np.zeros((r,))
    # print((X*w).shape)
    b = 0.0
    for i in range(0,steps):
        z = X@w + b
        p = _sigmoid(z)
        print(p.shape)
        L = (-1/c)*(np.sum((y*np.log(p)) + ((1-y)*np.log(1-p))))
        Lw = (1/c)*np.transpose(X)@(p-y)
        # print(Lw.shape)
        Lb = (1/c)*(np.sum(p-y))
        w-=(lr*Lw)
        b-=(lr*Lb)


    return w,b
        
        
        
        
    
    
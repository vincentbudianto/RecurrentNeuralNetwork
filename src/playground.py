import numpy as np

if __name__ == "__main__":
    result = np.ones((4, 3))
    multiplier = np.array([1, 2, 3, 2])
    print(result)   
    print(multiplier)

    tempo = np.matmul(multiplier, result)

    temporary = np.array([1, 2, 3])
    print(tempo + temporary )
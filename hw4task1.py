import numpy as np
import matplotlib.pyplot as plt

def normalize(matrix):
    # Normalize the columns of the matrix so that they sum to 1
    column_sums = matrix.sum(axis=0)
    return matrix / column_sums

def pagerank(A, alpha=0.85, threshold=1e-6, max_iterations=150):
    # Normalize the adjacency matrix
    M = normalize(A)

    # PageRank Iteration
    pages = 6
    r = np.ones(pages) / pages  # Initialize the rank vector
    s = np.ones(pages) / pages  # Damping vector

    history = [r]  # Store rank vectors at each iteration

    for _ in range(max_iterations):
        rnew = alpha * M @ r + (1 - alpha) * s
        if np.all(np.abs(r - rnew) < threshold):
            break
        r = rnew
        history.append(r)
    # Verify that matrix is normalized
    print("normalized matrix",M)
    return r, history

# Define the adjacency matrix A for the given web structure
A = np.array([[0, 1, 1, 0, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 1, 0],
              [0, 1, 0, 1, 0, 1],
              [0, 0, 0, 0, 1, 0]], dtype=float)

# Perform the PageRank computation
pagerank_result, history = pagerank(A)

# Analysis
print("PageRank Results:", pagerank_result)

# Visualization: Plot the rank of the pages after each iteration
count = len(history)
for i in range(A.shape[0]):
    rank_i = [rank[i] for rank in history]
    plt.plot(range(1, count + 1), rank_i, label=f'Page {i+1}')

plt.xlabel('Iteration')
plt.ylabel('Rank')
plt.title('Rank Convergence')
plt.legend()
plt.grid()
plt.show()

# Verification
# Find the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(normalize(A))

# Find the index of the largest eigenvalue
max_index = np.argmax(eigenvalues)

# Extract the largest eigenvalue and its associated eigenvector
largest_eigenvalue = eigenvalues[max_index]
largest_eigenvector = eigenvectors[:, max_index]
print("vector associated with largest eigen value:",largest_eigenvector)
print("Difference between PageRank and eigen vector:", np.abs(pagerank_result - largest_eigenvector))
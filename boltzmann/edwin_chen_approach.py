import numpy as np
from boltzmann.rbm import RBM

def start_edwin_chen():
    r = RBM(num_visible=6, num_hidden=2)

    training_data = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0]
        ]
    )

    r.train(training_data, max_epochs=5000)


    print("weights:")
    print(r.weights)

    # Given a new set of visible units, we can see what hidden units are activated.
    # A matrix with a single row that contains the states of the visible units. (We can also include more rows.)
    user = np.array([[0, 0, 0, 1, 1, 0]])
    print("visibles:")
    print(r.run_visible(user))

    # Given a set of hidden units, we can see what visible units are activated.
    # A matrix with a single row that contains the states of the hidden units. (We can also include more rows.)
    hidden_data = np.array([[1, 0]])
    # See what visible units are activated.
    r.run_hidden(hidden_data)

    # We can let the network run freely (aka, daydream).
    r.daydream(100)  # Daydream for 100 steps on a single initialization.
class Params():

    STEPS = 100000
    # ROLLING HORIZON (seq len 24, n evals 10, mut rate 0.3)
    SEQ_LEN = 5  # Length of the trajectory
    N_EVALS = 10  # Number of sequences to evaluate (1: only non-mutated one, N: N-1 mutated)
    MUT_RATE = 0.9  # Mutation rate (probability)
    SHIFT_BUFFER = True

    # ROLLING HORIZON
    # SEQ_LEN = 20  # Length of the trajectory
    # N_EVALS = 1  # Number of sequences to evaluate (1: only non-mutated one, N: N-1 mutated)
    # MUT_RATE = 0.3  # Mutation rate (probability)
    # SHIFT_BUFFER = True

    # BOOTSTRAPPING
    NUM_HEADS = 10
    #BERNOULLI = 0.5

    # RPF
    PRIOR_SCALE = 1.0

    #A2C
    ALR = 7e-4

    # ERROR CORRECTION
    ERROR_CORRECTION = False
    GHOST = 6
    PACMAN = 1
    EATEN_CELL = 5
    FRUIT = 0
    #UNEATEN_CELL = ?

    def __init__(self):
        pass

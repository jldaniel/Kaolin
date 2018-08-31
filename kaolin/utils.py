import numpy as np

# TODO: Test latin_hypercube
def latin_hypercube(bounds, n_samples, seed):
    """
    Calculate a latin hypercube design table
    :param bounds: The bounds for the factors, e.g. np.array([[0.0, 0.2], [0.5, 1.0]])
    :param n_samples: The number of samples for the design table
    :param seed: A random seed value to use
    :return: The latin hypercube design table
    """
    # Create the scaled set of samples between 0
    np.random.seed(seed)
    n_dim = bounds.shape[0]
    intervals = np.linspace(0.0, 1.0, n_samples + 1)

    # Fill in the points
    seed_points = np.random.rand(n_samples, n_dim)
    lower_intervals = intervals[:n_samples]
    upper_intervals = intervals[1:n_samples + 1]
    points = np.empty_like(seed_points)

    for j in range(n_dim):
        points[:, j] = seed_points[:, j] * (upper_intervals - lower_intervals) + lower_intervals

    # Generate the design table
    table = np.empty_like(points)
    for j in range(n_dim):
        order = np.random.permutation(range(n_samples))
        table[:, j] = points[order, j]

    # Scale the variable points
    scaled_table = np.empty_like(table)
    for j, bound in enumerate(bounds):
        lower_bound = bound[0]
        upper_bound = bound[1]
        scaled_table[:, j] = table[:, j] * (upper_bound - lower_bound) + lower_bound

    return scaled_table

# TODO: Test full_factorial
def full_factorial(bounds, levels):
    """
    Calculate a full factorial design table
    :param bounds: The bounds for the factors, e.g. np.array([[0.0, 0.2], [0.5, 1.0]])
    :param levels: The levels for each factor, e.g. [2, 3]
    :return: A full factorial design table
    """
    n_dim = bounds.shape[0]
    n_points = np.prod(levels)

    table = np.empty((n_points, n_dim))

    level_repeat = 1
    range_repeat = np.prod(levels)
    for i in range(n_dim):
        range_repeat //= levels[i]

        level = []
        for j in range(levels[i]):
            level += [j]*level_repeat

        rng = level*range_repeat
        level_repeat *= levels[i]
        table[:, i] = rng/np.max(rng)

        # Scale the variable points
        scaled_table = np.empty_like(table)
        for j, bound in enumerate(bounds):
            lower_bound = bound[0]
            upper_bound = bound[1]
            scaled_table[:, j] = table[:, j] * (upper_bound - lower_bound) + lower_bound

    return scaled_table


# TODO: Test mixed doe
def mixed_doe(n_points, bounds, seed=None):
    """
    Get the initial training points for the model
    :param n_points: Number of points to generate for the training set
    :param bounds: The design variable bounds
    :param seed: Random seed value
    :return: The initial design table for the model
    """

    n_dim = bounds.shape[0]

    # Ideally want around a 50/50 split between full factorial and lhc designs
    ff_points_max = np.floor(0.6*n_points)
    n_levels = 2
    ff_points = 0
    levels = None

    while True:
        levels_temp = [n_levels]*n_dim
        ff_points_temp = np.prod(levels_temp)

        if ff_points_temp < ff_points_max:
            ff_points = ff_points_temp
            levels = levels_temp
            n_levels += 1
        else:
            break

    lhc_points = int(n_points - ff_points)

    ff_designs = None
    lhc_designs = None
    if levels:
        ff_designs = full_factorial(bounds, levels)

    if lhc_points:
        lhc_designs = latin_hypercube(bounds, lhc_points, seed)

    if levels and lhc_points:
        designs = np.concatenate((ff_designs, lhc_designs), axis=0)
    elif levels and not lhc_points:
        designs = ff_designs
    else:
        designs = lhc_designs

    return designs
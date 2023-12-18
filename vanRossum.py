import math


def d(train_a, train_b, tau):
    """
    Compute the van Rossum metric for two spike trains.

    Args:
        train_a (list): The first spike train as a list of timestamps.
        train_b (list): The second spike train as a list of timestamps.
        tau (float): The time constant for the metric.

    Returns:
        float: The van Rossum distance between the two spike trains.
    """
    trains = [train_a, train_b]
    sq = [norm_train(trains[train_c], tau) for train_c in range(2)]
    return math.sqrt(sq[0] + sq[1] - corr(trains, tau))


def d_matrix(trains, tau):
    """
    Compute the van Rossum metric for multiple spike trains and return a matrix of distances.

    Args:
        trains (list of lists): List of spike trains as lists of timestamps.
        tau (float): The time constant for the metric.

    Returns:
        list of lists: The matrix of distances between spike trains.
    """
    trains_size = len(trains)
    fs = [[] for _ in range(trains_size)]
    d_matrix = [[0.0] * trains_size for _ in range(trains_size)]

    for i in range(trains_size):
        markage(fs[i], trains[i], tau)

    sq = [norm(fs[i]) for i in range(trains_size)]

    for i in range(trains_size):
        for j in range(i + 1, trains_size):
            this_d = math.sqrt(
                sq[i] + sq[j] - corr(trains[i], trains[j], fs[i], fs[j], tau)
            )
            d_matrix[i][j] = this_d
            d_matrix[j][i] = this_d

    return d_matrix


def d_matrix_exp(trains, tau):
    trains_size = len(trains)
    fs = [[] for _ in range(trains_size)]
    e_poss = [[] for _ in range(trains_size)]
    e_negs = [[] for _ in range(trains_size)]

    for i in range(trains_size):
        e_pos, e_neg = expage(trains[i], tau)
        e_poss[i] = e_pos
        e_negs[i] = e_neg

    for i in range(trains_size):
        markage(fs[i], e_poss[i], e_negs[i], trains[i], tau)

    sq = [norm(fs[i]) for i in range(trains_size)]

    for i in range(trains_size):
        for j in range(i + 1, trains_size):
            this_d = math.sqrt(
                sq[i]
                + sq[j]
                - corr(
                    trains[i],
                    trains[j],
                    fs[i],
                    fs[j],
                    e_poss[i],
                    e_poss[j],
                    e_negs[i],
                    e_negs[j],
                )
            )
            d_matrix[i][j] = this_d
            d_matrix[j][i] = this_d


def d_matrix_no_markage(d_matrix, trains, tau):
    trains_size = len(trains)
    sq = [norm(train, tau) for train in trains]

    for i in range(trains_size):
        for j in range(i + 1, trains_size):
            this_d = math.sqrt(sq[i] + sq[j] - corr(trains[i], trains[j], tau))
            d_matrix[i][j] = this_d
            d_matrix[j][i] = this_d


def d_matrix_no_markage_exp(d_matrix, trains, tau):
    trains_size = len(trains)
    e_poss = [[] for _ in range(trains_size)]
    e_negs = [[] for _ in range(trains_size)]

    for i in range(trains_size):
        e_pos, e_neg = expage(trains[i], tau)
        e_poss[i] = e_pos
        e_negs[i] = e_neg

    sq = [norm(train, e_poss[i], e_negs[i]) for i, train in enumerate(trains)]

    for i in range(trains_size):
        for j in range(i + 1, trains_size):
            this_d = math.sqrt(
                sq[i]
                + sq[j]
                - corr(trains[i], trains[j], e_poss[i], e_poss[j], e_negs[i], e_negs[j])
            )
            d_matrix[i][j] = this_d
            d_matrix[j][i] = this_d


def norm(fs):
    f_size = len(fs)

    if f_size == 0:
        return 0

    norm = f_size / 2.0

    for i in range(1, f_size):
        norm += fs[i]

    return norm


def norm_train(train, tau):
    train_size = len(train)

    if train_size == 0:
        return 0

    norm = train_size / 2.0

    for i in range(train_size):
        for j in range(i + 1, train_size):
            norm += math.exp((train[i] - train[j]) / tau)

    return norm


def norm_train_exp(train, e_pos, e_neg):
    train_size = len(train)

    if train_size == 0:
        return 0

    norm = train_size / 2.0

    for i in range(train_size):
        for j in range(i + 1, train_size):
            norm += e_pos[i] * e_neg[j]

    return norm


def corr(trains, tau):
    """
    Compute the correlation between two spike trains represented as lists of lists.

    Args:
        trains (list of list): List containing two spike trains as lists of timestamps.
        tau (float): The time constant for the correlation.

    Returns:
        float: The correlation value.
    """
    x = 0

    trains0_size = len(trains[0])
    trains1_size = len(trains[1])

    if trains0_size == 0 or trains1_size == 0:
        return 0

    for i in range(trains0_size):
        for j in range(trains1_size):
            x += math.exp(-abs(trains[0][i] - trains[1][j]) / tau)

    return x


def corr_train(train_a, train_b, tau):
    """
    Compute the correlation between two spike trains represented as lists.

    Args:
        train_a (list): The first spike train as a list of timestamps.
        train_b (list): The second spike train as a list of timestamps.
        tau (float): The time constant for the correlation.

    Returns:
        float: The correlation value.
    """
    x = 0

    train_a_size = len(train_a)
    train_b_size = len(train_b)

    if train_a_size == 0 or train_b_size == 0:
        return 0

    for i in range(train_a_size):
        for j in range(train_b_size):
            x += math.exp(-abs(train_a[i] - train_b[j]) / tau)

    return x


def corr_with_e(train_a, train_b, e_pos_a, e_pos_b, e_neg_a, e_neg_b):
    """
    Compute the correlation between two spike trains represented as lists with positive and negative exponential terms.

    Args:
        train_a (list): The first spike train as a list of timestamps.
        train_b (list): The second spike train as a list of timestamps.
        e_pos_a (list): Exponential positive values for train_a.
        e_pos_b (list): Exponential positive values for train_b.
        e_neg_a (list): Exponential negative values for train_a.
        e_neg_b (list): Exponential negative values for train_b.

    Returns:
        float: The correlation value.
    """
    x = 0

    train_a_size = len(train_a)
    train_b_size = len(train_b)

    if train_a_size == 0 or train_b_size == 0:
        return 0

    for i in range(train_a_size):
        for j in range(train_b_size):
            if train_a[i] > train_b[j]:
                x += e_pos_b[j] * e_neg_a[i]
            else:
                x += e_neg_b[j] * e_pos_a[i]

    return x


def corr_with_features(
    train_a, train_b, f_a, f_b, e_pos_a, e_pos_b, e_neg_a, e_neg_b, tau
):
    """
    Compute the correlation between two spike trains represented as lists with features and exponential terms.

    Args:
        train_a (list): The first spike train as a list of timestamps.
        train_b (list): The second spike train as a list of timestamps.
        f_a (list): Feature values for train_a.
        f_b (list): Feature values for train_b.
        e_pos_a (list): Exponential positive values for train_a.
        e_pos_b (list): Exponential positive values for train_b.
        e_neg_a (list): Exponential negative values for train_a.
        e_neg_b (list): Exponential negative values for train_b.
        tau (float): The time constant for the correlation.

    Returns:
        float: The correlation value.
    """
    train_a_size = len(train_a)
    train_b_size = len(train_b)

    if train_a_size == 0 or train_b_size == 0:
        return 0

    x = 0
    place_in_a = train_a_size - 1

    for i in range(train_b_size - 1, -1, -1):
        while place_in_a >= 0 and train_a[place_in_a] > train_b[i]:
            place_in_a -= 1
        if place_in_a < 0:
            break
        x += math.exp((train_a[place_in_a] - train_b[i]) / tau) * (1 + f_a[place_in_a])

    place_in_b = train_b_size - 1

    for i in range(train_a_size - 1, -1, -1):
        while place_in_b >= 0 and train_b[place_in_b] > train_a[i]:
            place_in_b -= 1
        if place_in_b < 0:
            break
        x += math.exp((train_b[place_in_b] - train_a[i]) / tau) * (1 + f_b[place_in_b])

    return x

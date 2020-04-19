"""Interface to load from and save to python pickles."""


def dump(data, filename):
    import pickle

    pickle.dump(data, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def load(filename):
    import pickle

    data = pickle.load(open(filename, 'rb'))

    return data

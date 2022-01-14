"""Interface to load from and save to python pickles."""


def dump(data, filename, **kwargs):
    import pickle

    from pyrates.utility import create_directory
    create_directory(filename)

    pickle.dump(data, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL, **kwargs)


def load(filename, **kwargs):
    import pickle

    data = pickle.load(open(filename, 'rb'), **kwargs)

    return data

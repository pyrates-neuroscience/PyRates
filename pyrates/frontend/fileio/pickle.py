"""Interface to load from and save to python pickles."""


def dump(data, filename):
    import pickle

    from pyrates.utility.filestorage import create_directory
    create_directory(filename)

    pickle.dump(data, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



def load(filename):
    import pickle

    data = pickle.load(open(filename, 'rb'))

    return data

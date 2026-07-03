import pickle

def save_chunks(documents, path):
    with open(path, "wb") as f:
        pickle.dump(documents, f)

def load_chunks(path):
    with open(path, "rb") as f:
        return pickle.load(f)
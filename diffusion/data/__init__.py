from . import mnist

def get_loader(data_id, batch_size):
    if data_id == "mnist":
        return mnist.get_loader(batch_size)
    
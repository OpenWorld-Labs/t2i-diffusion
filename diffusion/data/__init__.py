from . import imagenet_hf, mnist


def get_loader(data_id, batch_size):
    if data_id == "mnist":
        return mnist.get_loader(batch_size)
    if data_id == "imagenet":
        return imagenet_hf.get_loader(batch_size)

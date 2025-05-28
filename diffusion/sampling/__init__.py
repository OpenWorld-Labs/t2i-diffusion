from .simple import SimpleSampler


def get_sampler_cls(sampler_id):
    if sampler_id == "simple":
        return SimpleSampler

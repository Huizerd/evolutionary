import dask.bag as db


def dask_map(func, iterable):
    bag = db.from_sequence(iterable).map(func)
    return bag.compute()

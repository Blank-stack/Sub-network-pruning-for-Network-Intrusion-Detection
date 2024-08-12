from collections import OrderedDict, Mapping, Container
from pprint import pprint

from sys import getsizeof


def deep_compare(a, b, pointer='/'):
    if a == b:
        return

    if type(a) != type(b):
        reason = 'Different data types'
        extra = str((type(a), type(b)))
        x(pointer, reason, extra)

    elif type(a) in (set, frozenset):
        pointer += 'set()'
        if len(a) != len(b):
            pointer += 'set()'
            reason = 'Different number of items'
            extra = str((len(a), len(b)))
            x(pointer, reason, extra)

        reason = 'Different items'
        extra = (a, b)
        x(pointer, reason, extra)

        for i in range(len(a)):
            deep_compare(a[i], b[i], pointer + 'set()'.format(i))

    elif type(a) in (list, tuple):
        if len(a) != len(b):
            pointer += '[]'
            reason = 'Different number of items'
            extra = str((len(a), len(b)))
            x(pointer, reason, extra)

        if sorted(a) == sorted(b):
            pointer += '[]'
            reason = 'Different sort order'
            extra = 'N/A'
            x(pointer, reason, extra)

        for i in range(len(a)):
            deep_compare(a[i], b[i], pointer + '[{}]'.format(i))

    elif type(a) in (dict, OrderedDict):
        if len(a) != len(b):
            pointer += '{}'
            reason = 'Different number of items'
            extra = str((len(a), len(b)))
            x(pointer, reason, extra)

        if set(a.keys()) != set(b.keys()):
            pointer += '{}'
            reason = 'Different keys'
            extra = (a.keys(), b.keys())
            x(pointer, reason, extra)

        for k in a:
            deep_compare(a[k], b[k], pointer + '[{}]'.format(k))
    else:
        reason = 'Different objects'
        extra = (a, b)
        x(pointer, reason, extra)


def x(pointer, reason, extra):
    message = 'Objects are not the same. Pointer: {}. Reason: {}. Extra: {}'
    raise RuntimeError(message.format(pointer, reason, extra))


def compare(a, b):
    try:
        deep_compare(a, b, '/')
    except RuntimeError as e:
        pprint(e.message)


def deep_getsizeof(o, ids=set()):
    """Find the memory footprint of a Python object

    This is a recursive function that rills down a Python object graph
    like a dictionary holding nested ditionaries with lists of lists
    and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    :param o: the object
    :param ids:
    :return:
    """
    # d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    from classifier.new_MLP_IDS_plus import MLP_plus
    from prune.GateLayer import GateMLP
    from prune.Loss import DebiasedSupConLoss
    from prune.GumbelSigmoid import GumbelSigmoidMask
    import torch.nn as nn
    import numpy as np

    # if isinstance(o, str) or isinstance(0, unicode):
    if isinstance(o, str) or isinstance(o, int) or isinstance(o, float) or isinstance(o, np.float32) \
            or isinstance(o, nn.Sigmoid) or isinstance(o, nn.ReLU) or isinstance(o, nn.Softmax):
        return r
    elif isinstance(o, Mapping):
        return r + sum(deep_getsizeof(k, ids) + deep_getsizeof(v, ids) for k, v in o.items())
    elif isinstance(o, MLP_plus) or isinstance(o, GateMLP) or isinstance(o, DebiasedSupConLoss) \
            or isinstance(o, GumbelSigmoidMask) \
            or isinstance(o, nn.Linear):
        return r + sum(deep_getsizeof(k, ids) + deep_getsizeof(v, ids) for k, v in o.named_modules())
    # if isinstance(o, Container):
    else:
        return r + sum(deep_getsizeof(x, ids) for x in o)

    return r

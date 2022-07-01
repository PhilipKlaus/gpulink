from functools import wraps


class InstantiationError(Exception):
    """
    Raised when an object is being created without a factory method.
    """
    pass


def make(fn):
    """
    Decorator for converting a class method to a factory method.
    :param fn: The class method.
    :return: The wrapped factory method.
    """

    @wraps(fn)
    def _make(cls, *args, **kwargs):
        factory_key = getattr(cls, "__factory_key", None)
        if not factory_key:
            factory_key = object()
            setattr(cls, "__factory_key", factory_key)
        return fn(cls, factory_key, *args, **kwargs)

    return _make


def factory(cls):
    """
    Decorator for declaring a class as factory. In particular the decroator ensures that instances of the class are
    created using a factory method.
    :param cls: The class to be declared as a factory.
    :return: The decorated class.
    """
    old_init = getattr(cls, "__init__")

    def patch_init(self, factory_key, *_args, **_kwargs):
        stored_key = getattr(self, "__factory_key", None)
        if not stored_key or factory_key is not stored_key:
            raise InstantiationError(f"Not allowed to instantiate {type(self).__name__} directly")
        old_init(self, *_args, **_kwargs)

    setattr(cls, "__init__", patch_init)
    return cls

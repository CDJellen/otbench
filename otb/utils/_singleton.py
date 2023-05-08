class Singleton:
    """A basic singleton metaclass for single-instance objects."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Check if this class has already been initialized, return if so, init if not."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

ENCODERS = {}
CLASSIFIERS = {}

def register_encoder(name):
    def decorator(cls):
        ENCODERS[name] = cls
        return cls
    return decorator


def register_classifier(name):
    def decorator(cls):
        CLASSIFIERS[name] = cls
        return cls
    return decorator
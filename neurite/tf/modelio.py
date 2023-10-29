import tensorflow as tf
import h5py
import json
import inspect
import functools


def store_config_args(func):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'. This is used to assist
    model loading - see LoadableModel.
    """

    argspec = inspect.getfullargspec(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):

        # run the constructor first to avoid any issues with
        # the keras Model class not being initialized yet
        retval = func(self, *args, **kwargs)

        params = {}

        # first save the default values
        if argspec.defaults:
            for attr, val in zip(reversed(argspec.args), reversed(argspec.defaults)):
                params[attr] = val

        # next handle positional args
        for attr, val in zip(argspec.args[1:], args):
            params[attr] = val

        # lastly handle keyword args
        if kwargs:
            for attr, val in kwargs.items():
                params[attr] = val

        # cache for model input arguments to be saved with model
        self.config = ModelConfig(params)

        return retval
    return wrapper


class ModelConfig:
    """
    A separate class to contain the model config so that tensorflow
    doesn't try to wrap it when making checkpoints.
    """

    def __init__(self, params):
        self.params = params
        self.params['metadata'] = {}


class ReferenceContainer:
    """
    When subclassing keras Models, you can't just set some member reference a specific
    layer by doing something like:

    self.layer = layer

    because that will automatically re-add the layer weights into the model, even if they
    already exist. It's a pretty annoying feature, but I'm sure there's a valid reason for
    it... (riiight). A workaround is to configure a ReferenceContainer that wraps all layer
    pointers:

    self.references = LoadableModel.ReferenceContainer()
    self.references.layer = layer
    """

    def __init__(self):
        pass


class LoadableModel(tf.keras.Model):
    """
    Base class for easy keras model loading without having to manually
    specify the architecture configuration at load time.

    If the get_config() method is defined for a keras.Model subclass, the saved
    H5 model will automatically store the returned config. This way, we can cache
    the arguments used to the construct the initial network, so that we can construct
    the exact same network when loading from file. The arguments provided to __init__
    are automatically saved into the object (in self.config) if the __init__ method
    is decorated with the @store_config_args utility.
    """

    def get_config(self):
        """
        Returns the internal config params used to initialize the model.
        Loadable keras models expect this function to be defined.
        """
        if not hasattr(self, 'config'):
            raise RuntimeError('models that inherit from LoadableModel must decorate the '
                               'constructor with @store_config_args')
        return self.config.params

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """
        Constructs the model from the config arguments provided.
        """
        metadata = config.pop('metadata', {})
        model = cls(**config)
        model.metadata.update(metadata)
        return model

    @classmethod
    def load(cls, path, by_name=False, **kwargs):
        """
        Loads model config and weights from an H5 file. This first constructs a model using
        the config parameters stored in the H5 and then separately loads the weights. The
        keras load function is not used directly because it expects all training parameters,
        like custom losses, to be defined, which we don't want to do.
        """
        config = cls.load_config(path)
        config.update(kwargs)
        model = cls.from_config(config)
        model.load_weights(path, by_name=by_name)
        return model

    @classmethod
    def load_config(cls, path):
        """
        Loads model config dictionary from an H5 file.
        """
        with h5py.File(path, mode='r') as f:
            s = f.attrs['model_config']
            if not isinstance(s, str):
                s = s.decode('utf-8')
            config = json.loads(s)['config']

        # provide a temporary backport for the old-school enc_nf/dec_nf constructor params
        if config.get('enc_nf') and config.get('dec_nf'):
            config['nb_unet_features'] = [
                config.pop('enc_nf'),
                config.pop('dec_nf')
            ]

        return config

    @property
    def metadata(self):
        return self.config.params['metadata']

    class ReferenceContainer:
        """
        When subclassing keras Models, you can't just set some member reference a specific
        layer by doing something like:

        self.layer = layer

        because that will automatically re-add the layer weights into the model, even if they
        already exist. It's a pretty annoying feature, but I'm sure there's a valid reason for
        it... (riiight). A workaround is to configure a ReferenceContainer that wraps all layer
        pointers:

        self.references = LoadableModel.ReferenceContainer()
        self.references.layer = layer
        """

        def __init__(self):
            pass

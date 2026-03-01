"""
Example: AlexNet model plugin in the DMT-style system.

Current Brain-Score (text-search + lambda):
    # brainscore_vision/models/alexnet/__init__.py
    from brainscore_vision import model_registry
    model_registry['alexnet'] = lambda: ModelCommitment(
        identifier='alexnet',
        activations_model=get_model('alexnet'),
        layers=get_layers('alexnet'))

New style (class-based, auto-registered, validated):
    class AlexNetPlugin(ModelPlugin, identifier='alexnet'):
        @classmethod
        def build(cls):
            return ModelCommitment(...)

The new style:
- Auto-registers at class definition time (no text search)
- Validates that build() exists (fails at import, not at scoring)
- Can validate build output against BrainModelInterface (at test time)
- Keeps lazy loading (build() is called only when the model is scored)
"""

from brain_score_dmt.plugin import ModelPlugin


# --- Stub types (standing in for Brain-Score imports) ---

class PytorchWrapper:
    """Stub for brainscore_vision.model_helpers.activations.pytorch.PytorchWrapper"""
    def __init__(self, model, preprocessing, identifier=None):
        self.identifier = identifier or model.__class__.__name__
        self._model = model

    # BrainModel methods delegated through ModelCommitment
    def from_stimulus_set(self, stimulus_set, layers):
        pass


class ModelCommitment:
    """Stub for brainscore_vision.model_helpers.brain_transformation.ModelCommitment

    In real Brain-Score, this adapts a PytorchWrapper to the BrainModel interface.
    """
    def __init__(self, identifier, activations_model, layers,
                 region_layer_map=None, visual_degrees=8):
        self._identifier = identifier
        self.activations_model = activations_model
        self.layers = layers
        self._visual_degrees = visual_degrees

    @property
    def identifier(self):
        return self._identifier

    def visual_degrees(self):
        return self._visual_degrees

    def start_task(self, task, fitting_stimuli=None):
        pass

    def start_recording(self, recording_target, time_bins=None):
        pass

    def look_at(self, stimuli, number_of_trials=1):
        pass


# --- Layer definitions ---

ALEXNET_LAYERS = [
    'features.2', 'features.5', 'features.7', 'features.9', 'features.12',
    'classifier.2', 'classifier.5', 'classifier.6',
]


def get_alexnet_model():
    """Construct PytorchWrapper around torchvision AlexNet."""
    # In real code:
    #   import torchvision.models
    #   model = torchvision.models.alexnet(pretrained=True)
    #   preprocessing = load_preprocess_images(image_size=224)
    #   return PytorchWrapper(model=model, preprocessing=preprocessing,
    #                         identifier='alexnet')
    return PytorchWrapper(model=object(), preprocessing=None, identifier='alexnet')


# --- The plugin definition ---

class AlexNetPlugin(ModelPlugin, identifier='alexnet'):
    """AlexNet model plugin — DMT style.

    Subclassing ModelPlugin with identifier='alexnet' auto-registers
    cls.build as the factory in model_registry['alexnet'].
    """

    @classmethod
    def build(cls):
        return ModelCommitment(
            identifier='alexnet',
            activations_model=get_alexnet_model(),
            layers=ALEXNET_LAYERS)


# At this point, model_registry['alexnet'] == AlexNetPlugin.build
# and AlexNetPlugin._identifier == 'alexnet'


if __name__ == '__main__':
    from brain_score_dmt.plugin import model_registry

    # Load the model
    model = model_registry.load('alexnet')
    print(f"Loaded: {model.identifier}")
    print(f"Visual degrees: {model.visual_degrees()}")
    print(f"Type: {type(model).__name__}")

    # Validate against BrainModelInterface
    missing = model_registry.validate_output('alexnet')
    print(f"Missing BrainModel methods: {missing or 'none'}")

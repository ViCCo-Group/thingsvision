import warnings
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(init=True, repr=True)
class BaseExtractor:
    model_name: str
    pretrained: bool
    device: str
    model_path: str = None
    model_parameters: Any = field(default_factory=lambda: {})
    model: Any = None
    preprocess: Any = None

    def __post_init__(self) -> None:
        if not self.model:
            self.load_model()

    def show(self) -> str:
        warnings.warn(
            "\nThe .show() method is deprecated and will be removed in future versions. Use .show_model() instead.\n"
        )
        return self.show_model()

    def show_model(self) -> str:
        return self._show_model()

    def extract_features(
        self,
        batches: Iterator,
        module_name: str,
        flatten_acts: bool
    ):
        """Extract hidden unit activations (at specified layer) for every image in the database.

        Parameters
        ----------
        batches : Iterator
            Mini-batches. Iterator with equally sized
            mini-batches, where each element is a
            subsample of the full (image) dataset.
        module_name : str
            Layer name. Name of neural network layer for
            which features should be extraced.
        flatten_acts : bool
            Whether activation tensor (e.g., activations
            from an early layer of the neural network model)
            should be transformed into a vector.
         Returns
        -------
        output : np.ndarray
            Returns the feature matrix (e.g., X \in \mathbb{R}^{n \times p} if head or flatten_acts = True).
        """
        features = self._extract_features(batches, module_name, flatten_acts)

        print(
            f"...Features successfully extracted for all {len(features)} images in the database."
        )
        print(f"...Features shape: {features.shape}")

        return features

    def get_transformations(
        self, resize_dim: int = 256, crop_dim: int = 224, apply_center_crop: bool = True
    ) -> Any:
        """Load image transformations for a specific model. Image transformations depend on the backend."""
        if self.preprocess:
            return self.preprocess
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            composition = self.get_default_transformation(mean, std, resize_dim, crop_dim, apply_center_crop)
        return composition
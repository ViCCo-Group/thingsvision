from dataclasses import field
from typing import Any, Callable, Iterator, List, Optional, Union

import numpy as np
from torchtyping import TensorType
from torchvision import transforms as T

import torch

from .base import BaseExtractor

Array = np.ndarray


class PyTorchExtractor(BaseExtractor):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Any = field(default_factory=lambda: {}),
        model: Any = None,
        preprocess: Any = None,
    ) -> None:
        super().__init__(device, preprocess)
        self.model_name = model_name
        self.pretrained = pretrained
        self.model_path = model_path
        self.model_parameters = model_parameters
        self.model = model
        self.activations = {}
        self.hook_handle = None

        if not self.model:
            self.load_model()
        self.prepare_inference()

    def extract_features(
        self,
        batches: Iterator,
        module_name: str,
        flatten_acts: bool,
        output_type: str = "ndarray",
        output_dir: Optional[str] = None,
        step_size: Optional[int] = None,
    ):
        self.model = self.model.to(self.device)
        self.activations = {}
        self.register_hook(module_name=module_name)
        features = super().extract_features(
            batches=batches,
            module_name=module_name,
            flatten_acts=flatten_acts,
            output_type=output_type,
            output_dir=output_dir,
            step_size=step_size,
        )
        if self.hook_handle:
            self.hook_handle.remove()
        return features

    def get_activation(self, name: str) -> Callable:
        """Store copy of activations for a specific layer of the model."""

        def hook(model, input, output) -> None:
            # store copy of tensor rather than tensor itself
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            try:
                self.activations[name] = act.clone().detach()
            except AttributeError:
                self.activations[name] = act.clone()

        return hook

    def register_hook(self, module_name: str) -> None:
        """Register a forward hook to store activations."""
        for n, m in self.model.named_modules():
            if n == module_name:
                self.hook_handle = m.register_forward_hook(self.get_activation(n))
                break

    @torch.no_grad()
    def _extract_batch(
        self,
        batch: TensorType["b", "c", "h", "w"],
        module_name: str,
        flatten_acts: bool,
    ) -> Union[
        TensorType["b", "num_maps", "h_prime", "w_prime"],
        TensorType["b", "t", "d"],
        TensorType["b", "p"],
        TensorType["b", "d"],
    ]:
        # move current batch to torch device
        batch = batch.to(self.device)
        _ = self.forward(batch)
        act = self.activations[module_name]
        if flatten_acts:
            if self.model_name.lower().startswith("clip"):
                act = self.flatten_acts(act, batch, module_name)
            else:
                act = self.flatten_acts(act)
        return act

    def forward(
        self, batch: TensorType["b", "c", "h", "w"]
    ) -> TensorType["b", "num_cls"]:
        """Default forward pass."""
        return self.model(batch)

    @staticmethod
    def flatten_acts(
        act: Union[
            TensorType["b", "num_maps", "h_prime", "w_prime"], TensorType["b", "t", "d"]
        ]
    ) -> TensorType["b", "p"]:
        """Default flattening of activations."""
        return act.view(act.size(0), -1)

    def show_model(self) -> str:
        return self.model

    def load_model_from_source(self) -> None:
        raise NotImplementedError

    def load_model(self) -> None:
        self.load_model_from_source()
        if self.model_path:
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
            except FileNotFoundError:
                state_dict = torch.hub.load_state_dict_from_url(
                    self.model_path, map_location=self.device
                )
            self.model.load_state_dict(state_dict)

    def prepare_inference(self) -> None:
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_module_names(self) -> List[str]:
        module_names, _ = zip(*self.model.named_modules())
        module_names = list(filter(lambda n: len(n) > 0, module_names))
        return module_names

    def get_default_transformation(
        self,
        mean: List[float],
        std: List[float],
        resize_dim: int = 256,
        crop_dim: int = 224,
        apply_center_crop: bool = True,
    ) -> Callable:
        normalize = T.Normalize(mean=mean, std=std)
        composes = [T.Resize(resize_dim)]
        if apply_center_crop:
            composes.append(T.CenterCrop(crop_dim))
        composes += [T.ToTensor(), normalize]
        composition = T.Compose(composes)

        return composition

    def get_backend(self) -> str:
        return "pt"

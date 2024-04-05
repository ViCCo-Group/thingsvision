from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
from torchtyping import TensorType
from torchvision import transforms as T

from thingsvision.utils.alignment import gLocal

from .base import BaseExtractor

Array = np.ndarray
Tensor = torch.Tensor


class PyTorchExtractor(BaseExtractor):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        device: str,
        model_path: str = None,
        model_parameters: Dict[str, Union[str, bool, List[str]]] = None,
        model: Any = None,
        preprocess: Optional[Callable] = None,
    ) -> None:
        super().__init__(device, preprocess)
        self.model_name = model_name
        self.pretrained = pretrained
        self.model_path = model_path
        self.model_parameters = model_parameters
        self.model = model
        self.activations = {}
        self.hook_handle = None

        if self.model_parameters:
            if "extract_cls_token" in self.model_parameters:
                self.extract_cls_token = self.model_parameters["extract_cls_token"]

        if not self.model:
            self.load_model()
        self.prepare_inference()

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

    def _register_hook(self, module_name: str) -> None:
        """Register a forward hook to store activations."""
        for n, m in self.model.named_modules():
            if n == module_name:
                self.hook_handle = m.register_forward_hook(self.get_activation(n))
                break

    def _unregister_hook(self) -> None:
        self.hook_handle.remove()

    def batch_extraction(self, module_name: str, output_type: str) -> object:
        return BatchExtraction(
            extractor=self, module_name=module_name, output_type=output_type
        )

    def extract_batch(
        self,
        batch: TensorType["b", "c", "h", "w"],
        flatten_acts: bool,
    ) -> Union[
        TensorType["b", "num_maps", "h_prime", "w_prime"],
        TensorType["b", "t", "d"],
        TensorType["b", "p"],
        TensorType["b", "d"],
    ]:
        act = self._extract_batch(
            batch=batch, module_name=self.module_name, flatten_acts=flatten_acts
        )
        if self.output_type == "ndarray":
            act = self._to_numpy(act)
        return act

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
        if hasattr(self, "extract_cls_token"):
            if self.extract_cls_token:
                # we are only interested in the representations of the first token, i.e., [cls] token
                act = act[:, 0, :].clone()
        if flatten_acts:
            if self.model_name.lower().startswith("clip"):
                act = self.flatten_acts(act, batch, module_name)
            else:
                act = self.flatten_acts(act)
        if act.is_cuda or act.get_device() >= 0:
            torch.cuda.empty_cache()
            act = act.cpu()
        return act

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
        self._register_hook(module_name=module_name)
        features = super().extract_features(
            batches=batches,
            module_name=module_name,
            flatten_acts=flatten_acts,
            output_type=output_type,
            output_dir=output_dir,
            step_size=step_size,
        )
        if self.hook_handle:
            self._unregister_hook()
        return features

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

    def align(
        self,
        features: Union[Tensor, Array],
        module_name: str,
        alignment_type: str = "gLocal",
    ) -> Union[Tensor, Array]:
        if self.model_name == "OpenCLIP":
            base_model = self.model_name
            variant = self.model_parameters["variant"]
            dataset = self.model_parameters["dataset"]
            model_name = "_".join((base_model, variant, dataset))
        elif self.model_name == "clip" or self.model_name == "DreamSim":
            base_model = self.model_name
            variant = self.model_parameters["variant"]
            model_name = "_".join((base_model, variant))
        else:
            model_name = self.model_name
        if alignment_type == "gLocal":
            transform = gLocal(
                model_name=model_name,
                module_name=module_name,
            )
        else:
            raise NotImplementedError(
                f"\nRepresentational alignment of type: {alignment_type} is not yet implemented.\nChange type to gLocal!\n"
            )
        aligned_fetures = transform.apply_transform(features)
        return aligned_fetures

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


class BatchExtraction(object):

    def __init__(
        self, extractor: PyTorchExtractor, module_name: str, output_type: str
    ) -> None:
        self.extractor = extractor
        self.module_name = module_name
        self.output_type = output_type

    def __enter__(self) -> PyTorchExtractor:
        self.extractor._module_and_output_check(self.module_name, self.output_type)
        self.extractor._register_hook(self.module_name)
        setattr(self.extractor, "module_name", self.module_name)
        setattr(self.extractor, "output_type", self.output_type)
        return self.extractor

    def __exit__(self, *args):
        self.extractor._unregister_hook()
        delattr(self.extractor, "module_name")
        delattr(self.extractor, "output_type")

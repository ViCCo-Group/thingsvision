import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import numpy as np
from thingsvision.utils.alignment import gLocal
from torchtyping import TensorType
from torchvision import transforms as T

import torch
from torch.utils.data import DataLoader

from .base import BaseExtractor

Array = np.ndarray
Tensor = torch.Tensor

TOKEN_EXTRACTIONS = ["cls_token", "avg_pool", "cls_token+avg_pool"]


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
            if "token_extraction" in self.model_parameters:
                self.token_extraction = self.model_parameters["token_extraction"]
                assert (
                    self.token_extraction in TOKEN_EXTRACTIONS
                ), f"\nFor token extraction use one of the following: {TOKEN_EXTRACTIONS}.\n"
            elif "extract_cls_token" in self.model_parameters:
                warnings.warn(
                    "\nThe argument 'extract_cls_token' is deprecated since version 2.6.2!. "
                    "\nFor future calls, use the keyword argument 'token_extraction' instead. "
                    "\nSee the docs for more details.\n",
                    category=DeprecationWarning,
                )
                if self.model_parameters["extract_cls_token"]:
                    self.token_extraction = "cls_token"

        if not self.model:
            self.load_model()
        # move model to current device and set it to eval mode
        self.prepare_inference()

    def get_activation(self, name: str) -> Callable:
        """Store a copy of the representations for a specific module of the model."""

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

    def _register_hooks(self, module_names: List[str]) -> None:
        """Register a forward hook to store activations for multiple modules."""
        self.hook_handles = []
        for n, m in self.model.named_modules():
            if n in module_names:
                handle = m.register_forward_hook(self.get_activation(n))
                self.hook_handles.append(handle)

    def _unregister_hooks(self) -> None:
        """Unregister all forward hooks."""
        if self.hook_handles:
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles = []
        else:
            warnings.warn(
                "\nNo hooks were registered. Nothing to unregister.\n",
                category=UserWarning,
            )

    def batch_extraction(
        self,
        module_name: Optional[str] = None,
        module_names: Optional[List[str]] = None,
        output_type: str = "ndarray",
    ) -> object:
        """Allows mini-batch extraction for custom data pipeline using a with-statement."""
        return BatchExtraction(
            extractor=self,
            module_name=module_name,
            module_names=module_names,
            output_type=output_type,
        )

    def extract_batch(
        self,
        batch: TensorType["b", "c", "h", "w"],
        flatten_acts: bool,
    ) -> Union[
        # This is the return type when 'module_names' is used
        Dict[
            str,
            Union[
                Union[
                    TensorType["n", "num_maps", "h_prime", "w_prime"],
                    TensorType["n", "t", "d"],
                    TensorType["n", "p"],
                    TensorType["n", "d"],
                ],
                Array,
            ],
        ],
        # This is the return type when 'module_name' is used (for backward compatibility)
        Union[
            TensorType["n", "num_maps", "h_prime", "w_prime"],
            TensorType["n", "t", "d"],
            TensorType["n", "p"],
            TensorType["n", "d"],
            Array,
        ],
    ]:
        acts = self._extract_batch(
            batch=batch,
            module_names=self.module_names,
            flatten_acts=flatten_acts,
        )
        if self.output_type == "ndarray":
            for module_name, act in acts.items():
                acts[module_name] = self._to_numpy(act)
        if getattr(self, "module_name", None) is not None:
            return acts[self.module_name]
        return acts

    @torch.no_grad()
    def _extract_batch(
        self,
        batch: TensorType["b", "c", "h", "w"],
        module_names: List[str],
        flatten_acts: bool,
    ) -> Dict[
        str,
        Union[
            TensorType["b", "num_maps", "h_prime", "w_prime"],
            TensorType["b", "t", "d"],
            TensorType["b", "p"],
            TensorType["b", "d"],
        ],
    ]:
        """Extract representations from a batch of images."""
        # move mini-batch to current device
        batch = batch.to(self.device)
        batch_size = batch.shape[0]
        _ = self.forward(batch)
        acts = {}
        for module_name in module_names:
            act = self.activations[module_name]
            if act.shape[0] != batch_size:
                raise ValueError(
                    f"The number of extracted features ({act.shape=}) does not match the batch size ({batch.shape=}). "
                    "Please check the model, the module name, and the model parameters ..."
                )
            if len(act.shape) > 2:
                if hasattr(self, "token_extraction"):
                    if self.token_extraction == "cls_token":
                        act = act[:, 0, :].clone()
                    elif self.token_extraction == "avg_pool":
                        act = act[:, 1:, :].clone().mean(dim=1)
                    elif self.token_extraction == "cls_token+avg_pool":
                        cls_token = act[:, 0, :].clone()
                        pooled_tokens = act[:, 1:, :].clone().mean(dim=1)
                        act = torch.cat((cls_token, pooled_tokens), dim=1)
                    else:
                        raise ValueError(
                            f"\n{self.token_extraction} is not a valid value for token extraction. "
                            "\nChoose one of the following: {TOKEN_EXTRACTIONS}.\n "
                        )
                elif flatten_acts:
                    if self.model_name.lower().startswith("clip"):
                        act = self.flatten_acts(act, batch, module_name)
                    else:
                        act = self.flatten_acts(act)
            if act.is_cuda or act.get_device() >= 0:
                torch.cuda.empty_cache()
                act = act.cpu()
            acts[module_name] = act
        return acts

    def extract_features(
        self,
        batches: Iterator,
        module_name: Optional[str] = None,
        module_names: Optional[List[str]] = None,
        flatten_acts: bool = False,
        output_type: str = "ndarray",
        output_dir: Optional[str] = None,
        step_size: Optional[int] = None,
        file_name_suffix: str = "",
        save_in_one_file: bool = False,
    ):
        if not bool(module_name) ^ bool(module_names):
            raise ValueError(
                "\nPlease provide either a single module name or a list of module names, but not both.\n"
            )
        self.model = self.model.to(self.device)
        self.activations = {}
        if module_name is not None:
            self._register_hooks(module_names=[module_name])
        else:
            self._register_hooks(module_names=module_names)

        with ImageOnlyDataloaderModifier(dataloader=batches) as dataloader:
            features = super().extract_features(
                batches=dataloader,
                module_name=module_name,
                module_names=module_names,
                flatten_acts=flatten_acts,
                output_type=output_type,
                output_dir=output_dir,
                step_size=step_size,
                file_name_suffix=file_name_suffix,
                save_in_one_file=save_in_one_file,
            )
        self._unregister_hooks()
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
        ],
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
        """Align the representations with human (global) object similarity."""
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
        """Prepare the model for inference by moving it to current device and setting it to eval mode."""
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_module_names(self) -> List[str]:
        """Return the names of all modules in a model."""
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
        self,
        extractor: PyTorchExtractor,
        module_name: Optional[str] = None,
        module_names: Optional[List[str]] = None,
        output_type: str = "ndarray",
    ) -> None:
        """
        Mini-batch extraction object that can be used as a with-statement in a PyTorch extractor.

        Parameters
        ----------
        extractor (object): PyTorchExtractor class.
        module_name (str): The module of model for which features will be extracted.
        module_names (List[str]): List of modules of model for which features will be extracted.
        output_type (str): Type of the feature matrix returned by the extractor.

        """
        if not bool(module_name) ^ bool(module_names):
            raise ValueError(
                "\nPlease provide either a single module name or a list of module names, but not both.\n"
            )
        if module_name is not None:
            module_names = [module_name]
        self.extractor = extractor
        self.module_name = module_name
        self.module_names = module_names
        self.output_type = output_type

    def __enter__(self) -> PyTorchExtractor:
        """Registering hooks and setting attributes during opening."""
        self.extractor._module_and_output_check(self.module_names, self.output_type)
        self.extractor._register_hooks(self.module_names)
        setattr(self.extractor, "module_name", self.module_name)
        setattr(self.extractor, "module_names", self.module_names)
        setattr(self.extractor, "output_type", self.output_type)
        return self.extractor

    def __exit__(self, *args):
        """Removing hooks and deleting attributes at closing."""
        self.extractor._unregister_hooks()
        delattr(self.extractor, "module_name")
        delattr(self.extractor, "module_names")
        delattr(self.extractor, "output_type")


class ImageOnlyDataloaderModifier:
    """Class to temporarily replace the collate function of a dataloader with images only collate.
    This is useful when we want to use the dataloader for feature extraction with thingsvision.
    Assumes that a dataloader either returns image or a tuple of (image, *args) or a list of (image, *args).
    If the dataloader returns a tuple of (image, *args) or a list of (image, *args), it will return only the images.
    The class does not modify otherwise (e.g., specialized dataloaders).
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.dataloader = dataloader
        self.new_collate_fn = self._images_only_collate
        self.original_collate_fn = None
        self.should_replace = False

    @staticmethod
    def _images_only_collate(batch: list[tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """Collate function to return only the images from the batch.
        This is useful when we want to use the dataloader for feature extraction with thingsvision.
        """
        return torch.stack([item[0] for item in batch])
    
    def _check_dataloader_format(self) -> bool:
        sample_batch = next(iter(self.dataloader))
        return isinstance(sample_batch, (tuple, list)) and len(sample_batch) == 2

    def __enter__(self) -> DataLoader:
        """Enter the context manager and replace the collate function with images only collate,
        if the dataloader is in the correct format.
        """
        self.should_replace = self._check_dataloader_format()
        if self.should_replace:
            warnings.warn(
                "\nThe dataloader is not in the correct format. The collate function will be replaced with images only collate.\n"
            )
            self.original_collate_fn = self.dataloader.collate_fn
            self.dataloader.collate_fn = self.new_collate_fn
        return self.dataloader

    def __exit__(self, *args) -> None:
        """Exit the context manager and restore the original collate function,
        if the dataloader was not in the correct format.
        """
        if self.should_replace and self.original_collate_fn is not None:
            self.dataloader.collate_fn = self.original_collate_fn

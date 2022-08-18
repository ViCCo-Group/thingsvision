import re
from typing import Tuple, List, Iterator, Dict, Any

import numpy as np 

from PIL import Image
from numpy.core.fromnumeric import resize

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.applications as tensorflow_models
from tensorflow.keras import layers

import thingsvision.cornet as cornet
import thingsvision.clip as clip
import thingsvision.custom_models as custom_models

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.models as torchvision_models

class Model():
    def __init__(self, 
                 model_name: str, 
                 pretrained: bool,
                 device: str,
                 model_path: str=None,
                 source: str=None):
        """
        Parameters
        ----------
        Model_name : str
            Model name. Name of model for which features should
            subsequently be extracted.
        pretrained : bool
            Whether to load a model with pretrained or
            randomly initialized weights into memory.
        device : str
            Device. Whether model weights should be moved
            to CUDA or left on the CPU.
        model_path : str (optional)
            path/to/weights. If pretrained is set to False,
            model weights can be loaded from a path on the
            user's machine. This is useful when operating
            on a server without network access, or when
            features should be extracted for a model that
            was fine-tuned (or trained) on a custom image
            dataset.
        source: str (optional)
            Source of model and weights. If not set, all 
            models sources are searched for the model name 
            until the first occurence.
        """

        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.model_path = model_path
        self.source = source
        self.load_model()


    def get_model_from_torchvision(self) -> Tuple[Any, str]:
        backend = 'pt'
        if hasattr(torchvision_models, self.model_name):
            model = getattr(torchvision_models, self.model_name)
            model = model(pretrained=self.pretrained)
            return model, backend 
            
    def get_model_from_timm(self) -> Tuple[Any, str]:
        backend = 'pt'
        if self.model_name in timm.list_models():
            model = timm.create_model(self.model_name, self.pretrained)
            return model, backend

    def get_model_from_custom_models(self) -> Tuple[Any, str]:
        if self.model_name.startswith('clip'):
            if self.model_name.endswith('ViT'):
                clip_model_name = "ViT-B/32"
            else:
                clip_model_name = "RN50"
            model, self.clip_n_px = clip.load(
                                            clip_model_name,
                                            device=self.device,
                                            model_path=self.model_path,
                                            pretrained=self.pretrained,
                                            jit=False,
                                        )
            backend = 'pt'
            return model, backend

        if self.model_name.startswith('cornet'):
            try:
                model = getattr(cornet, f'cornet_{self.model_name[-1]}')
            except AttributeError:
                model = getattr(cornet, f'cornet_{self.model_name[-2:]}')
            model = model(pretrained=self.pretrained, map_location=torch.device(self.device))
            model = model.module    # remove DataParallel
            backend = 'pt'
            return model, backend

        if hasattr(custom_models, self.model_name):
            custom_model = getattr(custom_models, self.model_name)
            custom_model = custom_model(self.device)
            model = custom_model.create_model()
            backend = custom_model.get_backend()
            return model, backend 

    def get_model_from_keras(self) -> Tuple[Any, str]:
        backend = 'tf'
        if hasattr(tensorflow_models, self.model_name):
            model = getattr(tensorflow_models, self.model_name)
            if self.pretrained:
                weights = 'imagenet'
            elif self.model_path:
                weights = self.model_path
            else:
                weights = None
            model = model(weights=weights)
            return model, backend

    def load_model_from_source(self) -> None:
        model = None
        backend = None
        if self.source == 'timm':
            model, backend = self.get_model_from_timm()
        elif self.source == 'keras':
            model, backend = self.get_model_from_keras()
        elif self.source == 'torchvision':
            model, backend = self.get_model_from_torchvision()
        elif self.source == 'custom':
            model, backend = self.get_model_from_custom_models()
        else:
            raise ValueError(f'\nCannot load models from {self.source}.\nUse a different source.\n')

        if isinstance(model, type(None)):
            raise ValueError(f'\nCould not find {self.model_name} in {self.source}.\nCheck whether model name is correctly spelled or use a different model.\n')
        else:
            self.model = model
            self.backend = backend

    def load_model(self) -> None:
        """Load a pretrained model into memory."""
        if self.source:
            self.load_model_from_source()
        else:
            for model_loader in [self.get_model_from_torchvision, 
                                self.get_model_from_timm,
                                self.get_model_from_keras,
                                self.get_model_from_custom_models]:
                result = model_loader()
                if result:
                    found_model, backend = result
                    self.model = found_model
                    self.backend = backend
                    break
            
            if not self.model:
                raise Exception(f'Model {self.model_name} not found in all sources')

        if self.backend == 'pt':
            device = torch.device(self.device)
            if self.model_path:
                try:
                    state_dict = torch.load(self.model_path, map_location=device)
                except FileNotFoundError:
                    state_dict = torch.hub.load_state_dict_from_url(self.model_path, map_location=device)
                self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model = self.model.to(device)

    def show(self) -> str:
        """Show architecture of model to select a layer."""
        if self.backend == 'pt':
            if re.search(r'^clip', self.model_name):
                for l, (n, p) in enumerate(self.model.named_modules()):
                    if l > 1:
                        if re.search(r'^visual', n):
                            print(n)
                print('visual')
            else:
                print(self.model)
        else:
            print(self.model.summary())
        print(f'\nEnter module name for which you would like to extract features:\n')
        module_name = str(input())
        print()
        return module_name


    def tf_extraction(
        self,
        data_loader: Iterator,
        module_name: str,
        flatten_acts: bool,
        return_probabilities: bool=False,
    ) -> tuple:
        """Feature extraction helper function for TensorFlow models."""
        features, targets = [], []
        if return_probabilities:
            probabilities = []
        for batch in data_loader:
            X, y = batch
            layer_out = [self.model.get_layer(module_name).output]
            activation_model = keras.models.Model(
                inputs=self.model.input,
                outputs=layer_out,
            )
            activations = activation_model.predict(X)
            if flatten_acts:
                activations = activations.reshape(activations.shape[0], -1)
            features.append(activations)
            targets.extend(y.numpy())
            if return_probabilities:
                out = self.model.predict(X)
                probas = tf.nn.softmax(out, axis=1)
                probabilities.append(probas)
        features = np.vstack(features)
        targets = np.asarray(targets).ravel()
        if return_probabilities:
            probabilities = np.vstack(probabilities)
            return features, targets, probabilities
        return features, targets

    
    @torch.no_grad()
    def pt_extraction(
            self,
            data_loader: Iterator,
            module_name: str,
            flatten_acts: bool,
            clip: bool=False,
            return_probabilities: bool=False,
        ) -> tuple:
        """Feature extraction helper function for PyTorch models."""
        device = torch.device(self.device)
        # initialise dictionary to store hidden unit activations per mini-batch
        global activations
        activations = {}
        # register forward hook to store activations
        model = self.register_hook()
        features, targets = [], []
        if return_probabilities:
            probabilities = []
        for batch in data_loader:
            X, y = (t.to(device) for t in batch)
            if clip:
                img_features = model.encode_image(X)
                if module_name == 'visual':
                    assert torch.unique(activations[module_name] == img_features).item(
                    ), '\nImage features should represent activations in last encoder layer.\n'
            else:
                out = model(X)
                if return_probabilities:
                    probas = F.softmax(out, dim=1)
                    probabilities.append(probas)
            act = activations[module_name]
            if flatten_acts:
                if clip:
                    if re.compile(r'attn$').search(module_name):
                        if isinstance(act, tuple):
                            act = act[0]
                    else:
                        if act.size(0) != X.shape[0] and len(act.shape) == 3:
                            act = act.permute(1, 0, 2)
                act = act.view(act.size(0), -1)
            features.append(act.cpu().numpy())
            targets.extend(y.squeeze(-1).cpu().numpy())
        features = np.vstack(features)
        targets = np.asarray(targets).ravel()
        if return_probabilities:
            probabilities = np.vstack(probabilities)
            return features, targets, probabilities
        return features, targets


    def extract_features(
            self,
            data_loader: Iterator,
            module_name: str,
            flatten_acts: bool,
            clip: bool = False,
            return_probabilities: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract hidden unit activations (at specified layer) for every image in the database.

            Parameters
            ----------
            data_loader : Iterator
                Mini-batches. Iterator with equally sized
                mini-batches, where each element is a
                subsample of the full image dataset.
            module_name : str
                Layer name. Name of neural network layer for
                which features should be extraced.
            flatten_acts : bool
                Whether activation tensor (e.g., activations
                from an early layer of the neural network model)
                should be transformed into a vector.
            clip : bool (optional)
                Whether neural network model is a CNN-based
                torchvision or CLIP-based model. Since CLIP
                has a different training objective, feature
                extraction must be performed differently.
                For PyTorch only.
            return_probabilities : bool (optional)
                Whether class probabilities (softmax predictions)
                should be returned in addition to the feature matrix
                and the target vector.

            Returns
            -------
            output : Tuple[np.ndarray, np.ndarray] OR Tuple[np.ndarray, np.ndarray, np.ndarray]
                Returns the feature matrix and the target vector OR in addition to the feature
                matrix and the target vector, the class probabilities.
        """
        if self.backend == 'pt':
            if return_probabilities:
                assert not clip, '\nCannot extract features for CLIP and return class predictions simultaneously. We will add this possibility in a future version.\n'
                features, targets, probabilities = self.pt_extraction(
                    data_loader=data_loader,
                    module_name=module_name,
                    flatten_acts=flatten_acts,
                    clip=clip,
                    return_probabilities=True,
                )
            else:
                features, targets = self.pt_extraction(
                    data_loader=data_loader,
                    module_name=module_name,
                    flatten_acts=flatten_acts,
                    clip=clip, 
                )
        else:
            if return_probabilities:
                features, targets, probabilities = self.tf_extraction(
                    data_loader=data_loader,
                    module_name=module_name,
                    flatten_acts=flatten_acts,
                    return_probabilities=True,
                )
            else:
                features, targets = self.tf_extraction(
                    data_loader=data_loader,
                    module_name=module_name,
                    flatten_acts=flatten_acts,
                )        
        if return_probabilities:
            assert features.shape[0] == len(targets) == len(
                    probabilities), '\nFeatures, targets, and probabilities must correspond to the same number of images.\n'
            print(f'...Features successfully extracted for all {len(features)} images in the database.')
            print(f'...Features shape: {features.shape}')
            return features, targets, probabilities
        assert features.shape[0] == len(
                targets), '\nFeatures and targets must correspond to the same number of images.\n'
        print(f'...Features successfully extracted for all {len(features)} images in the database.')
        print(f'...Features shape: {features.shape}')
        return features, targets


    def get_activation(self, name):
        """Store copy of hidden unit activations at each layer of model."""
        def hook(model, input, output):
            # store copy of tensor rather than tensor itself
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            try:
                activations[name] = act.clone().detach()
            except AttributeError:
                activations[name] = act.clone()
        return hook


    def register_hook(self):
        """Register a forward hook to store activations."""
        for n, m in self.model.named_modules():
            m.register_forward_hook(self.get_activation(n))
        return self.model


    def get_transformations(self, resize_dim: int = 256, crop_dim: int = 224, apply_center_crop: bool = True):
        if re.search(r'^clip', self.model_name):
            if self.backend != 'pt':
                raise Exception("You need to use PyTorch 'pt' as backend if you want to use the CLIP model.")

            composes = [
                T.Resize(self.clip_n_px, interpolation=Image.BICUBIC)
            ]

            if apply_center_crop:
                composes.append(T.CenterCrop(self.clip_n_px))

            composes += [lambda image: image.convert("RGB"),
                         T.ToTensor(),
                         T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]

            composition = T.Compose(composes)
            return composition
            
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.backend == 'pt':
            normalize = T.Normalize(mean=mean, std=std)
            composes = [T.Resize(resize_dim)]
            if apply_center_crop:
                composes.append(T.CenterCrop(crop_dim))
            composes += [T.ToTensor(), normalize]
            composition = T.Compose(composes)
            return composition
        elif self.backend == 'tf':
            resize_dim = crop_dim
            composes = [layers.experimental.preprocessing.Resizing(resize_dim, resize_dim)]

            if apply_center_crop:
                pass
                #composes.append(layers.experimental.preprocessing.CenterCrop(crop_dim, crop_dim))
            composes += [layers.experimental.preprocessing.Normalization(mean=mean, variance=[std_ * std_ for std_ in std])]
            resize_crop_and_normalize = tf.keras.Sequential(composes)
            return resize_crop_and_normalize



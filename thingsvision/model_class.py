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
                 backend: str='pt'):
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
        backend : str (optional)
            Deep learning framework that should be used.
            'pt' for PyTorch and 'tf' for Tensorflow
        """

        self.model_name = model_name
        self.backend = backend
        self.pretrained = pretrained
        self.device = device
        self.model_path = model_path
        self.load_model()


    def load_model(self) -> Tuple[Any, Any]:
        """Load a pretrained *torchvision* or CLIP model into memory."""
        if self.backend == 'pt':
            if re.search(r'^clip', self.model_name):
                clip_model_name = "RN50"
                if re.search(r'ViT$', self.model_name):
                    clip_model_name = "ViT-B/32"
                self.model, self.clip_n_px = clip.load(
                                                    clip_model_name,
                                                    device=self.device,
                                                    model_path=self.model_path,
                                                    pretrained=self.pretrained,
                                                    jit=False,
                                            )
            else:
                device = torch.device(self.device)
                if re.search(r'^cornet', self.model_name):
                    try:
                        self.model = getattr(cornet, f'cornet_{self.model_name[-1]}')
                    except:
                        self.model = getattr(cornet, f'cornet_{self.model_name[-2:]}')
                    self.model = self.model(pretrained=self.pretrained, map_location=device)
                    self.model = self.model.module    # remove DataParallel
                else:
                    self.model = getattr(torchvision_models, self.model_name)
                    self.model = self.model(pretrained=self.pretrained)
                self.model = self.model.to(device)
            if self.model_path:
                state_dict = torch.load(self.model_path, map_location=device)
                self.model.load_state_dict(state_dict)
            self.model.eval()
        elif self.backend == 'tf':
            model = getattr(tensorflow_models, self.model_name)
            if self.pretrained:
                weights = 'imagenet'
            elif self.model_path:
                weights = self.model_path
            else:
                weights = None
            
            self.model = model(weights=weights)
    

    def show(self) -> str:
        """Show architecture of model to select a layer."""
        if re.search(r'^clip', self.model_name):
            for l, (n, p) in enumerate(self.model.named_modules()):
                if l > 1:
                    if re.search(r'^visual', n):
                        print(n)
            print('visual')
        else:
            print(self.model)
        print(f'\nEnter module name for which you would like to extract features:\n')
        module_name = str(input())
        print()
        return module_name

    def extract_features(
            self,
            data_loader: Iterator,
            module_name: str,
            batch_size: int,
            flatten_acts: bool,
            device: str,
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
            device : str
                Device. Whether feature extraction should
                be executed on CUDA or CPU.
            clip : bool (optional)
                Whether neural network model is a CNN-based
                torchvision or CLIP-based model. Since CLIP
                has a different training objective, feature
                extraction must be performed differently.
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
        features, targets = [], []
        if return_probabilities:
            assert not clip, '\nCannot extract activations for CLIP and return class predictions simultaneously. This feature will be implemented in a future version.\n'
            probabilities = []

        if self.backend == 'pt':
            if re.search(r'ensemble$', module_name):
                ensembles = ['conv_ensemble', 'maxpool_ensemble']
                assert module_name in ensembles, f'\nIf aggregating filters across layers and subsequently concatenating features, module name must be one of {ensembles}\n'
                if re.search(r'^conv', module_name):
                    feature_extractor = nn.Conv2d
                else:
                    feature_extractor = nn.MaxPool2d

            device = torch.device(self.device)
            # initialise dictionary to store hidden unit activations on the fly
            global activations
            activations = {}
            # register forward hook to store activations
            model = self.register_hook()

            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    batch = (t.to(device) for t in batch)
                    X, y = batch
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

                    if re.search(r'ensemble$', module_name):
                        layers = self.enumerate_layers(feature_extractor)
                        act = self.nsemble_featmaps(activations, layers, 'max')
                    else:
                        act = activations[module_name]
                        if flatten_acts:
                            if clip:
                                if re.search(r'attn$', module_name):
                                    act = act[0]
                                else:
                                    if act.size(0) != X.shape[0] and len(act.shape) == 3:
                                        act = act.permute(1, 0, 2)
                            act = act.view(act.size(0), -1)
                    features.append(act.cpu())
                    targets.extend(y.squeeze(-1).cpu()) 
        elif self.backend == 'tf':
            for i, batch in enumerate(data_loader):
                X, y = batch
                layer_outputs = [self.model.get_layer(module_name).output]
                activation_model = keras.models.Model(inputs=self.model.input, outputs=layer_outputs)
                activations = activation_model.predict(X)
                features.append(activations)
                targets.extend(y.numpy())
                if return_probabilities:
                    probas = tf.nn.softmax(activations, axis=1)
                    probabilities.append(probas)

        # stack each mini-batch of hidden activations to obtain an N x F matrix, and flatten targets to yield vector
        features = np.vstack(features)
        targets = np.asarray(targets).ravel()

        if return_probabilities:
            probabilities = np.vstack(probabilities)
            assert len(features) == len(targets) == len(
                    probabilities), '\nFeatures, targets, and probabilities must correspond to the same number of images.\n'
            return features, targets, probabilities
        assert len(features) == len(
                targets), '\nFeatures and targets must correspond to the same number of images.\n'
        print(f'...Features successfully extracted for all {len(features)} images in the database.')
        print(f'...Features shape: {features.shape}')
        return features, targets


    def enumerate_layers(self, feature_extractor) -> List[int]:
        layers = []
        k = 0
        for n, m in self.model.named_modules():
            if re.search(r'\d+$', n):
                if isinstance(m, feature_extractor):
                    layers.append(k)
                k += 1
        return layers


    def ensemble_featmaps(
        self,
        activations: dict,
        layers: list,
        pooling: str = 'max',
        alpha: float = 3.,
        beta: float = 5.,
    ) -> torch.Tensor:
        """Concatenate globally (max or average) pooled feature maps."""
        acts = [activations[''.join(('features.', str(l)))] for l in layers]
        func = torch.max if pooling == 'max' else torch.mean
        pooled_acts = [torch.tensor([list(map(func, featmaps))
                                    for featmaps in acts_i]) for acts_i in acts]
        # upweight second-to-last conv layer by 5.
        pooled_acts[-2] = pooled_acts[-2] * alpha
        # upweight last conv layer by 10.
        pooled_acts[-1] = pooled_acts[-1] * beta
        stacked_acts = torch.cat(pooled_acts, dim=1)
        return stacked_acts

    
    def get_activation(self, name):
        """Store hidden unit activations at each layer of model."""
        def hook(model, input, output):
            try:
                activations[name] = output.detach()
            except AttributeError:
                activations[name] = output
        return hook


    def register_hook(self):
        """Register a forward hook to store activations."""
        for n, m in self.model.named_modules():
            m.register_forward_hook(self.get_activation(n))
        return self.model


    def get_transformations(self, resize_dim: int = 256, crop_dim: int = 224):
        if re.search(r'^clip', self.model_name):
            if self.backend != 'pt':
                raise Exception("You need to use Tensorflow 'tf' as backend if you want to use the CLIP model.")

            composition = T.Compose([
                T.Resize(self.clip_n_px, interpolation=Image.BICUBIC),
                T.CenterCrop(self.clip_n_px),
                lambda image: image.convert("RGB"),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            return composition
            
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.backend == 'pt':
            normalize = T.Normalize(mean=mean, std=std)
            composition = T.Compose([T.Resize(resize_dim), T.CenterCrop(crop_dim), T.ToTensor(), normalize])
            return composition
        elif self.backend == 'tf':
            resize_dim = crop_dim
            resize_crop_and_normalize = tf.keras.Sequential([
                layers.experimental.preprocessing.Resizing(resize_dim, resize_dim),
                #layers.experimental.preprocessing.CenterCrop(crop_dim, crop_dim)
                layers.experimental.preprocessing.Normalization(mean=mean, variance=[std_ * std_ for std_ in std])
            ])
            return resize_crop_and_normalize



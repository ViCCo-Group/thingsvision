import torch
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset, DataLoader
from torchvision import transforms
import numpy as np
import pdb

model_name = 'DreamSim'
source = 'custom'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for model_type in ["clip_vitb32", "open_clip_vitb32"]:
    print(f"Testing {model_type}")
    print("Loading extractor")
    extractor = get_extractor(
        model_name=model_name,
        source=source,
        device=device,
        pretrained=True,
        model_parameters={"model_type": model_type}
    )

    print("Setting up dataset")
    root='./dreamsim_test/test_ims' # (e.g., './images/)
    batch_size = 32

    dataset = ImageDataset(
        root=root,
        out_path='./dreamsim_test/output_features',
        backend=extractor.get_backend(), # backend framework of model
        transforms=extractor.get_transformations(resize_dim=256, crop_dim=224) # set input dimensionality to whatever is required for your pretrained model
    )

    batches = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        backend=extractor.get_backend() # backend framework of model
    )

    module_name = 'model.mlp'
    features = extractor.extract_features(
        batches=batches,
        module_name=module_name,
        flatten_acts=True, # flatten 2D feature maps from convolutional layer
        output_type="ndarray", # or "tensor" (only applicable to PyTorch models)
    )
    # pdb.set_trace()
    print("Done extracting features. Loading ground-truth features")

    embeddings = np.load(f"./dreamsim_test/{model_type}_embeds.npy")
    assert np.allclose(embeddings.flatten(), features.flatten(), atol=1e-04)
    print("Passed")

print("done :)")
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# activate __version__ variable
exec(open("thingsvision/_version.py").read())

requirements = [
    "ftfy",
    "h5py",
    "matplotlib",
    "numba",
    "numpy<2",
    "open_clip_torch==3.*",
    "pandas",
    "regex",
    "safetensors<0.6",
    "scikit-image",
    "scikit-learn",
    "scipy",
    "tensorflow<2.16",
    "timm",
    "torch>=2.0.0",
    "torchvision==0.15.2",
    "torchtyping",
    "tqdm",
    "accelerate<1.10.0",
    "transformers==4.40.1",
    "pytest",
    ]

setuptools.setup(
    name="thingsvision",
    version=__version__,
    author="Lukas Muttenthaler",
    author_email="muttenthaler@cbs.mpg.de",
    description="Extracting image features from state-of-the-art neural networks for Computer Vision made easy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ViCCo-Group/thingsvision",
    packages=setuptools.find_packages(),
    license="MIT License",
    install_requires=requirements,
    keywords="feature extraction",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": ["thingsvision = thingsvision.thingsvision:main"]},
    python_requires=">=3.10"
)

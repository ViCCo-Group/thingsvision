import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
                'matplotlib',
                'numba',
                'ftfy',
                'numpy',
                'pandas',
                'torch',
                'torchvision',
                'tqdm',
                'regex',
                'scikit-image',
                'scipy',
                'h5py',
                ]

setuptools.setup(
                name="thingsvision",
                version="0.9.6",
                author="Lukas Muttenthaler",
                author_email="muttenthaler@cbs.mpg.de",
                description="Extracting image features from state-of-the-art neural networks for Computer Vision made easy",
                long_description=long_description,
                long_description_content_type="text/markdown",
                url="https://github.com/ViCCo-Group/THINGSvision",
                packages=setuptools.find_packages(),
                license="MIT License",
                install_requires=requirements,
                keywords="feature extraction",
                classifiers=[
                    "Programming Language :: Python :: 3.7",
                    "Natural Language :: English",
                    "License :: OSI Approved :: MIT License",
                    "Operating System :: OS Independent",
                ],
                python_requires='>=3.7',
)

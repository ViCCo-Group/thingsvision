import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = [
                'matplotlib==3.5.2',
                'numba==0.56.*',
                'ftfy',
                'numpy==1.22.*',
                'pandas==1.4.*',
                'tensorflow==2.9.*',
                'torch==1.12.*',
                'torchvision==0.13.*',
                'open_clip_torch==2.0.*',
                'tqdm==4.64.0',
                'timm==0.6.*',
                'regex',
                'scikit-image==0.19.3',
                'scikit-learn==1.1.*',
                'scipy==1.8.1',
                'h5py==3.7.0',
                'CLIP'
                ]

setuptools.setup(
                name="thingsvision",
                version="2.1.0",
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
                    "Programming Language :: Python :: 3.8",
                    "Natural Language :: English",
                    "License :: OSI Approved :: MIT License",
                    "Operating System :: OS Independent",
                ],
                python_requires='>=3.8',
                dependency_links=['git+https://github.com/openai/CLIP.git']
)

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="thingsvision-LukasMut", # Replace with your own username
    version="0.0.1",
    author="Lukas Muttenthaler",
    author_email="muttenthaler@cbs.mpg.de",
    description="A library to extract image features from state-of-the-art neural networks for Computer Vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/thingsvision",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

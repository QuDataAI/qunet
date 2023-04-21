import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QuNet",
    version="0.0.4",
    author="synset",
    author_email="steps137ai@gmail.com",
    description="Working with deep learning models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/step137/qunet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
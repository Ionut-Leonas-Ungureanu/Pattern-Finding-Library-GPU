import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pyDMLGPU',
    version='0.1',
    author="Ionut-Leonas Ungureanu",
    author_email="uionutleonas@gmail.com",
    description="A utility package for data mining on GPU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://url",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

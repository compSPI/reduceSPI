"""Create instructions to build the reduceSPI package."""

import setuptools

requirements = []

setuptools.setup(
    name="reduceSPI",
    maintainer="Nina Miolane",
    version="0.0.1",
    maintainer_email="nmiolane@gmail.com",
    description="Single particle imaging dimension reduction package",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/compSPI/reduceSPI.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)

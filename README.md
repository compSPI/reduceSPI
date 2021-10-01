[![Build](https://github.com/compSPI/reduceSPI/actions/workflows/build.yml/badge.svg)](https://github.com/compSPI/reduceSPI/actions/workflows/build.yml)
[![DeepSource](https://deepsource.io/gh/compSPI/reduceSPI.svg/?label=active+issues&show_trend=true&token=jLUY_liiut7OfeyiBI2ECnc9)](https://deepsource.io/gh/compSPI/reduceSPI/?ref=repository-badge)

# About

Welcome to the Python package: `reduceSPI`: Data Reduction Models for SPI

# Download

First create a conda environment with the required dependencies using the `enviroment.yml` file as follows:

    conda env create --file environment.yml

Then download:

    git clone https://github.com/compSPI/reduceSPI.git

# Contribute

We strongly recommend installing our pre-commit hook, to ensure that your code
follows our Python style guidelines. To do so, just run the following command line at the root of this repository:

    pre-commit install

With this hook, your code will be automatically formatted with our style conventions. If the pre-commit hook marks "Failed" upon commit, this means that your code was not correctly formatted, but has been re-formatted by the hook. Re-commit the result and check that the pre-commit marks "Passed".

See our [contributing](https://github.com/compspi/compspi/blob/master/docs/contributing.rst) guidelines!

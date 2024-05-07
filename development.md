# Code development of this Python package

As mentioned in 'README.md', the package manager ```conda``` is recommended for installing this package. We also use a conda environment and the pip installed by conda for the code development.

## Edit the Markdown documents

    - 'README.md'
    Run the following command to preview what an html output looks like.
    ```pandoc --toc --number-sections -f markdown -t html README.md -o README.html```
    - 'development.md'
    ```pandoc --toc --number-sections -f markdown -t html development.md -o development.html```

## Deployment of this package

### Test this package

For a local debugging, we import or install the local package by directly include the following lines at the beginning of a demo_1.py file under the "demo/" folder.
```
try:
    import ngslab as slab
except:
    srcpath = os.path.dirname(
        os.path.dirname(
            os.path.abspath(os.getcwd())
            )
        )
    sys.path.append(srcpath)
    import ngslab as slab
```

### Generate a distribution archive

Before publishing this package on public domain, such as the Python Package Index (PyPI) or Anaconda.org, we need to package this project.

A Python project can be packaged via the command ```python3 -m build```. But, before running that command, make sure to follow the tutorials and steps on how to [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

The directory structure of the "slab" project looks like as follows.
```
ngslab/
├── LICENSE
├── pyproject.toml
├── README.md
└── src/
    └── ngslab
        ├── __init__.py
        ├── model.py
        ├── ...
        └── demo/
            ├── demo_AgSiO2.py
            └── ...
├── ...
├── dist/
├── development.md
```

Firstly, one needs to choose a build backend. Here, we take ```setuptools``` as an example, which automatically includes the first three files and the ```src``` directory by the ```setuptools``` build backend. Importantly, the file 'pyproject.toml' tells build frontend tools, such as pip and build, which backend to use for the project.

An example of the 'pyproject.toml' file looks as follows (check "[Writing your pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#writing-pyproject-toml)" for more details on how to write the 'pyproject.toml' file)

```
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
[project]
name = "ngslab"
version = "1.0.0"
authors = [
  { name="shelvon", email="xiaorun.zang@outlook.com" },
  ]
description = "The package "slab" calculates waveguide modes in an unbounded multilayer planar waveguide with the FEM and the transparent boundary conditions (TBCs)."
readme = "README.md"
requires-python = ">=3.8"
license = {file = "LICENSE"}

```

Under the directory "slab", run ```python3 -m build``` to generate a source distribution (a *.whl and a *.tar.gz files under the directory "dist/").

### Uploading to a public domain

#### on TestPyPI

#### on PyPI

#### on Anaconda.org

### Installing and updating conda-build

Follow the steps "[Installing and updating conda-build](https://docs.conda.io/projects/conda-build/en/stable/install-conda-build.html)" in conda's document.

### Building the package

## TODO:
 - Change files in the Markdown format to reStructuredText format

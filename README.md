# README

## Description

Repository template that focuses on simplicity and ease of use.

This is ideal for quick projects or code publication.

The purpose of this template is to help with code quality, structure, and
reproducibility.

This template is not intended to be used as is for libraries or applications that will
be maintained over time. Several things are missing from it, like change logs,
advanced tools and coding standards (though it can be expanded for such uses).

This template creates a python package, contained in [src/](./src), that will
contain your different modules.

For more information about python packages and modules,
[Python Modules and Packages – An Introduction](https://realpython.com/python-modules-packages/).

## Initialization

Please follow these steps:

1. Set up the repository:

   - Automatic way—On the [template's GitHub page](https://github.com/RolnickLab/lab-uv-template),
     create a new repository by using the `Use this template` button, near the top right corner.
     Do not include all branches.

     - If you already have existing code, transfer it either in [src/](src) or [scripts/](scripts),
       depending on its nature
       - Modules (python code that is meant to be _imported_ in other python files) should go into the
         [src folder](src/README.md)
       - Python scripts that are meant to be executed via the command line
         should go into the [scripts folder](scripts/README.md)

   - It can also be done manually (though longer and more error-prone):

     1. Clone or download the `lab-basic-template` repository (this repository)
     2. Either start a new GitHub repository or select an existing one (the target repository)
     3. Copy the files and folders of the `lab-basic-template` repository into your target repository.
        - Do not copy the `.git` folder from the `lab-basic-template`.
        - Move your existing code
          - Modules (python code that is meant to be _imported_ in other python files) should go into the
            [src folder](src/README.md)
          - Python scripts that are meant to be executed via the command line
            should go into the [scripts folder](scripts/README.md)

2. Rename the python package (optional step)—This will allow you to use `from <package_name> import ...`
   instead of `from src import ...` :

   1. Rename [src folder](src) to your package name
      - Make sure the name is in `snake_case`, like other python modules and packages.
      - Bad examples : `my-package`, `MyPackage`, `My Package`
      - Good example : `my_package`
   2. Set the package name on line #2 of the [pyproject.toml](pyproject.toml) file by replacing `src` with the
      same package name used above.

3. Write your name on line #5 and write a short description on line #4 in [pyproject.toml](pyproject.toml)

4. Follow the rest of the instructions in this README

5. Remove this section (_Initialization_) from the README of your target repository and modify its title
   and description

**Important note**
If you are planning to use this for a new project and expect to use the DRAC cluster
as well as other clusters/locations, it is recommended to first set up your environment
on DRAC. The versions of Python libraries are often a bit behind compared to the Mila
cluster.

This will make your project more portable and will prevent many dependency management
problems while working across different clusters.

Installing this module for the first time (see [Installation](#install-package-and-dependencies))
will create the `uv.lock` file, which will set the different library versions used
by the project, and therefore help with reproducibility and reduce the classic but
annoying "but it works on my machine" situation.

However, this `uv.lock` file can be problematic when using locally compiled python
wheels.

If working on multiple different clusters, it might be better to add the `uv.lock`
file to your `.gitignore`, and manage your dependencies with either explicit versions or
capped like so : `uv add "pandas>=1.2.3,<1.3.0"`.

## Python Version

This project uses Python version 3.12.

## Build Tool

This project uses `uv` as a build tool. Using a build tool has the advantage of
streamlining script use as well as fix path issues related to imports.

To manage the python version for your environment, you can easily use `uv` directly.
See the [virtualenv](#create-projects-virtual-environment) section below, and the
[official documentation](https://docs.astral.sh/uv/concepts/python-versions/) for more info.

You are also free to use other means of installing and defining your python version,
like using available cluster modules : `module load python/3.12` (on DRAC. Mila cluster
doesn't have a python version higher than 3.10 as of this writing)

## Quick setup

How to get started:

### Create project's virtual environment

Create a virtual environment and activate it to install your dependencies:

```shell
# Create virtualenv with UV directly, specifying the wanted python version
uv venv -p 3.12

# Activate the created virtualenv
source .venv/bin/activate

# or use directly while inside the repository
uv run <command>
```

### Install package and dependencies

1. Command to install your package : `uv sync`
2. Command to initialize pre-commit in activated environment : `pre-commit install`
   - If non-activated environment : `uv run pre-commit install`

### How to use this repository

\<INSERT_YOUR_INSTRUCTIONS_HERE>

## Development

1. [Add required dependencies](./CONTRIBUTING.md#adding-dependencies)
2. Create some new modules in the [src](src) folder!

If you want to contribute to this repository, some development dependencies need to be
installed and used.

### Pre-commit

`pre-commit` is installed by default when installing the package using `uv sync`.
This is a very lightweight library. It is used for automated and low effort code
quality and code analysis.

- To create a git `pre-commit` hook, so the tool runs before each commit automatically,
  execute the following command:

  ```bash
  pre-commit install
  ```

  - This is a hands-off approach to code quality, as most of the work will be done
    automatically each time you create a commit. It will, however, force you to fix
    the remaining warnings after the automatic fixes.

- To remove the `pre-commit` hook, execute `pre-commit clean`

- To use it manually without needing to create an actual commit:

  ```bash
  pre-commit run --all-files
  ```

You can examine the configuration in the [.pre-commit-config.yaml](./.pre-commit-config.yaml)
file.

#### How to contribute

Read and follow the [Contributing guidelines](CONTRIBUTING.md)

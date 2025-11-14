# \<YOUR_PROJECT_NAME_HERE>

[Provide a brief, one-sentence description of your project here.]

______________________________________________________________________

## 🚀 Template Initialization

> **IMPORTANT:** This section is for the **initial setup** of your new repository.
> Follow these steps **once**, then **delete this entire "Template Initialization" section** from your `README.md`.

### 1. Create Your Repository

The easiest way to get started is to use the GitHub UI.

1. Navigate to the [template's GitHub page](https://github.com/RolnickLab/lab-uv-template).
2. Click the `Use this template` button (top right) and select `Create a new repository`.
3. **Do not** check the "Include all branches" box.
4. Choose a name and description for your new repository.
5. Clone your new repository (not the template) to your local machine.

<details>
<summary><b>Manual Setup (Advanced)</b></summary>

This method is longer and more error-prone, but useful if you are adding this template to an existing repository.

1. Clone or download the `lab-basic-template` repository.
2. In your target repository, copy all files and folders **except** for the `.git` folder.
3. If you have existing code, move it as follows:
   - **Modules** (Python code meant to be imported) go into the `src/` folder.
   - **Scripts** (Python files meant to be executed) go into the `scripts/` folder.

</details>

### 2. Configure Your Project

1. **Rename the Package (Optional, but Recommended):**
   This allows you to use `from <package_name> import ...` instead of `from src import ...`.

   - Rename the `src/` folder to your desired package name (e.g., `my_package`).
   - **Note:** The name *must* be in `snake_case`. (Bad: `my-package`, `MyPackage`. Good: `my_package`).
   - Open `pyproject.toml` and change line 2: `name = "src"` to `name = "my_package"`.

2. **Update Project Metadata:**

   - In `pyproject.toml`, edit line 4 (`description`) and line 5 (`authors`) to reflect your project and name.

3. **Update This README:**

   - Change the title at the top of this file (`# <YOUR_PROJECT_NAME_HERE>`) to your project's title.
   - Write a brief description in the section directly below the title.

### 3. Final Step

- **Delete this entire "Template Initialization" section.** The rest of this file will serve as the `README.md` for *your* new project.

______________________________________________________________________

## 🐍 Python Version

This project uses **Python 3.12**.

The virtual environment created by `uv venv -p 3.12` will manage this. If you use other tools (like `conda` or cluster modules), ensure you are using a compatible Python version.

## 📦 Package & Environment Management

This project uses **`uv`** for high-speed package and environment management.

`uv` handles:

- Creating the virtual environment (`.venv`).
- Resolving and installing dependencies listed in `pyproject.toml`.
- Creating a `uv.lock` file to ensure reproducible builds.
- Installing the project's own code (from the `src/` or renamed folder) as an **editable package**. This is what allows you to use project-wide imports (e.g., `from my_package.module_a import ...`) in your scripts and notebooks.

For more information, see the [official `uv` documentation](https://docs.astral.sh/uv/).

## ⚡ Quick Start

These steps are for anyone cloning this project to set it up for development.

1. **Create and Activate Virtual Environment:**
   This command creates a `.venv` folder using the Python version specified in the project.

   ```bash
   # Create virtualenv with UV, specifying the Python version
   uv venv -p 3.12

   # Activate the virtual environment
   source .venv/bin/activate

   # To deactivate, simply run: deactivate

   # or use directly while inside the repository
   uv run <command>
   ```

2. **Install Dependencies:**
   This command installs all dependencies from `pyproject.toml` and locks them using `uv.lock`. It also installs your local package (e.g., `src` or `my_package`) in editable mode.

   ```bash
   uv sync
   ```

3. **Set Up Pre-commit Hooks:**
   This will run automated code quality checks (like `ruff` and `black`) before each commit.

   ```bash
   pre-commit install
   ```

You are now ready to start development!

## 📖 Project Usage

\<INSERT_YOUR_INSTRUCTIONS_HERE>

(e.g., How to run your main scripts, what the package does, basic examples)

______________________________________________________________________

## 🌐 Environment & Portability Note

This template is designed for reproducibility using the `uv.lock` file.

**Working Across Different Clusters (e.g., DRAC, Mila):**

You may encounter dependency issues if you generate the `uv.lock` file on one machine (e.g., Mila, with newer libraries) and then try to `uv sync` on another (e.g., DRAC, which often has older system libraries).

**Recommendation:**

- **If you work on DRAC:** It is usually recommended to **first** set up your environment on DRAC, especially if you plan on using DRAC's pre-built python wheels. This ensures you are using library versions compatible with the cluster's older environment, which will also work on newer systems like Mila or your local machine.
- **If you encounter persistent issues:** As a last resort, you can add `uv.lock` to your `.gitignore` file. This is generally discouraged as it reduces reproducibility. If you do this, you must be very careful to manage your dependencies in `pyproject.toml` with explicit version ranges (e.g., `pandas>=1.2.3,<1.3.0`).

## 🛠️ Development Workflow

### Adding Dependencies

To add new dependencies, see the [Contributing guidelines](CONTRIBUTING.md#adding-dependencies).

### Pre-commit

This project uses `pre-commit` for automated code formatting and linting. The hooks are defined in `.pre-commit-config.yaml`.

- **Installation:** The `pre-commit install` command (in the [Quick Start](#quick-start)) installs git hooks that run automatically before each commit.
- **Automatic Fixes:** When you `git commit`, `pre-commit` will run. It will automatically fix many formatting issues (like `black`). If it makes changes, your commit will be aborted. Simply `git add .` the changes and commit again.
- **Manual Run:** You can run all checks on all files manually at any time:
  ```bash
  pre-commit run --all-files
  ```
- **Uninstalling:** To remove the git hooks:
  ```bash
  pre-commit uninstall
  ```

### Contributing

Please read and follow the [Contributing guidelines](CONTRIBUTING.md) for details on submitting code, running tests, and managing dependencies.

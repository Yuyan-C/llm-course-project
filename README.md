# \<YOUR_PROJECT_NAME_HERE>

[Provide a brief, one-sentence description of your project here.]

______________________________________________________________________

## 🚀 Template Initialization

> **IMPORTANT:** This section is for the **initial setup** of your new repository.
> Follow these steps **once**, then **delete this entire "Template Initialization" section** from your `README.md`.

### 1. Create Your Repository

- **Recommended:** On the [template's GitHub page](https://github.com/RolnickLab/lab-uv-template), click the `Use this template` button. Do not include all branches.

- **Manual:**

  <details>
      <summary>Click to expand for manual setup instructions</summary>

  ```
  1. Clone or download this template repository.
  2. Create or select your new target repository on GitHub.
  3. Copy all files and folders (except the `.git` folder) from the template into your target repository.
  ```

  </details>

### 2. Transfer Existing Code (If Applicable)

- **Modules** (Python code meant to be imported) go into the `src/` folder.
- **Scripts** (Python files meant to be executed) go into the `scripts/` folder.

### 3. Configure the Project

1. **Rename the Package (Optional but Recommended):** This changes the import name from `src`.
   - Rename the `src/` directory to your desired package name (e.g., `my_package`).
   - **Rule:** The name **must** be `snake_case`.
   - Update the `pyproject.toml` file. Change `name = "src"` (line 2) to `name = "my_package"`.
2. **Update Project Metadata:**
   - In `pyproject.toml`, edit the `description` (line 4) and `authors` (line 5).
3. **Review Tooling:**
   - Read the `## 📦 Package & Environment Management` section below.
   - Pay special attention to the note on `uv.lock` if you work across multiple clusters (e.g., DRAC and Mila).
   - The first time you run `uv sync`, it will create the `uv.lock` file, locking your dependency versions.

### 4. Final Cleanup

1. **Delete This Section:** Remove this entire `## ⚠️ ATTENTION` section from your new `README.md`.
2. **Update the README:** Change the title and description at the top of this file.
3. **Add Instructions:** Fill out the `## 📖 Project Usage` section with instructions for *your* project.
4. **Commit:** Commit your changes.

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

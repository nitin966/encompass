
# Contributing to EnCompass

We follow a strict **Pull Request Workflow** to ensure code quality and stability.

## Workflow

1.  **Branch**: Create a new branch for your feature or fix.
    ```bash
    git checkout -b feature/my-new-feature
    ```
    *Do NOT push directly to `main`.*

2.  **Test**: Ensure all tests pass locally.
    ```bash
    pytest tests/ -v
    ```

3.  **PR**: Open a Pull Request against `main`.

4.  **CI**: Wait for the GitHub Actions CI to pass.
    - The CI runs `pytest` on Python 3.10, 3.11, and 3.12.
    - **Merging is blocked** until CI passes.

## Branch Protection

The `main` branch is protected.
- **No direct pushes** allowed.
- **Status checks must pass** before merging.

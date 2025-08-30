# Contributing Guidelines

Thank you for your interest in contributing. This project values clarity, maintainability, and professionalism. Please review the following guidelines before submitting contributions.

---

## 1. Getting Started

* Fork the repository and create a dedicated branch for your work.
* Ensure your environment is consistent with the documented dependencies and versions.
* Run tests locally before pushing changes.

---

## 2. Commit Messages

Commit messages must be concise, meaningful, and follow a consistent style.

**Format:**

```
<type>(<scope>): <short summary>
```

**Types:**

* `feat`: A new feature
* `fix`: A bug fix
* `refactor`: Code restructuring without functional changes
* `docs`: Documentation changes only
* `style`: Non-functional code changes (formatting, naming, whitespace)
* `test`: Adding or updating tests
* `chore`: Maintenance tasks (build, CI, tooling)

**Examples:**

* `fix(progress): correct tqdm update behavior with last batch`
* `style(config): normalize docstring spacing`
* `docs(README): clarify installation steps`

For trivial changes (e.g., a typo fix, minor whitespace), it is acceptable to use a minimal message:

* `style: fix typo`
* `docs: minor edit`

---

## 3. Pull Requests

* Keep PRs focused and small. Large unrelated changes should be split into multiple PRs.
* Include a clear description of the change and its motivation.
* Reference related issues where applicable.

---

## 4. Code Style

* Follow PEP8 (for Python) or the project's defined style guide.
* Ensure code is self-explanatory, with comments only where necessary.
* Avoid redundant or unused code.
* Prefer clarity over cleverness.

---

## 5. Documentation

* Update documentation when adding or changing features.
* Keep docstrings accurate, concise, and consistent.
* Follow the NumPy or Google docstring style, depending on the project convention.

---

## 6. Tests

* All new features or bug fixes must include appropriate tests.
* Ensure all tests pass before submission.
* Aim for readability and clarity in test code.

---

## 7. Minor Changes Policy

For very small changes (indentation, whitespace, renaming a local variable), you may:

* Commit with a brief `style:` or `chore:` message.
* Combine multiple trivial edits into a single commit if they are related.

---

## 8. Communication

* Use respectful, professional language.
* Prefer clarity over ambiguity when discussing issues.
* If in doubt, open a draft PR early for feedback.

---

## 9. Release Notes

* Significant changes should include a note for maintainers to update release notes.
* Trivial fixes (typos, spacing) do not require a release note entry.

---

## 10. Final Notes

* Contributions are welcome, but consistency and maintainability come first.
* The maintainers reserve the right to request revisions or decline contributions that do not align with the project’s standards.

Thank you for helping maintain a professional and sustainable project.

# 📦 UV Package Manager Guide

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python dependency management.

## Why uv?

- ⚡ **10-100x faster** than pip for dependency resolution and installation
- 🔒 **Deterministic builds** with `uv.lock` file
- 🎯 **Better dependency resolution** avoids conflicts
- 💾 **Efficient caching** reduces disk space usage
- 🛠️ **Single tool** for virtual environments, package installation, and project management
- 🔄 **Drop-in replacement** for pip commands

## Installation

### Windows
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Linux / macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Verify Installation
```bash
uv --version
```

## Common Commands

### Project Setup

```bash
# Create virtual environment
uv venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install all dependencies from pyproject.toml
uv sync

# Install in editable mode
uv pip install -e .
```

### Managing Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a specific version
uv add "package-name>=1.0.0"

# Remove a dependency
uv remove package-name

# Update all dependencies
uv sync --upgrade

# Update a specific package
uv pip install --upgrade package-name
```

### Running Scripts

```bash
# Run Python script with project dependencies
uv run python main.py

# Run with specific Python version
uv run --python 3.12 python main.py

# Run one-off commands
uv run streamlit run frontend/app.py
```

### Working with Requirements

```bash
# Generate requirements.txt for compatibility
uv pip freeze > requirements.txt

# Install from requirements.txt
uv pip install -r requirements.txt

# Sync with lock file
uv sync
```

## Project-Specific Workflows

### First Time Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd Hospital-AI-Backend

# 2. Create virtual environment
uv venv

# 3. Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 4. Install dependencies
uv sync

# 5. Configure environment
copy .env.example .env  # Windows
cp .env.example .env    # Linux/Mac

# 6. Train models
uv run python training/clinical_training/train_clinical_model.py
uv run python training/image_training/train_image_model.py
```

### Daily Development

```bash
# Activate environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Start backend
uv run python main.py

# In another terminal - start frontend
uv run streamlit run frontend/app.py
```

### Updating Dependencies

```bash
# Check for outdated packages
uv pip list --outdated

# Update all dependencies
uv sync --upgrade

# Update specific package
uv pip install --upgrade tensorflow

# Regenerate lock file
uv lock --upgrade
```

### Troubleshooting

#### Cache Issues
```bash
# Clear uv cache
uv cache clean

# Reinstall everything
uv sync --reinstall
```

#### Dependency Conflicts
```bash
# If you see resolution errors for pandas or pillow
# This is due to streamlit compatibility requirements:
# - pandas must be <3.0 (we use 2.x)
# - pillow must be <12.0 (we use 11.x)

# These are already configured in pyproject.toml
# If issues persist, try:
uv cache clean
uv sync --reinstall
```

#### Lock File Issues
```bash
# Regenerate lock file
uv lock

# Force update lock file
uv lock --upgrade
```

#### Virtual Environment Issues
```bash
# Remove and recreate venv
rm -rf .venv  # Linux/Mac
rmdir /s .venv  # Windows

uv venv
uv sync
```

## Migration from pip

If you were using pip before:

```bash
# Old way (pip)
pip install -r requirements.txt

# New way (uv) - much faster!
uv sync
```

```bash
# Old way (pip)
pip install package-name

# New way (uv)
uv add package-name
```

## Continuous Integration

### GitHub Actions Example

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v1

- name: Setup Python
  run: uv python install 3.12

- name: Install dependencies
  run: uv sync

- name: Run tests
  run: uv run pytest
```

### Docker Example

```dockerfile
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .
CMD ["uv", "run", "python", "main.py"]
```

## Performance Comparison

| Operation | pip | uv | Speedup |
|-----------|-----|-----|---------|
| Fresh install | 45s | 2s | **22x faster** |
| Cached install | 25s | 0.5s | **50x faster** |
| Dependency resolution | 15s | 0.3s | **50x faster** |

## Advanced Features

### Lock Files

uv automatically creates `uv.lock` for reproducible builds:

```bash
# Update lock file
uv lock

# Install from lock file (CI/CD)
uv sync --frozen
```

### Python Version Management

```bash
# Install Python version
uv python install 3.12

# List available Python versions
uv python list

# Use specific Python version
uv venv --python 3.12
```

### Workspace Support

For monorepo setups:

```toml
[tool.uv.workspace]
members = ["backend", "frontend"]
```

## Best Practices

1. **Always commit `uv.lock`** for reproducible builds
2. **Use `uv sync`** instead of `uv pip install` when possible
3. **Run `uv lock --upgrade`** regularly to get security updates
4. **Use virtual environments** for isolation
5. **Keep `pyproject.toml`** as the source of truth for dependencies

## Resources

- 📚 [uv Documentation](https://github.com/astral-sh/uv)
- 🚀 [Getting Started Guide](https://docs.astral.sh/uv/getting-started/)
- 💬 [Discord Community](https://discord.gg/astral-sh)
- 🐛 [Issue Tracker](https://github.com/astral-sh/uv/issues)

## Need Help?

For project-specific issues:
1. Check [TROUBLESHOOTING.md](README.md#troubleshooting)
2. Run `uv --help` for command help
3. Visit [uv documentation](https://docs.astral.sh/uv/)

For uv-specific questions:
1. Check [uv FAQ](https://docs.astral.sh/uv/faq/)
2. Search [GitHub issues](https://github.com/astral-sh/uv/issues)
3. Ask in [Discord](https://discord.gg/astral-sh)

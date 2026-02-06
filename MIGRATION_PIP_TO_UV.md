# 🔄 Migration Guide: pip to uv

Quick guide for migrating from pip to uv package manager.

## Why Migrate?

- ⚡ **10-100x faster** installations
- 🔒 **Reproducible builds** with lock files
- 💾 **Reduced disk usage** with global cache
- 🎯 **Better dependency resolution**

## Quick Migration (In 3 Steps)

### Step 1: Install uv

```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version
```

### Step 2: Remove Old Virtual Environment

```bash
# Deactivate current environment
deactivate

# Remove old venv
rm -rf venv       # Linux/Mac
rmdir /s venv     # Windows

# OR if you used .venv
rm -rf .venv      # Linux/Mac
rmdir /s .venv    # Windows
```

### Step 3: Create New Environment with uv

```bash
# Create virtual environment
uv venv

# Activate it
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies (MUCH faster!)
uv sync
```

## Command Translation

| pip Command | uv Equivalent | Notes |
|-------------|---------------|-------|
| `pip install -r requirements.txt` | `uv sync` | Uses pyproject.toml |
| `pip install package` | `uv add package` | Adds to pyproject.toml |
| `pip install -e .` | `uv pip install -e .` | Editable install |
| `pip uninstall package` | `uv remove package` | Updates pyproject.toml |
| `pip freeze > requirements.txt` | `uv pip freeze` | For compatibility |
| `pip list` | `uv pip list` | List packages |
| `pip list --outdated` | `uv pip list --outdated` | Check updates |
| `python -m venv venv` | `uv venv` | Create environment |

## Project-Specific Migration

### Before (using pip)

```bash
# Old workflow
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### After (using uv)

```bash
# New workflow (much faster!)
uv venv
source .venv/bin/activate
uv sync
uv run python main.py  # Can skip activation
```

## Common Questions

### Q: Do I need to keep requirements.txt?

**A:** No! uv uses `pyproject.toml` as the source of truth. But you can generate one if needed:

```bash
uv pip freeze > requirements.txt
```

### Q: What about the lock file?

**A:** uv automatically creates `uv.lock`. Commit this to git for reproducible builds:

```bash
git add uv.lock
git commit -m "Add uv lock file"
```

### Q: Can I still use pip?

**A:** Yes! uv is compatible with pip. But you'll miss out on the speed benefits.

### Q: How do CI/CD workflows change?

**Before:**
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
```

**After:**
```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v1

- name: Install dependencies
  run: uv sync
```

## Troubleshooting

### Issue: "command not found: uv"

**Solution:** Add uv to PATH or restart terminal after installation.

```bash
# Check installation
which uv  # Linux/Mac
where uv  # Windows

# Reinstall if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Issue: Import errors after migration

**Solution:** Fully reinstall dependencies:

```bash
uv sync --reinstall
```

### Issue: Cache taking up space

**Solution:** Clean cache periodically:

```bash
uv cache clean
```

## Performance Comparison

Here's what to expect:

| Operation | pip Time | uv Time | Result |
|-----------|----------|---------|--------|
| Fresh install (this project) | ~45s | ~2s | ⚡ 22x faster |
| Cached install | ~25s | ~0.5s | ⚡ 50x faster |
| Adding single package | ~10s | ~0.3s | ⚡ 33x faster |

## Next Steps

1. ✅ Complete migration steps above
2. 📚 Read [UV_GUIDE.md](UV_GUIDE.md) for advanced usage
3. 🚀 Enjoy faster development!

## Need Help?

- Check [UV_GUIDE.md](UV_GUIDE.md) for detailed documentation
- Visit [uv documentation](https://docs.astral.sh/uv/)
- Ask in project's issue tracker

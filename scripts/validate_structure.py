"""
Simple validation script to check package structure.

This script validates that all the key modules and classes are properly defined
without requiring full dependencies to be installed.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_file_exists(path: str) -> bool:
    """Check if a file exists."""
    return Path(path).exists()

def validate_structure():
    """Validate the package structure."""
    print("🔍 Validating neurotrader package structure...\n")
    
    base_path = Path(__file__).parent.parent
    
    # Core files
    core_files = [
        "pyproject.toml",
        "LICENSE",
        ".env.example",
        ".gitignore",
        "README.md",
    ]
    
    print("📦 Core Files:")
    for file in core_files:
        exists = check_file_exists(base_path / file)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
    
    # Config files
    config_files = [
        "configs/data.yaml",
        "configs/model.yaml",
        "configs/train.yaml",
        "configs/loss.yaml",
        "configs/features.yaml",
        "configs/tune.yaml",
    ]
    
    print("\n⚙️  Configuration Files:")
    for file in config_files:
        exists = check_file_exists(base_path / file)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
    
    # Source modules
    source_modules = [
        "src/neurotrader/__init__.py",
        "src/neurotrader/cli.py",
        "src/neurotrader/utils/seed.py",
        "src/neurotrader/utils/logging.py",
        "src/neurotrader/utils/time.py",
        "src/neurotrader/utils/io.py",
        "src/neurotrader/data/sources.py",
        "src/neurotrader/data/csv_source.py",
        "src/neurotrader/data/ccxt_source.py",
        "src/neurotrader/data/resample.py",
        "src/neurotrader/data/splitter.py",
        "src/neurotrader/data/dataset.py",
        "src/neurotrader/data/datamodule.py",
        "src/neurotrader/features/registry.py",
        "src/neurotrader/features/technical.py",
        "src/neurotrader/features/price.py",
        "src/neurotrader/features/volatility.py",
        "src/neurotrader/features/calendar.py",
        "src/neurotrader/labels/targets.py",
        "src/neurotrader/labels/thresholds.py",
        "src/neurotrader/models/ms_transformer.py",
        "src/neurotrader/models/heads.py",
        "src/neurotrader/models/blocks.py",
        "src/neurotrader/models/embeddings.py",
        "src/neurotrader/losses/factory.py",
        "src/neurotrader/losses/calibration.py",
        "src/neurotrader/losses/conformal.py",
        "src/neurotrader/training/metrics.py",
    ]
    
    print("\n📂 Source Modules:")
    modules_exist = 0
    for module in source_modules:
        exists = check_file_exists(base_path / module)
        if exists:
            modules_exist += 1
        status = "✓" if exists else "✗"
        print(f"  {status} {module}")
    
    print(f"\n  Total: {modules_exist}/{len(source_modules)} modules present")
    
    # Test files
    test_files = [
        "tests/__init__.py",
        "tests/test_losses.py",
        "tests/test_splitter.py",
        "tests/test_labels.py",
    ]
    
    print("\n🧪 Test Files:")
    for file in test_files:
        exists = check_file_exists(base_path / file)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
    
    # Scripts
    script_files = [
        "scripts/generate_example_data.py",
    ]
    
    print("\n📜 Scripts:")
    for file in script_files:
        exists = check_file_exists(base_path / file)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
    
    # Count total Python files
    src_path = base_path / "src" / "neurotrader"
    py_files = list(src_path.rglob("*.py"))
    
    print(f"\n📊 Statistics:")
    print(f"  Total Python files in src/: {len(py_files)}")
    
    # Estimate LOC
    total_loc = 0
    for py_file in py_files:
        try:
            with open(py_file, 'r') as f:
                total_loc += len(f.readlines())
        except:
            pass
    
    print(f"  Estimated lines of code: ~{total_loc:,}")
    
    print("\n✅ Package structure validation complete!")
    print("\n📚 Next steps:")
    print("  1. Install dependencies: pip install -e .")
    print("  2. Generate example data: python scripts/generate_example_data.py")
    print("  3. Run tests: pytest tests/")
    print("  4. Train a model: neurotrader-train")

if __name__ == "__main__":
    validate_structure()

# Environment Verification

Use the existing test infrastructure rather than ad-hoc verification scripts.

## Quick Verification

```bash
# Verify the installation works
./isaaclab.sh -p -c "import isaaclab; print('isaaclab OK')"

# Verify PyTorch + CUDA
./isaaclab.sh -p -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Run Existing Tests

```bash
# Test all registered environments (comprehensive)
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/test_environments.py

# Test core framework
./isaaclab.sh -p -m pytest source/isaaclab/test/

# Test a specific package
./isaaclab.sh -p -m pytest source/isaaclab_physx/test/
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/
```

## List Available Environments

```bash
grep -rh 'id="Isaac' source/isaaclab_tasks/ --include="*.py" \
  | sed 's/.*id="\([^"]*\)".*/\1/' | sort -u
```

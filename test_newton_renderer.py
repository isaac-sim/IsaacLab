#!/usr/bin/env python3
"""Test Newton Warp renderer initialization."""

import sys

def test_imports():
    """Test that all Newton renderer imports work."""
    print("Testing imports...")
    
    try:
        import newton
        print(f"✓ Newton {newton.__version__} imported")
    except ImportError as e:
        print(f"✗ Failed to import Newton: {e}")
        return False
    
    try:
        from isaaclab.renderer import get_renderer_class, NewtonWarpRendererCfg
        print("✓ Renderer imports successful")
    except ImportError as e:
        print(f"✗ Failed to import renderer: {e}")
        return False
    
    try:
        from isaaclab.sim._impl.newton_manager import NewtonManager
        print("✓ NewtonManager import successful")
    except ImportError as e:
        print(f"✗ Failed to import NewtonManager: {e}")
        return False
    
    return True


def test_renderer_registry():
    """Test renderer registry."""
    print("\nTesting renderer registry...")
    
    from isaaclab.renderer import get_renderer_class
    
    renderer_cls = get_renderer_class("newton_warp")
    if renderer_cls is None:
        print("✗ Failed to get Newton Warp renderer class")
        return False
    
    print(f"✓ Got renderer class: {renderer_cls.__name__}")
    return True


def test_renderer_config():
    """Test renderer configuration."""
    print("\nTesting renderer configuration...")
    
    from isaaclab.renderer import NewtonWarpRendererCfg
    
    try:
        cfg = NewtonWarpRendererCfg(
            width=64,
            height=64,
            num_cameras=1,
            num_envs=1,
            data_types=["rgb", "depth"]
        )
        print(f"✓ Created renderer config: {cfg.renderer_type}")
        return True
    except Exception as e:
        print(f"✗ Failed to create config: {e}")
        return False


def test_newton_manager_class():
    """Test NewtonManager class structure."""
    print("\nTesting NewtonManager class...")
    
    from isaaclab.sim._impl.newton_manager import NewtonManager
    
    # Check methods exist
    required_methods = ['initialize', 'get_model', 'get_state_0', 'update_state_from_usdrt', 'reset']
    
    for method_name in required_methods:
        if not hasattr(NewtonManager, method_name):
            print(f"✗ Missing method: {method_name}")
            return False
    
    print(f"✓ NewtonManager has all required methods")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Newton Warp Renderer Integration Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Renderer Registry", test_renderer_registry),
        ("Renderer Config", test_renderer_config),
        ("NewtonManager Class", test_newton_manager_class),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

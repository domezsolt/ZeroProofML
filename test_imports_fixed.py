"""Test that import errors are fixed."""

try:
    print("Testing core imports...")
    from zeroproof.core import real, TRTag
    print("✓ Core imports successful")
    
    print("Testing autodiff imports...")
    from zeroproof.autodiff import TRNode
    print("✓ Autodiff imports successful")
    
    print("Testing layers imports...")
    from zeroproof.layers import TRRational, MonomialBasis
    print("✓ Layers imports successful")
    
    print("Testing training imports...")
    from zeroproof.training import Optimizer
    print("✓ Training imports successful")
    
    print("Testing utils imports...")
    from zeroproof.utils.metrics import PoleLocation
    from zeroproof.utils.logging import StructuredLogger
    print("✓ Utils imports successful")
    
    # Test plotting with graceful fallback
    try:
        from zeroproof.utils.plotting import TrainingCurvePlotter, MATPLOTLIB_AVAILABLE
        print(f"✓ Plotting imports successful (matplotlib available: {MATPLOTLIB_AVAILABLE})")
    except ImportError as e:
        print(f"✓ Plotting imports handled gracefully: {e}")
    
    print("Testing new enhanced layers...")
    from zeroproof.layers import FullyIntegratedRational, PoleAwareRational
    print("✓ Enhanced layers imports successful")
    
    print("Testing new training features...")
    from zeroproof.training import HybridTRTrainer, HybridTrainingConfig
    print("✓ Enhanced training imports successful")
    
    print("\n🎉 ALL IMPORTS SUCCESSFUL!")
    print("The import errors have been fixed.")
    
except ImportError as e:
    print(f"❌ Import error still exists: {e}")
    import traceback
    traceback.print_exc()

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

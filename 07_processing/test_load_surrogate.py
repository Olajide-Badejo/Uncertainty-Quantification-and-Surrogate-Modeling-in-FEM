from uncertainty_quantification_07 import load_surrogate, Config
import traceback

try:
    _ = load_surrogate(Config())
    print("SURROGATE_LOAD_OK")
except Exception as e:
    print("SURROGATE_LOAD_ERROR", type(e).__name__, str(e))
    traceback.print_exc()
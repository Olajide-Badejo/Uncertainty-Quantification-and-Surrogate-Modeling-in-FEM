from sensitivity_analysis_08 import load_simple_surrogate, Config
import traceback

try:
    surrogate = load_simple_surrogate(Config())
    print("SENSITIVITY_SURROGATE_OK")
except Exception as e:
    print("SENSITIVITY_SURROGATE_ERROR", type(e).__name__, str(e))
    traceback.print_exc()
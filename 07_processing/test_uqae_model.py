from uncertainty_quantification_07 import UQAESurrogateModel
import traceback

try:
    m = UQAESurrogateModel(r".")
    print("UQ_AE_MODEL_OK")
except Exception as e:
    print("UQ_AE_MODEL_ERROR", type(e).__name__, str(e))
    traceback.print_exc()
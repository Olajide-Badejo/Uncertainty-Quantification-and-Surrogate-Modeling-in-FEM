from sensitivity_analysis_08 import run_sobol_analysis, Config
import json

c = Config()
from sensitivity_analysis_08 import load_simple_surrogate
m = load_simple_surrogate(c)
res = run_sobol_analysis(m, c)
output_data = {
    'parameters': res['parameters'],
    'qoi_list': c.QOIS,
    'sobol_analysis': res['sobol_analysis']
}

# Try dumping to JSON to verify serializability
print('TRY_SERIALIZE')
json.dumps(output_data) 
print('SERIALIZE_OK')
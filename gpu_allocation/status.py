"""Report progress metadata without opening intermediate test outcomes."""
from pathlib import Path
import json
import time

P = Path(__file__).resolve().parent / 'www_short/wsdm_recenter_20261116/owned_r18_gpu_allocation_factorial'
R = P / 'run_v1'
fits = list((R / 'fits').glob('*/RESULT.json'))
selected = list((R / 'fits').glob('*/SELECTION.json'))
started = list((R / 'fits').glob('*/INPUTS.json'))
failed = list((R / 'claims').glob('*FAILURE*')) + list((R / 'eligible_claims').glob('*FAILURE*'))
print(json.dumps({'checked_unix': time.time(), 'planned_fits': 45,
    'started_fits': len(started), 'selected_checkpoints': len(selected),
    'completed_fits': len(fits),
    'eligible_evaluated_checkpoints': len(list((R / 'fits').glob('*/eligible_eval/RESULT.json'))),
    'failed_attempts': len(failed), 'pipeline_complete': (P / 'PIPELINE_COMPLETE.json').exists(),
    'pipeline_failure': (P / 'PIPELINE_FAILURE.json').exists(),
    'numerical_results_opened': False}, indent=2))

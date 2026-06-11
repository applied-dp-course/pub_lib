from typing import Dict, TypeAlias, Any
import numpy as np

LabeledData: TypeAlias = Dict[str, np.ndarray[Any, np.dtype[np.float64]]]

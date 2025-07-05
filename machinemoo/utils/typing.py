import pandas as pd
import numpy as np
import numpy.typing as npt
import typing_extensions
from scipy.sparse import spmatrix

from machinemoo.scalarization.scalarization_interface import scalar_interface, w_interface, single_interface

MatrixLike: typing_extensions.TypeAlias = npt.NDArray | pd.DataFrame | spmatrix
scalar: typing_extensions.TypeAlias = scalar_interface | single_interface | w_interface

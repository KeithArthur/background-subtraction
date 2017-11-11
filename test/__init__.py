import sys
sys.path.append('./src')
import warnings
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix, csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
warnings.simplefilter('ignore', SparseEfficiencyWarning)

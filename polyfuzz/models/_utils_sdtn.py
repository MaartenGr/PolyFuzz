import sys
import importlib.util
from scipy.sparse import csr_matrix

from typing import Optional

_HAVE_SPARSE_DOT = importlib.util.find_spec("sparse_dot_topn") is not None
if _HAVE_SPARSE_DOT:
    if sys.version_info >= (3, 8):
        from sparse_dot_topn import sp_matmul_topn
    else:
        from sparse_dot_topn import awesome_cossim_topn

        def sp_matmul_topn(
            A: csr_matrix,
            B: csr_matrix,
            top_n: int,
            threshold: float,
            sort: bool = True,
            n_threads: Optional[int] = None,
        ):
            n_threads = n_threads or 1
            use_threads = n_threads > 1
            return awesome_cossim_topn(
                A,
                B.T,
                ntop=max(top_n, 2),
                lower_bound=threshold,
                use_threads=use_threads,
                n_jobs=n_threads,
            )
else:

    def sp_matmul_topn(*args, **kwargs):
        raise NotImplementedError(
            "`sp_matmul_topn` requires `sparse_dot_topn` be installed"
        )


__all__ = ["sp_matmul_topn"]

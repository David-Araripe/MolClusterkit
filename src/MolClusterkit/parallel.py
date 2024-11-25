"""Module for parallel processing of functions with joblib and tqdm for the progress bar"""

import contextlib
from functools import partial
from math import ceil
from typing import Any, Callable, Iterable

import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

from .logger import logger


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    Source:
    https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#comment95750316_49950707
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class ParallelApplier:
    def __init__(
        self,
        func: Callable,
        iterable: Iterable,
        show_progress: bool = True,
        n_jobs: int = 8,
        backend: str = "loky",
        chunk_size: int = None,
    ):
        """
        Initialize the parallel applier.

        Args:
            func: Function to apply to each element
            iterable: Input data to process
            show_progress: Whether to show progress bars
            n_jobs: Number of parallel jobs (-1 for all cores)
            backend: Parallelization backend ('loky', 'threading', or 'multiprocessing')
            chunk_size: Size of chunks to process (if None, calculated automatically)
        """
        self.func = self._set_func(func)
        self.iterable = list(iterable)  # Convert to list for length calculation
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs > 0 else None  # None means all cores in joblib
        self.backend = self._set_backend(backend)
        self.total_items = len(self.iterable)
        self.chunk_size = self._set_chunk_size(chunk_size)
        logger.debug(f"Chunk size: {self.chunk_size}")

        if self.total_items == 0:
            raise ValueError("Empty iterable provided.")

    def _set_chunk_size(self, chunk_size: int) -> int:
        if self.total_items <= self.n_jobs:
            return 1
        elif chunk_size is None:
            return ceil(self.total_items / self.n_jobs)
        return chunk_size

    def _set_func(self, func):
        """Set the function to be parallelized.

        Args:
            func: Function to apply to each element

        Raises:
            TypeError: if the provided input is not a callable

        Returns:
            Callable: The function to apply
        """
        if not callable(func):
            raise TypeError("func should be a callable function.")
        if hasattr(func, "__name__"):
            if func.__name__ == "<lambda>":
                raise TypeError("parallel_applier does not support lambda functions. ")
            else:
                func_name = func.__name__
        else:
            if isinstance(func, partial):
                func_name = func.func.__name__
            else:
                func_name = "function"
        self.func_name = func_name
        return func

    def _set_backend(self, backend: str) -> str:
        """Set the parallelization backend.

        Args:
            backend: Parallelization backend ('loky', 'threading', or 'multiprocessing')

        Raises:
            ValueError: If an invalid backend is provided

        Returns:
            str: Selected backend
        """
        if backend not in ["loky", "threading", "multiprocessing"]:
            raise ValueError(
                "Invalid backend. Choose from 'loky', 'threading', 'multiprocessing'."
            )
        return backend

    def _make_chunks(self) -> list[list[Any]]:
        chunks = []
        for i in range(0, self.total_items, self.chunk_size):
            chunk = self.iterable[i : i + self.chunk_size]
            chunks.append(chunk)
        self.n_chunks = len(chunks)
        return chunks

    def _process_chunk(self, chunk: list[Any], **kwargs) -> list[Any]:
        """
        Process a single chunk with progress bar.

        Args:
            chunk (list[Any]): Chunk of data to process
            **kwargs: Additional arguments to pass to the function

        Returns:
            list[Any]: Processed results
        """
        return [self.func(item, **kwargs) for item in chunk]

    def __call__(self, **kwargs) -> list[Any]:
        """
        Apply the function to all items in parallel.

        Args:
            **kwargs: Additional arguments to pass to the function

        Returns:
            list[Any]: list of results
        """
        chunks = self._make_chunks()

        with tqdm_joblib(
            tqdm(
                total=self.n_chunks,
                desc=f"Applying {self.func_name} to chunks",
                unit="chunk",
                position=0,
                leave=True,
            )
        ) as progress_bar:  # noqa: F841
            results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(self._process_chunk)(chunk, **kwargs) for chunk in chunks
            )

        return [item for chunk_result in results for item in chunk_result]

from math import ceil
from typing import Any, Callable, Iterable

from joblib import Parallel, delayed
from tqdm import tqdm


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
        self.func = func
        self.iterable = list(iterable)  # Convert to list for length calculation
        self.show_progress = show_progress
        self.n_jobs = n_jobs if n_jobs > 0 else None  # None means all cores in joblib
        self.backend = backend
        self.total_items = len(self.iterable)

        # Calculate chunk size if not provided
        if chunk_size is None:
            # Aim for at least 4 items per chunk, but no more than 100 chunks
            suggested_chunks = min(self.total_items // 4, 100)
            self.chunk_size = ceil(self.total_items / suggested_chunks)
        else:
            self.chunk_size = chunk_size

    def _set_tqdm_fmt(self, level: str) -> dict:
        """
        Set format for different progress bar levels.

        Args:
            level (str): Either 'chunk' or 'item'

        Returns:
            dict: tqdm formatting parameters
        """
        if level == "chunk":
            return {
                "desc": "Processing chunks",
                "unit": "chunk",
                "position": 0,
                "leave": True,
            }
        else:  # item level
            return {
                "desc": "Processing items",
                "unit": "item",
                "position": 1,
                "leave": True,
            }

    def _make_chunks(self) -> list[list[Any]]:
        """
        Split the iterable into chunks.

        Returns:
            list[list[Any]]: list of chunks
        """
        chunks = []
        for i in range(0, self.total_items, self.chunk_size):
            chunk = self.iterable[i : i + self.chunk_size]
            chunks.append(chunk)
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
        if self.show_progress:
            chunk_iter = tqdm(chunk, **self._set_tqdm_fmt("item"))
        else:
            chunk_iter = chunk

        return [self.func(item, **kwargs) for item in chunk_iter]

    def __call__(self, **kwargs) -> list[Any]:
        """
        Apply the function to all items in parallel.

        Args:
            **kwargs: Additional arguments to pass to the function

        Returns:
            list[Any]: list of results
        """
        chunks = self._make_chunks()

        parallel = Parallel(n_jobs=self.n_jobs, backend=self.backend)

        if self.show_progress:  # create progress bar for chunks
            chunk_iter = tqdm(chunks, **self._set_tqdm_fmt("chunk"))
        else:
            chunk_iter = chunks

        results = parallel(  # process each chunk in parallel
            delayed(self._process_chunk)(chunk, **kwargs) for chunk in chunk_iter
        )
        return [item for chunk_result in results for item in chunk_result]

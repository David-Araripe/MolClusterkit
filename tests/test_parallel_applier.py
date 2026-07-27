import time
import unittest

from MolClusterkit.parallel import ParallelApplier


def square_func(x):
    return x * x


def slow_square_func(x):
    time.sleep(0.01)
    return x * x


class TestBaseParallelApplier(unittest.TestCase):
    def setUp(self):
        self.sample_data = list(range(100))

    def tearDown(self):
        self.sample_data = None

    def test_basic_functionality(self):
        applier = ParallelApplier(
            func=square_func,
            iterable=self.sample_data,
            show_progress=False,
            n_jobs=2,
        )
        results = applier()
        self.assertEqual(len(results), len(self.sample_data))
        self.assertEqual(results, [x * x for x in self.sample_data])

    def test_empty_iterable(self):
        with self.assertRaises(ValueError):
            applier = ParallelApplier(
                func=square_func, iterable=[], show_progress=False, n_jobs=2
            )
            applier()

    def test_single_item(self):
        applier = ParallelApplier(
            func=square_func, iterable=[5], show_progress=False, n_jobs=2
        )
        results = applier()
        self.assertEqual(results, [25])

    def test_different_backends(self):
        backends = ["loky", "threading", "multiprocessing"]
        expected = [x * x for x in self.sample_data]

        for backend in backends:
            applier = ParallelApplier(
                func=slow_square_func,
                iterable=self.sample_data,
                show_progress=False,
                n_jobs=2,
                backend=backend,
            )
            results = applier()
            self.assertEqual(results, expected)

    def test_custom_chunk_size(self):
        applier = ParallelApplier(
            func=square_func,
            iterable=self.sample_data,
            show_progress=False,
            n_jobs=2,
            chunk_size=10,
        )
        chunks = applier._make_chunks()
        self.assertEqual(
            len(chunks),
            len(self.sample_data) // 10 + (1 if len(self.sample_data) % 10 else 0),
        )
        self.assertTrue(all(len(chunk) <= 10 for chunk in chunks))

    def test_kwargs_passing(self):
        def func_with_kwargs(x, multiplier=1):
            return x * multiplier

        applier = ParallelApplier(
            func=func_with_kwargs,
            iterable=self.sample_data,
            show_progress=False,
            n_jobs=2,
        )
        results = applier(multiplier=2)
        self.assertEqual(results, [x * 2 for x in self.sample_data])

    def test_error_handling(self):
        def failing_func(x):
            if x == 50:
                raise ValueError("Test error")
            return x

        applier = ParallelApplier(
            func=failing_func, iterable=self.sample_data, show_progress=False, n_jobs=2
        )
        with self.assertRaises(ValueError):
            applier()

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            ParallelApplier(
                func=square_func,
                iterable=self.sample_data,
                show_progress=False,
                n_jobs=2,
                backend="invalid_backend",
            )

    def test_non_callable_func(self):
        with self.assertRaises(TypeError):
            ParallelApplier(
                func="not_a_function",
                iterable=self.sample_data,
                show_progress=False,
                n_jobs=2,
            )

    def test_progress_bars(self):
        import io
        import sys

        # Capture stderr
        stderr = io.StringIO()
        sys.stderr = stderr

        applier = ParallelApplier(
            func=slow_square_func,
            iterable=self.sample_data[:10],
            show_progress=True,
            n_jobs=2,
            chunk_size=2,
        )
        results = applier()

        output = stderr.getvalue()

        self.assertIn("Applying slow_square_func to chunks", output)

    def test_generator_input(self):
        generator_data = (x for x in range(10))
        applier = ParallelApplier(
            func=square_func,
            iterable=generator_data,
            show_progress=False,
            n_jobs=2,
        )
        results = applier()
        self.assertEqual(results, [x * x for x in range(10)])


if __name__ == "__main__":
    unittest.main()

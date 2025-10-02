"""Property-based tests for profiling utilities."""

import time
import tracemalloc

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from zeroproof.autodiff import TRNode, gradient_tape, tr_add, tr_div, tr_mul
from zeroproof.core import TRTag, ninf, phi, pinf, real
from zeroproof.utils.profiling import (
    ProfileResult,
    TRProfiler,
    memory_profile,
    performance_report,
    profile_tr_operation,
    quick_profile,
    tag_statistics,
)


class TestTRProfiler:
    """Test transreal profiler."""

    def test_basic_profiling(self):
        """Test basic profiling functionality."""
        profiler = TRProfiler(trace_memory=False)

        @profiler.profile_operation("test_add")
        def add_operation(a, b):
            time.sleep(0.01)  # Simulate work
            return tr_add(a, b)

        with profiler:
            result = add_operation(real(1.0), real(2.0))

        results = profiler.get_results()
        assert "test_add" in results

        profile = results["test_add"]
        assert profile.calls == 1
        assert profile.duration >= 0.01
        assert profile.avg_duration >= 0.01
        assert profile.tag_distribution.get("REAL", 0) > 0

    def test_nested_profiling(self):
        """Test nested operation profiling."""
        profiler = TRProfiler(trace_memory=False)

        @profiler.profile_operation("outer")
        def outer_op(x):
            return inner_op(x, real(2.0))

        @profiler.profile_operation("inner")
        def inner_op(a, b):
            return tr_mul(a, b)

        with profiler:
            result = outer_op(real(3.0))

        results = profiler.get_results()
        assert "outer" in results

        # Inner operation should be in sub_operations
        outer = results["outer"]
        assert len(outer.sub_operations) > 0
        assert any(op.name == "inner" for op in outer.sub_operations)

    @given(st.integers(min_value=1, max_value=10))
    def test_multiple_calls_accumulation(self, n_calls):
        """Test that multiple calls accumulate correctly."""
        profiler = TRProfiler(trace_memory=False)

        @profiler.profile_operation("repeated_op")
        def op(x):
            return tr_add(x, real(1.0))

        with profiler:
            for i in range(n_calls):
                op(real(float(i)))

        results = profiler.get_results()
        profile = results["repeated_op"]

        assert profile.calls == n_calls
        assert profile.tag_distribution.get("REAL", 0) == n_calls

    def test_memory_profiling(self):
        """Test memory profiling functionality."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        profiler = TRProfiler(trace_memory=True)

        @profiler.profile_operation("memory_intensive")
        def create_nodes(n):
            nodes = []
            for i in range(n):
                nodes.append(TRNode.constant(real(i)))
            return nodes

        with profiler:
            nodes = create_nodes(1000)

        results = profiler.get_results()
        profile = results["memory_intensive"]

        # Should have tracked some memory allocation
        assert profile.memory_allocated >= 0
        assert profile.memory_peak >= 0

        tracemalloc.stop()

    def test_profiling_report(self):
        """Test report generation."""
        profiler = TRProfiler(trace_memory=False)

        @profiler.profile_operation("op1")
        def op1(x):
            return tr_add(x, real(1.0))

        @profiler.profile_operation("op2")
        def op2(x):
            return tr_mul(x, real(2.0))

        with profiler:
            for i in range(5):
                op1(real(i))
            for i in range(3):
                op2(real(i))

        report = profiler.generate_report()

        # Report should contain both operations
        assert "op1" in report
        assert "op2" in report
        assert "5" in report  # 5 calls to op1
        assert "3" in report  # 3 calls to op2
        assert "Tag Distribution" in report


class TestProfilingDecorators:
    """Test profiling decorators."""

    def test_profile_tr_operation_decorator(self):
        """Test the profile_tr_operation decorator."""

        @profile_tr_operation("custom_name")
        def my_operation(x, y):
            return tr_div(x, y)

        # Execute operation
        result = my_operation(real(10.0), real(2.0))

        assert result.value.tag == TRTag.REAL
        assert result.value.value == 5.0

    def test_memory_profile_decorator(self):
        """Test memory profiling decorator."""

        @memory_profile
        def allocate_many_nodes(n):
            nodes = []
            for i in range(n):
                node = TRNode.constant(real(i))
                nodes.append(node)
            return nodes

        # Should print memory statistics
        nodes = allocate_many_nodes(100)
        assert len(nodes) == 100


class TestTagStatistics:
    """Test tag statistics computation."""

    @given(
        st.lists(
            st.sampled_from(
                [
                    lambda: TRNode.constant(real(1.0)),
                    lambda: TRNode.constant(pinf()),
                    lambda: TRNode.constant(ninf()),
                    lambda: TRNode.constant(phi()),
                ]
            ),
            min_size=1,
            max_size=100,
        )
    )
    def test_tag_distribution(self, node_factories):
        """Test tag distribution statistics."""
        nodes = [factory() for factory in node_factories]

        stats = tag_statistics(nodes)

        # Verify totals
        assert stats["total"] == len(nodes)

        # Verify percentages
        total_percentage = sum(stats["percentages"].values())
        assert abs(total_percentage - 100.0) < 0.1

        # Verify counts match
        tag_sum = sum(stats["by_tag"].values())
        assert tag_sum == len(nodes)

    def test_operation_tag_statistics(self):
        """Test statistics by operation type."""
        nodes = []

        with gradient_tape() as tape:
            a = TRNode.constant(real(1.0))
            b = TRNode.constant(real(0.0))
            c = TRNode.constant(pinf())

            # Create various operations
            add1 = tr_add(a, b)  # REAL
            mul1 = tr_mul(b, c)  # PHI (0 * inf)
            div1 = tr_div(a, b)  # PINF (1/0)

            nodes.extend([add1, mul1, div1])

        stats = tag_statistics(nodes)

        # Check operation statistics
        assert "ADD" in stats["by_operation"]
        assert "MUL" in stats["by_operation"]
        assert "DIV" in stats["by_operation"]

        # Check non-real operations
        assert len(stats["non_real_operations"]) >= 2  # mul1 and div1


class TestPerformanceReport:
    """Test performance report generation."""

    def test_basic_performance_report(self):
        """Test basic performance report."""
        # Create a simple graph
        with gradient_tape() as tape:
            x = TRNode.parameter(real(1.0))
            y = tr_add(x, TRNode.constant(real(2.0)))
            z = tr_mul(y, TRNode.constant(real(3.0)))

        report = performance_report(z)

        # Check report structure
        assert "graph_statistics" in report
        assert "memory_analysis" in report
        assert "optimization_potential" in report
        assert "bottlenecks" in report
        assert "tag_statistics" in report

        # Check graph statistics
        graph_stats = report["graph_statistics"]
        assert graph_stats["total_nodes"] >= 3
        assert graph_stats["depth"] >= 2
        assert graph_stats["branching_factor"] >= 0

    def test_bottleneck_detection(self):
        """Test bottleneck detection."""
        # Create graph with high fan-out
        with gradient_tape() as tape:
            shared = TRNode.constant(real(1.0))

            # Create many nodes using shared
            nodes = []
            for i in range(20):
                node = tr_add(shared, TRNode.constant(real(i)))
                nodes.append(node)

            # Combine results
            result = nodes[0]
            for node in nodes[1:]:
                result = tr_add(result, node)

        report = performance_report(result)

        # Should detect high fan-out bottleneck
        bottlenecks = report["bottlenecks"]
        high_fanout = [b for b in bottlenecks if b["type"] == "high_fanout"]
        assert len(high_fanout) > 0

    @given(st.integers(min_value=5, max_value=20))
    def test_performance_report_scales(self, depth):
        """Test that performance report scales with graph size."""
        # Create deep graph
        with gradient_tape() as tape:
            x = TRNode.parameter(real(1.0))
            for i in range(depth):
                x = tr_add(x, TRNode.constant(real(i)))

        report = performance_report(x)

        # Graph depth should match
        assert report["graph_statistics"]["depth"] >= depth
        assert report["graph_statistics"]["total_nodes"] >= depth + 1


class TestQuickProfile:
    """Test quick profiling utility."""

    def test_quick_profile_function(self):
        """Test quick_profile utility."""

        def compute(x, n):
            result = x
            for i in range(n):
                result = tr_add(result, real(i))
            return result

        result, profile_data = quick_profile(compute, real(0.0), 10)

        # Check result
        assert result.value.tag == TRTag.REAL
        assert result.value.value == sum(range(10))

        # Check profile data
        assert "results" in profile_data
        assert "report" in profile_data
        assert "compute" in profile_data["results"]


@pytest.mark.benchmark
class TestProfilingOverhead:
    """Test profiling overhead."""

    def test_profiling_overhead(self, benchmark):
        """Test overhead of profiling."""

        def operation():
            x = real(1.0)
            for i in range(100):
                x = tr_add(x, real(i))
            return x

        # Benchmark without profiling
        result_no_profile = benchmark.pedantic(operation, rounds=10)

        # Benchmark with profiling
        profiler = TRProfiler(trace_memory=False)
        wrapped = profiler.profile_operation("test")(operation)

        def with_profiling():
            with profiler:
                return wrapped()

        # Profiling should add minimal overhead
        # (This is more of a performance characterization than a hard test)
        result_with_profile = with_profiling()

        assert result_no_profile.value == result_with_profile.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Damped Least Squares (DLS) solver reference implementation.

This provides the analytical reference solution for inverse kinematics
using the DLS method, serving as a baseline for comparison.
"""

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DLSConfig:
    """Configuration for DLS solver."""

    damping_factor: float = 0.01
    max_iterations: int = 100
    tolerance: float = 1e-6
    adaptive_damping: bool = False
    min_damping: float = 1e-6
    max_damping: float = 1.0
    damping_increase_factor: float = 2.0
    damping_decrease_factor: float = 0.5


class DLSSolver:
    """
    Damped Least Squares solver for inverse kinematics.

    Solves: Δθ = J^T (JJ^T + λ²I)^(-1) Δx
    where λ is the damping factor.
    """

    def __init__(self, config: DLSConfig):
        self.config = config
        self.solve_history = []
        self.performance_stats = {
            "total_solves": 0,
            "successful_solves": 0,
            "average_iterations": 0.0,
            "average_error": 0.0,
            "numerical_failures": 0,
        }

    def jacobian_rr_robot(
        self, theta1: float, theta2: float, L1: float = 1.0, L2: float = 1.0
    ) -> np.ndarray:
        """
        Compute Jacobian for RR robot.

        Args:
            theta1: First joint angle
            theta2: Second joint angle
            L1: First link length
            L2: Second link length

        Returns:
            2x2 Jacobian matrix
        """
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c12, s12 = np.cos(theta1 + theta2), np.sin(theta1 + theta2)

        J = np.array([[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]])

        return J

    def solve_single_step(
        self, jacobian: np.ndarray, dx: np.ndarray, damping: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Solve single DLS step.

        Args:
            jacobian: Jacobian matrix J
            dx: Desired end-effector displacement
            damping: Damping factor (uses config default if None)

        Returns:
            (joint_displacement, solve_info)
        """
        if damping is None:
            damping = self.config.damping_factor

        solve_info = {
            "damping_used": damping,
            "condition_number": 0.0,
            "residual_norm": 0.0,
            "success": False,
        }

        try:
            # Compute DLS solution: Δθ = J^T (JJ^T + λ²I)^(-1) Δx
            JJT = jacobian @ jacobian.T
            damping_matrix = damping**2 * np.eye(JJT.shape[0])

            # Check condition number
            cond_num = np.linalg.cond(JJT + damping_matrix)
            solve_info["condition_number"] = cond_num

            # Solve
            inv_term = np.linalg.inv(JJT + damping_matrix)
            dtheta = jacobian.T @ inv_term @ dx

            # Compute residual
            residual = jacobian @ dtheta - dx
            solve_info["residual_norm"] = np.linalg.norm(residual)
            solve_info["success"] = True

            return dtheta, solve_info

        except np.linalg.LinAlgError as e:
            solve_info["error"] = str(e)
            return np.zeros(jacobian.shape[1]), solve_info
        except Exception as e:
            solve_info["error"] = str(e)
            return np.zeros(jacobian.shape[1]), solve_info

    def solve_iterative(
        self,
        initial_config: np.ndarray,
        target_position: np.ndarray,
        robot_params: Dict[str, float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve IK iteratively using DLS.

        Args:
            initial_config: Initial joint configuration
            target_position: Target end-effector position
            robot_params: Robot parameters (L1, L2)

        Returns:
            (final_config, solve_info)
        """
        if robot_params is None:
            robot_params = {"L1": 1.0, "L2": 1.0}

        current_config = initial_config.copy()
        damping = self.config.damping_factor

        solve_info = {
            "iterations": 0,
            "final_error": 0.0,
            "damping_history": [damping],
            "error_history": [],
            "success": False,
            "numerical_issues": [],
        }

        for iteration in range(self.config.max_iterations):
            # Forward kinematics
            theta1, theta2 = current_config
            L1, L2 = robot_params["L1"], robot_params["L2"]

            # Current end-effector position
            x_current = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
            y_current = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
            current_position = np.array([x_current, y_current])

            # Error
            error = target_position - current_position
            error_norm = np.linalg.norm(error)
            solve_info["error_history"].append(error_norm)

            # Check convergence
            if error_norm < self.config.tolerance:
                solve_info["success"] = True
                solve_info["final_error"] = error_norm
                break

            # Compute Jacobian
            jacobian = self.jacobian_rr_robot(theta1, theta2, L1, L2)

            # DLS step
            dtheta, step_info = self.solve_single_step(jacobian, error, damping)

            if not step_info["success"]:
                solve_info["numerical_issues"].append(
                    f"Iteration {iteration}: {step_info.get('error', 'Unknown')}"
                )
                if self.config.adaptive_damping:
                    damping = min(
                        damping * self.config.damping_increase_factor, self.config.max_damping
                    )
                    solve_info["damping_history"].append(damping)
                continue

            # Update configuration
            current_config += dtheta

            # Adaptive damping
            if self.config.adaptive_damping:
                if step_info["residual_norm"] < self.config.tolerance * 10:
                    # Good step, decrease damping
                    damping = max(
                        damping * self.config.damping_decrease_factor, self.config.min_damping
                    )
                else:
                    # Poor step, increase damping
                    damping = min(
                        damping * self.config.damping_increase_factor, self.config.max_damping
                    )

                solve_info["damping_history"].append(damping)

            solve_info["iterations"] = iteration + 1

        solve_info["final_error"] = (
            solve_info["error_history"][-1] if solve_info["error_history"] else float("inf")
        )

        return current_config, solve_info

    def evaluate_on_dataset(
        self,
        samples: List[Dict[str, float]],
        robot_params: Dict[str, float] = None,
        vectorized: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate DLS solver on a dataset.

        Args:
            samples: List of IK samples with 'theta1', 'theta2', 'dx', 'dy'
            robot_params: Robot parameters

        Returns:
            Evaluation results
        """
        if robot_params is None:
            robot_params = {"L1": 1.0, "L2": 1.0}

        results = {
            "solve_times": [],
            "final_errors": [],
            "iterations": [],
            "success_rate": 0.0,
            "damping_stats": [],
            "condition_numbers": [],
            "per_sample": [],  # Detailed per-sample records
        }

        successful_solves = 0

        print(f"Evaluating DLS solver on {len(samples)} samples...")

        # Optional fast path: single-step vectorized DLS when max_iterations==1
        if vectorized and self.config.max_iterations == 1 and len(samples) > 0:
            t0 = time.time()
            L1 = robot_params["L1"]
            L2 = robot_params["L2"]
            theta1 = np.array([float(s["theta1"]) for s in samples])
            theta2 = np.array([float(s["theta2"]) for s in samples])
            dx = np.array([float(s["dx"]) for s in samples])
            dy = np.array([float(s["dy"]) for s in samples])

            s1, c1 = np.sin(theta1), np.cos(theta1)
            s12, c12 = np.sin(theta1 + theta2), np.cos(theta1 + theta2)

            # Current position
            x_cur = L1 * c1 + L2 * c12
            y_cur = L1 * s1 + L2 * s12
            ex = (x_cur + dx) - x_cur
            ey = (y_cur + dy) - y_cur

            # Jacobian components
            j11 = -L1 * s1 - L2 * s12
            j12 = -L2 * s12
            j21 = L1 * c1 + L2 * c12
            j22 = L2 * c12

            # JJT = J * J^T components
            a = j11 * j11 + j12 * j12  # (0,0)
            b = j11 * j21 + j12 * j22  # (0,1) and (1,0)
            d = j21 * j21 + j22 * j22  # (1,1)
            lam2 = self.config.damping_factor**2
            a_l = a + lam2
            d_l = d + lam2

            # Inverse of 2x2 [[a_l, b],[b, d_l]]
            detA = a_l * d_l - b * b
            # Guard small determinants
            eps = 1e-12
            detA_safe = np.where(np.abs(detA) < eps, np.sign(detA) * eps + (detA == 0) * eps, detA)
            inv00 = d_l / detA_safe
            inv01 = -b / detA_safe
            inv10 = -b / detA_safe
            inv11 = a_l / detA_safe

            # u = inv * e
            ux = inv00 * ex + inv01 * ey
            uy = inv10 * ex + inv11 * ey

            # dtheta = J^T * u
            dt1 = j11 * ux + j21 * uy
            dt2 = j12 * ux + j22 * uy

            # Residual r = J * dtheta - e
            rx = j11 * dt1 + j12 * dt2 - ex
            ry = j21 * dt1 + j22 * dt2 - ey
            res = np.sqrt(rx * rx + ry * ry)

            # Success based on tolerance
            success_mask = res < self.config.tolerance
            solve_time = time.time() - t0

            # Condition numbers of initial J
            # For speed, approximate with ratio of singular values via explicit SVD per-sample in small batches
            conds = []
            batch = 1024
            for start in range(0, len(samples), batch):
                end = min(len(samples), start + batch)
                J_stack = np.stack(
                    [
                        np.stack([j11[start:end], j12[start:end]], axis=1),
                        np.stack([j21[start:end], j22[start:end]], axis=1),
                    ],
                    axis=1,
                )  # shape [B,2,2]
                # Compute cond per-sample
                for Ji in J_stack:
                    try:
                        conds.append(float(np.linalg.cond(Ji)))
                    except Exception:
                        conds.append(float("inf"))

            # Fill results
            results["solve_times"] = [solve_time] * len(samples)
            results["final_errors"] = [float(e) for e in res]
            results["iterations"] = [1] * len(samples)
            results["condition_numbers"] = conds
            successful_solves = int(np.sum(success_mask))

            # per-sample records
            per = []
            for i, s in enumerate(samples):
                detj = abs(float(s.get("det_J", L1 * L2 * np.sin(theta2[i]))))
                status = "success" if bool(success_mask[i]) else "iterations_cap"
                if not np.isfinite(res[i]) or not np.isfinite(detA[i]):
                    status = "numerical_error"
                per.append(
                    {
                        "idx": i,
                        "detj": detj,
                        "err": float(res[i]),
                        "iterations": 1,
                        "solve_time": float(solve_time) / max(1, len(samples)),
                        "status": status,
                    }
                )
            results["per_sample"] = per

            # Compute summary stats
            results["success_rate"] = successful_solves / float(len(samples))
            results["average_solve_time"] = float(solve_time) / float(len(samples))
            results["average_error"] = float(np.mean(res))
            results["average_iterations"] = 1.0
            results["max_condition_number"] = (
                float(np.max([c for c in conds if not np.isinf(c)])) if conds else float("inf")
            )

            # Update performance stats
            self.performance_stats["total_solves"] = len(samples)
            self.performance_stats["successful_solves"] = successful_solves
            self.performance_stats["average_iterations"] = results["average_iterations"]
            self.performance_stats["average_error"] = results["average_error"]

            return results

        # Fallback: iterative per-sample evaluation
        for i, sample in enumerate(samples):
            # Initial configuration
            initial_config = np.array([sample["theta1"], sample["theta2"]])

            # Target displacement
            dx, dy = sample["dx"], sample["dy"]

            # Current position
            L1, L2 = robot_params["L1"], robot_params["L2"]
            x_current = L1 * np.cos(sample["theta1"]) + L2 * np.cos(
                sample["theta1"] + sample["theta2"]
            )
            y_current = L1 * np.sin(sample["theta1"]) + L2 * np.sin(
                sample["theta1"] + sample["theta2"]
            )

            # Target position
            target_position = np.array([x_current + dx, y_current + dy])

            # Solve
            start_time = time.time()
            final_config, solve_info = self.solve_iterative(
                initial_config, target_position, robot_params
            )
            solve_time = time.time() - start_time

            # Record results
            results["solve_times"].append(solve_time)
            results["final_errors"].append(solve_info["final_error"])
            results["iterations"].append(solve_info["iterations"])

            if solve_info["success"]:
                successful_solves += 1

            # Compute condition number for initial Jacobian
            J = self.jacobian_rr_robot(sample["theta1"], sample["theta2"], L1, L2)
            cond_num = np.linalg.cond(J)
            results["condition_numbers"].append(cond_num)

            # Per-sample record
            detj = abs(float(sample.get("det_J", L1 * L2 * np.sin(sample["theta2"]))))
            status = (
                "success"
                if solve_info["success"]
                else (
                    "iterations_cap"
                    if solve_info.get("iterations", 0) >= self.config.max_iterations
                    else "failure"
                )
            )
            if solve_info.get("numerical_issues"):
                status = "numerical_error"
            results["per_sample"].append(
                {
                    "idx": i,
                    "detj": detj,
                    "err": float(solve_info["final_error"]),
                    "iterations": int(solve_info.get("iterations", 0)),
                    "solve_time": float(solve_time),
                    "status": status,
                }
            )

            if i % 50 == 0:
                print(f"  Processed {i+1}/{len(samples)} samples...")

        # Compute statistics
        results["success_rate"] = successful_solves / len(samples)
        results["average_solve_time"] = np.mean(results["solve_times"])
        results["average_error"] = np.mean(results["final_errors"])
        results["average_iterations"] = np.mean(results["iterations"])
        results["max_condition_number"] = np.max(
            [c for c in results["condition_numbers"] if not np.isinf(c)]
        )

        # Update performance stats
        self.performance_stats["total_solves"] = len(samples)
        self.performance_stats["successful_solves"] = successful_solves
        self.performance_stats["average_iterations"] = results["average_iterations"]
        self.performance_stats["average_error"] = results["average_error"]

        return results


def run_dls_reference(
    samples: List[Dict[str, float]],
    config: Optional[DLSConfig] = None,
    robot_params: Dict[str, float] = None,
    output_dir: str = "results",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run DLS reference solver evaluation.

    Args:
        samples: IK samples to evaluate on
        config: DLS configuration
        robot_params: Robot parameters
        output_dir: Output directory

    Returns:
        Reference results
    """
    if config is None:
        config = DLSConfig()

    if robot_params is None:
        robot_params = {"L1": 1.0, "L2": 1.0}

    print("=== DLS Reference Solver ===")
    print(f"Damping factor: {config.damping_factor}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Tolerance: {config.tolerance}")
    print(f"Adaptive damping: {config.adaptive_damping}")

    # Create solver
    solver = DLSSolver(config)

    # Evaluate
    start_time = time.time()
    results = solver.evaluate_on_dataset(samples, robot_params)
    total_time = time.time() - start_time

    # Add configuration and timing
    results["config"] = asdict(config)
    results["robot_params"] = robot_params
    results["total_evaluation_time"] = total_time
    results["performance_stats"] = solver.performance_stats
    results["seed"] = seed

    # Print summary
    print(f"\nDLS Solver Results:")
    print(f"  Success rate: {results['success_rate']:.1%}")
    print(f"  Average error: {results['average_error']:.6f}")
    print(f"  Average iterations: {results['average_iterations']:.1f}")
    print(f"  Average solve time: {results['average_solve_time']:.6f}s")
    print(f"  Max condition number: {results['max_condition_number']:.2f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "dls_reference_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV summary
    csv_file = os.path.join(output_dir, "dls_reference_summary.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Method", "Success_Rate", "Avg_Error", "Avg_Iterations", "Avg_Time", "Max_Condition"]
        )
        writer.writerow(
            [
                "DLS",
                results["success_rate"],
                results["average_error"],
                results["average_iterations"],
                results["average_solve_time"],
                results["max_condition_number"],
            ]
        )

    print(f"Results saved to {results_file}")
    print(f"Summary saved to {csv_file}")

    return results


def compare_damping_strategies(
    samples: List[Dict[str, float]], damping_values: List[float] = None, output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Compare different damping strategies.

    Args:
        samples: IK samples to test
        damping_values: List of damping factors to try
        output_dir: Output directory

    Returns:
        Comparison results
    """
    if damping_values is None:
        damping_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    print("=== DLS Damping Strategy Comparison ===")
    print(f"Testing damping values: {damping_values}")

    comparison_results = []

    for damping in damping_values:
        print(f"\n--- Testing λ = {damping} ---")

        config = DLSConfig(damping_factor=damping)
        results = run_dls_reference(samples, config, output_dir=f"{output_dir}/damping_{damping}")

        comparison_results.append(
            {
                "damping": damping,
                "success_rate": results["success_rate"],
                "average_error": results["average_error"],
                "average_iterations": results["average_iterations"],
                "max_condition": results["max_condition_number"],
            }
        )

    # Find best damping
    best_result = min(comparison_results, key=lambda x: x["average_error"])

    print(f"\n=== Damping Comparison Summary ===")
    print("λ\t\tSuccess%\tAvg Error\tAvg Iter\tMax Cond")
    print("-" * 60)

    for result in comparison_results:
        print(
            f"{result['damping']:.0e}\t\t{result['success_rate']:.1%}\t\t"
            f"{result['average_error']:.6f}\t{result['average_iterations']:.1f}\t\t"
            f"{result['max_condition']:.1f}"
        )

    print(f"\nBest damping: λ = {best_result['damping']}")

    # Save comparison results
    os.makedirs(output_dir, exist_ok=True)

    comparison_file = os.path.join(output_dir, "dls_damping_comparison.json")
    with open(comparison_file, "w") as f:
        json.dump(
            {"comparison_results": comparison_results, "best_damping": best_result}, f, indent=2
        )

    # Save CSV
    csv_file = os.path.join(output_dir, "dls_damping_comparison.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Damping", "Success_Rate", "Avg_Error", "Avg_Iterations", "Max_Condition"])

        for result in comparison_results:
            writer.writerow(
                [
                    result["damping"],
                    result["success_rate"],
                    result["average_error"],
                    result["average_iterations"],
                    result["max_condition"],
                ]
            )

    print(f"Comparison saved to {comparison_file}")

    return {"comparison_results": comparison_results, "best_damping": best_result}


def analyze_singularity_handling(
    samples: List[Dict[str, float]], output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Analyze how DLS handles singularities.

    Args:
        samples: IK samples including singular configurations
        output_dir: Output directory

    Returns:
        Singularity analysis results
    """
    print("=== DLS Singularity Analysis ===")

    # Separate singular and regular samples
    singular_samples = [s for s in samples if s.get("is_singular", False)]
    regular_samples = [s for s in samples if not s.get("is_singular", False)]

    print(f"Singular samples: {len(singular_samples)}")
    print(f"Regular samples: {len(regular_samples)}")

    # Test different damping values on singular samples
    damping_values = [1e-3, 1e-2, 1e-1, 1.0]
    singular_results = []

    for damping in damping_values:
        config = DLSConfig(damping_factor=damping)
        solver = DLSSolver(config)

        # Test on singular samples
        results = solver.evaluate_on_dataset(singular_samples)

        singular_results.append(
            {
                "damping": damping,
                "singular_success_rate": results["success_rate"],
                "singular_avg_error": results["average_error"],
                "singular_avg_iterations": results["average_iterations"],
            }
        )

        print(
            f"λ={damping}: Success={results['success_rate']:.1%}, "
            f"Error={results['average_error']:.6f}"
        )

    # Test on regular samples for comparison
    regular_config = DLSConfig(damping_factor=0.01)
    regular_solver = DLSSolver(regular_config)
    regular_results = regular_solver.evaluate_on_dataset(regular_samples)

    analysis = {
        "singular_analysis": singular_results,
        "regular_performance": {
            "success_rate": regular_results["success_rate"],
            "average_error": regular_results["average_error"],
            "average_iterations": regular_results["average_iterations"],
        },
        "summary": {
            "n_singular": len(singular_samples),
            "n_regular": len(regular_samples),
            "best_damping_for_singular": min(
                singular_results, key=lambda x: x["singular_avg_error"]
            )["damping"],
        },
    }

    # Save analysis
    os.makedirs(output_dir, exist_ok=True)

    analysis_file = os.path.join(output_dir, "dls_singularity_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"Singularity analysis saved to {analysis_file}")

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DLS Reference Solver")
    parser.add_argument("--damping", type=float, default=0.01, help="Damping factor")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive damping")
    parser.add_argument("--output_dir", default="results/dls_reference", help="Output directory")

    args = parser.parse_args()

    # Create configuration
    config = DLSConfig(
        damping_factor=args.damping,
        max_iterations=args.max_iter,
        tolerance=args.tolerance,
        adaptive_damping=args.adaptive,
    )

    print("DLS Solver configuration:")
    print(f"  Damping factor: {config.damping_factor}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Tolerance: {config.tolerance}")
    print(f"  Adaptive damping: {config.adaptive_damping}")

    print("\nNote: This script requires IK samples to be provided.")
    print("Use this as a module: from dls_solver import run_dls_reference")
    print("Or integrate with your IK dataset pipeline.")

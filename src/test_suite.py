from dataclasses import dataclass
import importlib
import importlib.util
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_correctness(
    reference_output: any, candidate_output: any
) -> tuple[bool, str | None]:
    if isinstance(reference_output, torch.Tensor) and isinstance(
        candidate_output, torch.Tensor
    ):
        if torch.allclose(reference_output, candidate_output):
            return True, None
        else:
            return (
                False,
                f"Reference output: {reference_output}, Candidate output: {candidate_output}",
            )
    elif isinstance(reference_output, list) and isinstance(candidate_output, list):
        if len(reference_output) != len(candidate_output):
            return (
                False,
                f"Expected {len(reference_output)} outputs, got {len(candidate_output)}.",
            )

        for ref_item, cand_item in zip(reference_output, candidate_output):
            is_correct, error = _check_correctness(ref_item, cand_item)
            if not is_correct:
                return False, error
        return True, None
    else:
        return (
            False,
            f"Expected {type(reference_output)} output, got {type(candidate_output)}.",
        )


@dataclass(frozen=True)
class Reference:
    model: torch.nn.Module
    get_inputs: callable


@dataclass(frozen=True)
class Result:
    error: str | None = None
    is_correct: bool = False
    speedup: float = 0.0


def get_reference(module_name) -> Reference:
    spec = importlib.util.spec_from_file_location("reference_module", module_name)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)

    return Reference(
        model=getattr(reference_module, "Model")(),
        get_inputs=getattr(reference_module, "get_inputs"),
    )


def load_candidate_from_file(file_path: str) -> torch.nn.Module:
    spec = importlib.util.spec_from_file_location("candidate_module", file_path)
    candidate_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate_module)

    if not hasattr(candidate_module, "NewModel"):
        raise RuntimeError("Candidate module must define a 'NewModel' class")

    return candidate_module.NewModel()


def run_tests(reference: Reference, candidate: torch.nn.Module) -> list[Result]:
    results = []

    for _ in range(10):
        seed = random.randint(0, 1000000)
        torch.manual_seed(seed)

        inputs = reference.get_inputs()

        reference_output = reference.model(*inputs)
        candidate_output = candidate(*inputs)

        is_correct, error = _check_correctness(reference_output, candidate_output)

        results.append(Result(error=error, is_correct=is_correct, speedup=1.0))

    return results


def get_results(reference: Reference, candidate: torch.nn.Module) -> list[Result]:
    results = run_tests(reference, candidate)

    return results


def main():
    if len(sys.argv) < 3:
        raise ValueError(
            "Candidate file path and reference function name must be provided"
        )

    candidate_file_path = sys.argv[1]
    reference_file_path = sys.argv[2]

    reference = get_reference(reference_file_path)
    candidate = load_candidate_from_file(candidate_file_path)

    results = get_results(reference, candidate)

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.is_correct)

    print(f"Tests passed: {passed_tests}/{total_tests}")

    if passed_tests < total_tests:
        print("Failed tests:")
        for i, result in enumerate(results):
            if not result.is_correct:
                print(f"  Test {i + 1}: {result.error}")


if __name__ == "__main__":
    main()

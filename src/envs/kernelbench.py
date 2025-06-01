from copy import deepcopy
import subprocess
import tempfile
import os
import sys
from typing import Any

import datasets
from verifiers.parsers import XMLParser
from verifiers.envs.code_env import CodeEnv
from verifiers.utils.data_utils import format_prompt


class KernelBenchEnv(CodeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dataset = datasets.load_dataset("ScalingIntelligence/KernelBench")

        self.eval_dataset = dataset.map(
            lambda x: {
                "code": x["code"],
                "prompt": format_prompt(
                    x["code"], system_prompt=self.system_prompt, few_shot=self.few_shot
                ),
                "problem_name": x["name"],
            }
        )
        self.eval_dataset = self.eval_dataset["level_1"].select(range(1))
        self.parser = XMLParser(fields=["answer"])

    def env_response(
        self, messages: list[dict[str, str]], reference_code: str = "", **kwargs: Any
    ) -> dict[str, str]:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid code field (not just None from failed parsing)
            if hasattr(parsed, "code") and parsed.code is not None:
                output = self.run_code(reference_code, parsed.code)
                if len(output.strip()) > 0:
                    return {
                        "role": "user",
                        "content": self.env_parser.format(output=output),
                    }
                else:
                    return {
                        "role": "user",
                        "content": "Error: Code execution returned empty output.",
                    }
        except Exception:
            pass
        return {
            "role": "user",
            "content": "Error: Code not found or invalid XML format. Please ensure correct formatting.",
        }

    def run_code(self, reference_code: str, code: str) -> str:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()
                temp_file_path = f.name

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(reference_code)
                f.flush()
                reference_file_path = f.name

            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "src/test_suite.py",
                        temp_file_path,
                        reference_file_path,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=60,
                )
                if result.stderr:
                    return f"Error: {result.stderr.strip()}"
                return result.stdout.strip() if result.stdout else ""
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)
                os.unlink(reference_file_path)
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out after 10 seconds"

    def step_api(
        self,
        client: Any,
        model: str,
        messages: list[dict[str, str]],
        sampling_args: dict[str, Any] = {},
        reference_code: str = "",
        **kwargs: Any,
    ) -> tuple[list[dict[str, str]], bool]:
        """
        Execute a single step using OpenAI API, including environment response if needed.

        Args:
            client: OpenAI client instance
            messages: Conversation history
            model: Model name to use
            **kwargs: Additional arguments for the chat completion API

        Returns:
            Updated messages list with assistant response and possibly environment response
        """
        messages_copy = deepcopy(messages)

        try:
            # Get assistant response
            response = client.chat.completions.create(
                model=model, messages=messages_copy, extra_body=sampling_args
            )

            # Add assistant response to messages
            assistant_msg = {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
            messages_copy.append(assistant_msg)

            # Check if we're done
            if self.is_completed(messages_copy):
                rollout_is_completed = True
            else:
                rollout_is_completed = False
                # If not done, get and add environment response
                env_msg = self.env_response(
                    messages_copy, reference_code=reference_code
                )
                messages_copy.append(env_msg)

            return messages_copy, rollout_is_completed

        except Exception as e:
            # Handle errors by adding error message and returning
            error_msg = {"role": "assistant", "content": f"Error in API call: {str(e)}"}
            messages_copy.append(error_msg)
            return messages_copy, True

    def eval_api(
        self,
        client: Any,
        model: str,
        max_concurrent: int = 32,
        timeout: int = 60,
        sampling_args: dict[str, Any] = {},
        **kwargs: Any,
    ):
        eval_sampling_args = deepcopy(self.sampling_args)
        eval_sampling_args.update(sampling_args)
        """
        Evaluate model using OpenAI API with proper concurrency.
        
        Args:
            client: OpenAI client instance
            model: Model name as string
            max_concurrent: Maximum number of concurrent API calls
            timeout: Maximum seconds to wait for each example
            sampling_args: Arguments specific to sampling (separate from env sampling_args)
            **kwargs: Additional arguments for evaluation
        
        Returns:
            Tuple of (eval_dataset, rewards)
        """

        def run_evaluation():
            # Import libraries here to avoid requiring them for normal operation
            import asyncio
            from asyncio import Semaphore

            # Get the evaluation dataset
            if self.eval_dataset is None:
                self.eval_dataset = self.get_eval_dataset(**kwargs)

            if self.eval_dataset is None:
                raise ValueError("Failed to load evaluation dataset")

            eval_dataset = self.eval_dataset

            async def process_example(example, semaphore):
                async with semaphore:
                    prompt = example["prompt"]
                    messages = deepcopy(example["prompt"])
                    reference_code = example["code"]
                    problem_name = example["problem_name"]

                    initial_length = len(messages)

                    for _ in range(self.max_steps):
                        try:
                            step_result = (
                                await asyncio.get_event_loop().run_in_executor(
                                    None,
                                    lambda: self.step_api(
                                        client=client,
                                        model=model,
                                        messages=messages,
                                        sampling_args=eval_sampling_args,
                                        reference_code=reference_code,
                                    ),
                                )
                            )

                            # Unpack the step_api result
                            messages, is_completed = step_result

                            # If the rollout is completed, break the loop
                            if is_completed:
                                break

                        except Exception as e:
                            print(
                                f"Error processing example {example.get('id', 'unknown')}: {str(e)}"
                            )
                            break

                    # Extract only the interaction part (not system/few-shot)
                    completions = messages[initial_length:]
                    answer = self.parser.parse(completions[-1]["content"])

                    return {
                        "prompt": prompt,
                        "completions": completions,
                        "problem_name": problem_name,
                        "answer": answer,
                    }

            async def run_all_examples():
                # Create semaphore for concurrency control
                from tqdm.asyncio import tqdm_asyncio

                semaphore = Semaphore(max_concurrent)

                # Process all examples concurrently
                tasks = [
                    process_example(example, semaphore) for example in eval_dataset
                ]
                results = await tqdm_asyncio.gather(
                    *tasks,
                    total=len(eval_dataset),
                    desc=f"Evaluating {len(eval_dataset)} examples",
                )

                return results

            # Run the async evaluation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(run_all_examples())
            finally:
                loop.close()

            results_prompt = [result["prompt"] for result in results]
            results_completions = [result["completions"] for result in results]
            results_answers = [result["answer"].__dict__ for result in results]
            results = {
                "prompt": results_prompt,
                "completions": results_completions,
                "answers": results_answers,
            }

            return results

        return run_evaluation()

    def get_reward_funcs(self, *args, **kwargs):
        return [lambda *args, **kwargs: [1.0, 1.0]]

    def get_reward_weights(self, *args, **kwargs):
        return [1.0]

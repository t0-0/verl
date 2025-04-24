# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import signal
import subprocess

# from sys import settrace
# from time import monotonic


class TimeoutError(Exception):
    pass


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = (
                final_answer.split("#### ")[1].replace(",", "").replace("$", "")
            )
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def try_float(s):
    try:
        f = float(s)
    except Exception:
        f = None
    return f


def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.0, score=1.0
):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    # start = monotonic()

    # def _trace(frame, event, arg):
    #     if monotonic() - start > 60:
    #         print(solution_str, flush=True)
    #         raise TimeoutError("Execution exceeded the time limit!")
    #     return _trace

    # settrace(_trace)
    def timeout_handler(signum, frame):
        # print(solution_str, flush=True)
        raise TimeoutError("Execution exceeded the time limit!")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)
    try:
        # if "import" not in solution_str:
        #    result = subprocess.run(
        #        [
        #            "python",
        #            "-c",
        #            f"def simple_math_problem() -> int:\n{solution_str}\nassert simple_math_problem()=={ground_truth}",
        #        ],
        #        timeout=60,
        #    )
        #    if result.returncode == 0:
        #        ret = score
        #    else:
        #        ret = -1
        if "import" not in solution_str:
            exec(
                f"def simple_math_problem() -> int:\n{solution_str}\nassert simple_math_problem()=={ground_truth}"
            )
            # exec(
            #    f"""
            #     import signal

            #     def _timeout_handler(signum, frame):
            #         raise ScriptTimeoutError("Script execution timed out")

            #     signal.signal(signal.SIGALRM, _timeout_handler)
            #     signal.alarm({60})

            #     def simple_math_problem() -> int:
            #        import time
            #        time.sleep(120)
            #        print("NG", flush=True)

            #     assert simple_math_problem() == {ground_truth}

            #     signal.alarm(0)
            #     """
            # )
            ret = score
        else:
            ret = -1
    except Exception as e:
        ret = -1
    finally:
        #    settrace(None)
        signal.alarm(0)
    if ret == -1:
        answer = extract_solution(solution_str=solution_str, method="strict")
        answer2 = extract_solution(solution_str=solution_str, method="flexible")
        if answer is None and answer2 is None:
            ret = 0
        else:
            float_answer = try_float(answer)
            float_answer2 = try_float(answer2)
            float_ground_truth = try_float(ground_truth)
            if answer == ground_truth or answer2 == ground_truth:
                ret = score
            elif (
                float_answer is not None
                and float_ground_truth is not None
                and abs(float_answer - float_ground_truth) < 1e-5
            ):
                ret = score
            elif (
                float_answer2 is not None
                and float_ground_truth is not None
                and abs(float_answer2 - float_ground_truth) < 1e-5
            ):
                ret = score
            else:
                ret = format_score
    return ret

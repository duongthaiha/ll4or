#!/usr/bin/env python3
"""
ORLM-LLaMA-3-8B inference using Apple MLX.

Drop-in replacement for scripts/inference.py that runs on Apple Silicon
via mlx-lm instead of vLLM (CUDA).

Usage:
    source ~/ll4or-mlx-env/bin/activate
    python scripts/inference_mlx.py \
        --model_path ~/ll4or-mlx-env/models/ORLM-LLaMA-3-8B-4bit
"""

import argparse
import sys
import time

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

TEMPLATE_q2mc_en = (
    "Below is an operations research question. Build a mathematical model "
    "and corresponding python code using `coptpy` that appropriately "
    "addresses the question.\n\n"
    "# Question:\n{Question}\n\n# Response:\n"
)

ONE_QUESTION = (
    "A lab has 1000 units of medicinal ingredients to make two pills, "
    "a large pill and a small pill. A large pill requires 3 units of "
    "medicinal ingredients and 2 units of filler. A small pill requires "
    "2 units of medicinal ingredients and 1 unit of filler. The lab has "
    "to make at least 100 large pills. However, since small pills are "
    "more popular at least 60% of the total number of pills must be small. "
    "How many of each should be made to minimize the total number of filler "
    "material needed?"
)


def run_inference(args):
    print(f"Loading model from: {args.model_path}")
    t0 = time.time()
    model, tokenizer = load(args.model_path)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Build prompt
    if args.question:
        question = args.question
    elif args.question_file:
        with open(args.question_file) as f:
            question = f.read().strip()
    else:
        question = ONE_QUESTION

    prompt = TEMPLATE_q2mc_en.replace("{Question}", question.strip())

    # Decoding parameters
    if args.decoding_method == "greedy":
        temp, top_p = 0.0, 1.0
    else:
        temp, top_p = args.temperature, args.top_p

    sampler = make_sampler(temp=temp, top_p=top_p)

    print(f"Decoding: {args.decoding_method} (temp={temp}, top_p={top_p})")
    print("-" * 20 + " prompt " + "-" * 20)
    print(prompt)
    print("-" * 20 + " generating " + "-" * 20)

    t0 = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        verbose=True,
    )
    elapsed = time.time() - t0

    print("-" * 20 + " completion " + "-" * 20)
    print(response)
    print("-" * 80)
    print(f"Generation completed in {elapsed:.1f}s")

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(response)
        print(f"Output saved to {args.output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ORLM inference on Apple Silicon via MLX"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="~/ll4or-mlx-env/models/ORLM-LLaMA-3-8B-4bit",
        help="Path to the MLX-converted model directory",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="OR question to solve (inline text)",
    )
    parser.add_argument(
        "--question_file",
        type=str,
        default=None,
        help="Path to a text file containing the OR question",
    )
    parser.add_argument(
        "--decoding_method",
        type=str,
        default="greedy",
        choices=["greedy", "sampling"],
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save the generated response",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)

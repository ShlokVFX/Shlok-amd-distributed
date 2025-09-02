#!/usr/bin/env python3
import argparse
import os
import re
import textwrap

import torch


def parse_line(line: str):
    parts = [p.strip() for p in line.split(";") if p.strip()]
    args = {}
    for p in parts:
        k, v = [x.strip() for x in p.split(":", 1)]
        try:
            args[k] = int(v)
        except Exception:
            args[k] = v
    return args


def load_tests(file_path):
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith('#')]
    return [parse_line(l) for l in lines]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tests-file', default='local_tests_2gpus.txt')
    p.add_argument('--index', type=int, default=3, help='0-based index into the tests file')
    p.add_argument('--rank', type=int, default=0, help='rank to profile (device index)')
    p.add_argument('--output', default=None, help='optional file to save profiler table')
    args = p.parse_args()

    tests = load_tests(args.tests_file)
    if args.index < 0 or args.index >= len(tests):
        print('index out of range')
        return 2

    test_args = tests[args.index]

    # import local modules
    import reference
    from submission import custom_kernel

    # generate input for rank
    gen_args = dict(test_args)
    # ensure rank and world_size exist
    if 'world_size' not in gen_args:
        gen_args['world_size'] = 1
    gen = reference.generate_input(**gen_args, rank=args.rank, world_size=gen_args['world_size']) if False else None

    # reference.generate_input signature differs in file; call directly by position
    # signature: generate_input(num_experts, experts_per_token, hidden_dim, max_num_tokens, seed, rank, world_size)
    cfg, rank_data, rank, world_size = reference.generate_input(
        gen_args['num_experts'], gen_args['experts_per_token'], gen_args['hidden_dim'], gen_args['max_num_tokens'], gen_args['seed'], args.rank, gen_args.get('world_size', 1)
    )

    # run profiler
    from torch.profiler import profile, record_function, ProfilerActivity

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        with record_function('submission_kernel'):
            out = custom_kernel((cfg, rank_data, args.rank, gen_args.get('world_size', 1)))
        torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=50)
    print('\nProfiler table:\n')
    print(table)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(table)


if __name__ == '__main__':
    main()

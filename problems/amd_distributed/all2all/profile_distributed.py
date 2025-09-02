#!/usr/bin/env python3
import argparse
import base64
import multiprocessing
import os
import sys
from multiprocessing import Process

def _worker(rank, world_size, test_args, return_pipe):
    try:
        import torch
        import torch.distributed as dist
        from torch.profiler import profile, ProfilerActivity, record_function
        # local imports
        import reference
        from submission import custom_kernel

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12356'

        # init
        dist.init_process_group('nccl', init_method='env://', rank=rank, world_size=world_size, device_id=torch.device(f'cuda:{rank}'))
        torch.cuda.set_device(rank)

        # generate input for this rank
        cfg, rank_data, r, ws = reference.generate_input(
            test_args['num_experts'], test_args['experts_per_token'], test_args['hidden_dim'], test_args['max_num_tokens'], test_args['seed'], rank, world_size
        )

        # warmup
        torch.cuda.synchronize()
        _ = custom_kernel((cfg, rank_data, rank, world_size))
        torch.cuda.synchronize()

        # only record on rank 0 to keep output small
        if rank == 0:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with record_function('submission_kernel'):
                    _ = custom_kernel((cfg, rank_data, rank, world_size))
                torch.cuda.synchronize()
            table = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=50)
            # send table back through pipe
            return_pipe.send(table)
        else:
            # just run once for profiling synchronization
            _ = custom_kernel((cfg, rank_data, rank, world_size))
            torch.cuda.synchronize()
            return_pipe.send(None)

    except Exception as e:
        try:
            return_pipe.send(f'ERROR: {e}')
        except Exception:
            pass
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tests-file', default='local_tests_2gpus.txt')
    p.add_argument('--index', type=int, default=3)
    p.add_argument('--world-size', type=int, default=2)
    args = p.parse_args()

    # read tests file
    lines = [l.strip() for l in open(args.tests_file).read().splitlines() if l.strip() and not l.strip().startswith('#')]
    if args.index < 0 or args.index >= len(lines):
        print('index out of range')
        return 2
    # parse key: value; semicolon-separated
    def parse_line(line):
        parts = [p.strip() for p in line.split(';') if p.strip()]
        d = {}
        for p in parts:
            k, v = [x.strip() for x in p.split(':', 1)]
            try:
                d[k] = int(v)
            except Exception:
                d[k] = v
        return d

    test_args = parse_line(lines[args.index])

    # spawn processes
    procs = []
    parent_conns = []
    for rank in range(args.world_size):
        parent_conn, child_conn = multiprocessing.Pipe()
        p = Process(target=_worker, args=(rank, args.world_size, test_args, child_conn))
        p.start()
        procs.append((p, parent_conn))

    # collect outputs
    tables = []
    for p, conn in procs:
        try:
            res = conn.recv()
        except EOFError:
            res = None
        tables.append(res)

    for p, _ in procs:
        p.join()

    # print rank0 table
    if tables and tables[0]:
        print('\nProfiler table (rank 0):\n')
        print(tables[0])
    else:
        print('No profiler table returned; output from ranks:')
        for i, t in enumerate(tables):
            print(f'rank {i}:', t)


if __name__ == '__main__':
    main()

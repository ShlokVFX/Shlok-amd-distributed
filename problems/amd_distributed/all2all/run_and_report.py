#!/usr/bin/env python3
import os
import platform
import subprocess
import sys
import re
from shutil import which

try:
    import torch
except Exception:
    torch = None


def system_info():
    info = {}
    info['platform'] = platform.platform()
    info['cpu'] = platform.processor() or platform.node()
    info['python'] = platform.python_version()
    info['torch'] = getattr(torch, '__version__', 'not-installed')
    # GPUs
    gpus = []
    if torch is not None and torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                gpus.append(torch.cuda.get_device_name(i))
        except Exception:
            gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    info['gpus'] = gpus
    return info


def run_eval(mode, tests_file, n_gpus):
    env = os.environ.copy()
    env['POPCORN_FD'] = '1'
    env['POPCORN_GPUS'] = str(n_gpus)
    cmd = [sys.executable, 'eval.py', mode, tests_file]
    proc = subprocess.run(cmd, cwd=os.path.dirname(__file__), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def parse_popcorn_output(output):
    tests = {}
    benchmarks = {}
    other = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        # expected format: key: value
        parts = line.split(': ', 1)
        if len(parts) != 2:
            other.append(line)
            continue
        key, val = parts
        if key.startswith('test.'):
            # e.g. test.0.spec or test.0.status
            m = re.match(r'test\.(\d+)\.(.+)', key)
            if m:
                idx = int(m.group(1))
                field = m.group(2)
                tests.setdefault(idx, {})[field] = val
            else:
                other.append(line)
        elif key.startswith('benchmark.'):
            # e.g. benchmark.0.mean
            m = re.match(r'benchmark\.(\d+)\.(.+)', key)
            if m:
                idx = int(m.group(1))
                field = m.group(2)
                try:
                    v = float(val)
                except Exception:
                    v = val
                benchmarks.setdefault(idx, {})[field] = v
            else:
                other.append(line)
        else:
            other.append(line)

    return tests, benchmarks, other


def print_report(info, tests, benchmarks):
    print('Running on:')
    print(f"GPU: {', '.join(info['gpus']) if info['gpus'] else 'None detected'}")
    print(f"CPU: {info['cpu']}")
    print(f"Platform: {info['platform']}")
    print(f"Torch: {info['torch']}")
    print('\n')

    # Tests
    total = len(tests)
    passed = sum(1 for v in tests.values() if v.get('status', '') == 'pass')
    print(f"Passed {passed}/{total} tests:\n")
    for idx in sorted(tests.keys()):
        spec = tests[idx].get('spec', '').strip()
        status = tests[idx].get('status', '')
        mark = '✅' if status == 'pass' else '❌'
        print(f"{mark} {spec}")

    print('\nBenchmarks:\n')
    if not benchmarks:
        print('No benchmark data captured.')
        return
    for idx in sorted(benchmarks.keys()):
        b = benchmarks[idx]
        mean = b.get('mean', None)
        err = b.get('err', None)
        best = b.get('best', None)
        worst = b.get('worst', None)
        spec = b.get('spec', '')
        # convert ns -> ms if mean seems large
        # eval.py logs ns for distributed benchmark and ns for single-benchmark too
        def to_ms(x):
            try:
                return float(x) / 1e6
            except Exception:
                return None

        mean_ms = to_ms(mean) if mean is not None else None
        err_ms = to_ms(err) if err is not None else None
        best_ms = to_ms(best) if best is not None else None
        worst_ms = to_ms(worst) if worst is not None else None

        print(f"{spec}")
        if mean_ms is not None:
            print(f" ⏱ {mean_ms:.2f} ± {err_ms:.3f} ms")
        if best_ms is not None and worst_ms is not None:
            print(f" ⚡ {best_ms:.2f} ms 🐌 {worst_ms:.2f} ms")
        print('\n')


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--tests-file', default='local_tests_2gpus.txt', help='test/benchmark file (semicolon-separated lines)')
    p.add_argument('--gpus', type=int, default=None, help='number of GPUs to use (defaults to torch.cuda.device_count())')
    p.add_argument('--bench', action='store_true', help='also run benchmarks (may take longer)')
    args = p.parse_args()

    n_gpus = args.gpus
    if n_gpus is None and torch is not None:
        try:
            n_gpus = torch.cuda.device_count()
        except Exception:
            n_gpus = 1
    n_gpus = n_gpus or 1

    info = system_info()

    ret, out = run_eval('test', args.tests_file, n_gpus)
    tests, benchmarks, other = parse_popcorn_output(out)

    # If benchmarks were logged during the test run (unlikely), they will appear too
    print_report(info, tests, benchmarks)

    if args.bench:
        print('\nRunning benchmarks (this can take time)...\n')
        ret, out = run_eval('benchmark', args.tests_file, n_gpus)
        _, bench_parsed, _ = parse_popcorn_output(out)
        # merge/update benchmarks
        if bench_parsed:
            print('\nBenchmark results:')
            print_report(info, tests, bench_parsed)


if __name__ == '__main__':
    main()

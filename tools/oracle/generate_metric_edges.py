"""Generate AUC-mu edge evidence with the pinned native Metric factory.

Run: uv run --project tools/oracle python tools/oracle/generate_metric_edges.py
No NumPy metric implementation is used. All floating inputs include IEEE bits.
The behavior contract is the single-thread Apple libc++ build, not all C++ sorts.
"""
import json
import math
from pathlib import Path
import shlex
import struct
import subprocess

from common import COMMIT, OUT, ROOT, SOURCE, load_reference, save_json, sha

LIBRARY_SHA = '9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e'
MASK = (1 << 64) - 1


def bits(value):
    return struct.pack('>d', value).hex()


def case(seed, rows, classes=2, weighted=False, clustered=False):
    state = seed
    labels, scores, weights = [], [0.0] * (rows * classes), []
    for row in range(rows):
        state = (state * 6364136223846793005 + 1) & MASK
        labels.append(float((state >> 32) % classes))
        state = (state * 6364136223846793005 + 1) & MASK
        scores[row] = float((state >> 32) % 40) * 1e-15 / 8.0
        if clustered:
            scores[row] += float(row % 3) * 1e-12
        weights.append(0.0 if rows > 5 and row % 11 == 0 else float(1 + (state >> 40) % 17) / 7.0)
        if classes > 2:
            for k in range(1, classes):
                scores[k * rows + row] = float((state >> (k * 8)) % 32) * 1e-15 / 8.0
    result = dict(name=f'auc_mu_seed_{seed}_rows_{rows}_classes_{classes}'
                  + ('_weighted' if weighted else '') + ('_clusters' if clustered else ''),
                  metric='auc_mu', objective='multiclass',
                  params=dict(objective='multiclass', num_class=classes, num_threads=1, verbosity=-1),
                  labels=labels, scores=scores)
    if weighted:
        result['weights'] = weights
    return result


def specifications():
    cases = [case(1, 128)]
    for seed in [1, 7, 42, 2026]:
        for rows in [5, 23, 24, 127, 129, 257, 1025]:
            for weighted in [False, True]:
                cases.append(case(seed, rows, 3 if seed == 42 else 2, weighted, rows % 2 == 1))
    for seed in [1, 7, 42, 2026]:
        for weighted in [False, True]:
            if seed != 1 or weighted:
                cases.append(case(seed, 128, 2, weighted))
    epsilon = struct.unpack('>f', struct.pack('>f', 1e-15))[0]
    boundaries = [0.0, epsilon / 2, math.nextafter(epsilon / 2, 0.0),
                  math.nextafter(epsilon / 2, math.inf), epsilon,
                  1.5 * epsilon, -epsilon / 2, -epsilon]
    for weighted in [False, True]:
        c = case(42, 64, 2, weighted)
        c['name'] += '_exact_epsilon_boundaries'
        c['scores'][:64] = [boundaries[i % len(boundaries)] for i in range(64)]
        cases.append(c)
        c = case(42, 129, 3, weighted)
        c['name'] += '_asymmetric_matrix'
        c['params']['auc_mu_weights'] = [0.0, 2.0, 0.5, 1.0, 0.0, 3.0, 4.0, 0.25, 0.0]
        cases.append(c)

    return cases


NAMES = {c['name'] for c in specifications()}


def generate(fixtures=ROOT / 'tests/fixtures'):
    _, identity = load_reference()
    assert COMMIT == '5d84f3f222b41409a6983d2ac90513ad9702fa9e'
    assert identity['library']['loaded_sha256'] == LIBRARY_SHA
    work = OUT / 'metric-edges'
    work.mkdir(parents=True, exist_ok=True)
    build = OUT / 'build'
    command = shlex.split(json.loads((build / 'compile_commands.json').read_text())[0]['command'])
    command[command.index('-o') + 1] = str(work / 'driver.o')
    command[command.index('-c') + 1] = str(ROOT / 'tools/oracle/extended_driver.cpp')
    command.extend([f'-I{SOURCE}', f'-I{OUT}'])
    subprocess.run(command, cwd=build, check=True, timeout=300)
    objects = sorted((build / 'CMakeFiles/lightgbm_objs.dir').rglob('*.o'))
    link = [*command[:command.index('-o')], str(work / 'driver.o'), *map(str, objects),
            str(build / 'external_libs/nanoarrow/libnanoarrow_static.a'), '-o', str(work / 'driver')]
    subprocess.run(link, check=True, timeout=300)
    cases = specifications()
    request = work / 'request.json'
    save_json(request, dict(metrics=cases))
    subprocess.run([str(work / 'driver'), str(request), str(work)], check=True, timeout=300)
    native = json.loads((work / 'metrics.json').read_text())
    assert native['cases'][0]['results'][0]['value'] == 0.5346219721066797
    for c in native['cases']:
        c['score_bits'] = [bits(v) for v in c['scores']]
    sysroot = Path(command[command.index('-isysroot') + 1])
    sort_headers = ['sort.h', 'make_heap.h', 'sift_down.h', 'pop_heap.h', 'push_heap.h', 'sort_heap.h']
    native.update(schema_version=1, upstream_commit=COMMIT,
                  compiler_version=identity['compiler']['version'],
                  selected_toolchain_sha256=identity['compiler']['selected_toolchain_sha256'],
                  sort_header_sha256={name: sha(sysroot / 'usr/include/c++/v1/__algorithm' / name)
                                      for name in sort_headers},
                  loaded_library_sha256=LIBRARY_SHA,
                  loaded_library_path=identity['library']['loaded_path'],
                  native_driver_source_sha256=sha(ROOT / 'tools/oracle/extended_driver.cpp'),
                  native_driver_sha256=sha(work / 'driver'),
                  generator_sha256=sha(ROOT / 'tools/oracle/generate_metric_edges.py'),
                  sort_contract='single-thread Apple clang 21.0.0 / libc++ 2200.27.0 std::sort; EPS comparison is not strict weak ordering',
                  epsilon_bits=bits(struct.unpack('>f', struct.pack('>f', 1e-15))[0]), absolute_bound=1e-12)
    save_json(fixtures / 'metric_edges.json', native)
    print(f"Generated {len(cases)} native AUC-mu cases: {fixtures / 'metric_edges.json'}")


if __name__ == '__main__':
    generate()

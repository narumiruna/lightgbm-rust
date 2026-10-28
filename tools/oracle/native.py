"""Compile retained helpers against the pinned Release CPU objects."""
import json
import shlex
import subprocess
import time

from common import OUT, ROOT, SOURCE, check_resources, load_reference, save_json, sha


def compile_helper(name):
    started = time.monotonic()
    _, record = load_reference()
    build = OUT / 'build'
    source = ROOT / 'tools/oracle' / f'{name}.cpp'
    commands = json.loads((build / 'compile_commands.json').read_text())
    command = shlex.split(commands[0]['command'])
    command[command.index('-o') + 1] = str(OUT / f'{name}.o')
    command[command.index('-c') + 1] = str(source)
    command.extend([f'-I{SOURCE}', f'-I{OUT}'])
    subprocess.run(command, cwd=build, check=True, timeout=300)
    objects = sorted((build / 'CMakeFiles/lightgbm_objs.dir').rglob('*.o'))
    objects += sorted((build / 'CMakeFiles/lightgbm_capi_objs.dir').rglob('*.o'))
    link = [*command[:command.index('-o')], str(OUT / f'{name}.o'), *map(str, objects),
            str(build / 'external_libs/nanoarrow/libnanoarrow_static.a'), '-o', str(OUT / name)]
    subprocess.run(link, check=True, timeout=300)
    check_resources(started, children=True)
    provenance = dict(compile_command=command, link_command=link, source_sha256=sha(source),
                      executable_sha256=sha(OUT / name), loaded_library_sha256=record['library']['loaded_sha256'],
                      object_sha256=[dict(path=str(p), sha256=sha(p)) for p in [*objects, build / 'external_libs/nanoarrow/libnanoarrow_static.a']])
    save_json(OUT / f'{name}-record.json', provenance)
    return OUT / name, provenance

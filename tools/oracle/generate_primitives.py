"""Compile and execute the internal reference driver, reusing pinned CPU objects."""
import json
import re
import shlex
import subprocess
import time

from common import OUT, ROOT, SOURCE, check_resources, load_reference, save_json, sha


def generate(fixtures=ROOT / "tests/fixtures"):
    started = time.monotonic()
    _, record = load_reference()
    build = OUT / "build"
    commands = json.loads((build / "compile_commands.json").read_text())
    # Read public data members, then let C++ emit their actual constructed values.
    # This includes no-save CLI fields which Config::ToString intentionally omits.
    header = (SOURCE / "include/LightGBM/config.h").read_text()
    members = re.findall(r"^  (TaskType|std::[\w:<>]+|int|int32_t|int64_t|double|bool) (\w+)(?: = [^;\n]+)?;", header, re.MULTILINE)
    include = OUT / "parameter_defaults.inc"
    code = []
    for typename, name in members:
        value = f"static_cast<int>(defaults.{name})" if typename == "TaskType" else f"defaults.{name}"
        code.append(f'if (Config::parameter2aliases().count("{name}")) declared["{name}"] = J({value});')
    include.write_text("\n".join(code) + "\n")
    # Bound helper compiler memory. Native reference objects retain their Release flags.
    command = ['-O0' if arg == '-O3' else arg for arg in shlex.split(commands[0]["command"])]
    command[command.index("-o") + 1] = str(OUT / "primitive_driver.o")
    command[command.index("-c") + 1] = str(ROOT / "tools/oracle/primitive_driver.cpp")
    command.extend([f"-I{SOURCE}", f"-I{OUT}"])
    subprocess.run(command, cwd=build, check=True)
    objects = sorted((build / "CMakeFiles/lightgbm_objs.dir").rglob("*.o"))
    arrow = build / "external_libs/nanoarrow/libnanoarrow_static.a"
    assert arrow.exists(), "nanoarrow static archive missing"
    # Retain platform, SDK, and deployment flags from the same CMake compile command.
    link = [*command[:command.index("-o")], str(OUT / "primitive_driver.o"),
            *map(str, objects), str(arrow), "-o", str(OUT / "primitive_driver")]
    subprocess.run(link, check=True)
    fixtures.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(OUT / "primitive_driver"), str(fixtures)], check=True)
    for name in ("rng", "bins", "objectives", "histograms", "parameters"):
        # Convert upstream json11's exact 17-digit doubles to readable finite JSON.
        path = fixtures / f"{name}.json"
        save_json(path, json.loads(path.read_text()))
    save_json(OUT / "driver-record.json", {"compile_command": command, "link_command": link,
              "source_sha256": sha(ROOT / "tools/oracle/primitive_driver.cpp"),
              "shared_header_sha256": sha(ROOT / "tools/oracle/driver_common.hpp"),
              "generated_defaults_include_sha256": sha(include),
              "executable_sha256": sha(OUT / "primitive_driver")})
    print("Compiled and ran reference driver: RNG, f32 objectives, BinMapper, histograms, numerical splits, parameters",
          check_resources(started), check_resources(started, children=True))


if __name__ == "__main__":
    generate()

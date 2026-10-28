"""Build only the pinned CPU oracle. Run with the isolated uv project."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

if not __debug__:
    raise RuntimeError("Oracle checks require Python assertions; do not use optimized Python")

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "references/LightGBM"
OUT = ROOT / "target/oracle"
COMMIT = "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
VERSION = "4.7.0.99"


def run(*args, cwd=ROOT):
    return subprocess.check_output(list(map(str, args)), cwd=cwd, text=True).strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    assert run("git", "rev-parse", "HEAD", cwd=SOURCE) == COMMIT, "wrong upstream commit"
    assert (SOURCE / "VERSION.txt").read_text().strip() == VERSION
    assert not run("git", "status", "--porcelain", "--untracked-files=no", cwd=SOURCE), "dirty upstream"
    return {"url": "https://github.com/lightgbm-org/LightGBM", "commit": COMMIT,
            "version": VERSION, "tree": run("git", "rev-parse", "HEAD^{tree}", cwd=SOURCE),
            "license": "MIT", "license_file": "references/LightGBM/LICENSE",
            "license_sha256": sha(SOURCE / "LICENSE")}


def main():
    source_identity()
    # CPU sources include Eigen, fmt, fast_double_parser, and nanoarrow. No compute/GPU fetch.
    for dep in ("eigen", "fmt", "fast_double_parser", "nanoarrow"):
        subprocess.run(["git", "submodule", "update", "--init", "--depth", "1",
                        f"external_libs/{dep}"], cwd=SOURCE, check=True)
    source = source_identity()
    OUT.mkdir(parents=True, exist_ok=True)
    compiler = Path(shutil.which("clang++")).resolve()
    cc = Path(shutil.which("clang")).resolve()
    selected_compiler = Path(run("xcrun", "--find", "clang++")).resolve() if sys.platform == "darwin" else compiler
    flags = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
             f"-DCMAKE_CXX_COMPILER={compiler}", f"-DCMAKE_C_COMPILER={cc}",
             f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={OUT / 'lib'}",
             "-DUSE_OPENMP=OFF", "-DUSE_GPU=OFF", "-DUSE_CUDA=OFF", "-DUSE_ROCM=OFF",
             "-DUSE_MPI=OFF", "-DUSE_HOMEBREW_FALLBACK=OFF", "-DBUILD_CLI=OFF", "-DBUILD_CPP_TEST=OFF"]
    subprocess.run(["cmake", "-S", str(SOURCE), "-B", str(OUT / "build"), *flags], check=True)
    subprocess.run(["cmake", "--build", str(OUT / "build"), "--parallel", "2"], check=True)
    library = OUT / "lib" / ("lib_lightgbm.dylib" if sys.platform == "darwin" else "lib_lightgbm.so")
    stage = OUT / "python/lightgbm"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(SOURCE / "python-package/lightgbm", stage)
    shutil.copyfile(SOURCE / "VERSION.txt", stage / "VERSION.txt")
    (stage / "lib").mkdir(exist_ok=True)
    (stage / "lib" / library.name).symlink_to(library)
    record = {
        "source": source,
        "submodules": run("git", "submodule", "status", cwd=SOURCE).splitlines(),
        "compiler": {"path": str(compiler), "sha256": sha(compiler), "version": run(compiler, "--version"),
                     "selected_toolchain_path": str(selected_compiler), "selected_toolchain_sha256": sha(selected_compiler)},
        "cmake_version": run("cmake", "--version").splitlines()[0],
        "ninja_version": run("ninja", "--version"), "configure_flags": flags,
        "compile_commands_sha256": sha(OUT / "build/compile_commands.json"),
        "library": {"path": str(library.resolve()), "sha256": sha(library)},
        "python": {"version": sys.version, "executable": sys.executable},
    }
    (OUT / "build-record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

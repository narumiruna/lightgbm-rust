"""Reject installed wheels and prove source, build, and loaded image identity."""
import ctypes
import json
from pathlib import Path
import resource
import sys
import time

from build import COMMIT, OUT, ROOT, SOURCE, VERSION, run, sha, source_identity


def load_reference():
    record = json.loads((OUT / "build-record.json").read_text())
    assert source_identity() == record["source"], "source identity changed"
    assert sha(OUT / "build/compile_commands.json") == record["compile_commands_sha256"]
    assert sha(record["compiler"]["path"]) == record["compiler"]["sha256"]
    assert run(record["compiler"]["path"], "--version") == record["compiler"]["version"]
    assert sha(record["compiler"]["selected_toolchain_path"]) == record["compiler"]["selected_toolchain_sha256"]
    stage = OUT / "python"
    sys.path.insert(0, str(stage))
    import lightgbm as lgb
    from lightgbm.libpath import _LIB
    assert Path(lgb.__file__).resolve() == stage / "lightgbm/__init__.py", "installed wheel rejected"
    assert lgb.__version__ == VERSION
    for original in (SOURCE / "python-package/lightgbm").glob("*.py"):
        assert sha(original) == sha(stage / "lightgbm" / original.name), "Python source changed"
    verify_loaded_library(_LIB, record)
    return lgb, record


def verify_loaded_library(library, record):
    expected = Path(record["library"]["path"]).resolve()
    assert Path(library._name).resolve() == expected, "unexpected library search result"
    # dladdr reports the image which supplies the actual C function address.
    class DlInfo(ctypes.Structure):
        _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                    ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]
    info = DlInfo()
    process = ctypes.CDLL(None)
    process.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    process.dladdr.restype = ctypes.c_int
    assert process.dladdr(ctypes.cast(library.LGBM_GetLastError, ctypes.c_void_p), ctypes.byref(info))
    actual = Path(info.filename.decode()).resolve()
    assert actual == expected, "unrelated loaded library rejected"
    assert sha(actual) == record["library"]["sha256"], "loaded library hash changed"
    record["library"]["loaded_path"] = str(actual)
    record["library"]["loaded_sha256"] = sha(actual)
    record["library"]["verification"] = "dladdr(LGBM_GetLastError) + SHA-256"


def check_resources(started, children=False):
    """Check the development-generator budget, not product memory parity."""
    peak = resource.getrusage(resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == 'darwin' else peak * 1024
    elapsed = time.monotonic() - started
    assert peak_bytes < 512 * 1024 * 1024, f'oracle process exceeded 512 MiB: {peak_bytes}'
    assert elapsed < 300, f'oracle case exceeded 300 seconds: {elapsed}'
    return dict(peak_rss_bytes=peak_bytes, elapsed_seconds=elapsed)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

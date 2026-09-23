import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tomllib
from urllib.error import URLError

import pytest

from ine_ocr import r2_profile


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    specification = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def entry(content):
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def archive_file(tmp_path, members):
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        for name, content, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(content)
            if kind == tarfile.SYMTYPE:
                member.linkname = "outside"
            target.addfile(member, io.BytesIO(content))
    return archive


def test_r2_frozen_configuration_is_not_experimental(tmp_path):
    profile = r2_profile.contract()
    environment = r2_profile.runtime_environment(tmp_path)
    assert len(profile["files"]) == 13
    assert profile["license"] == "AGPL-3.0-only"
    assert environment["INE_OCR_PROFILE"] == "r2-v1"
    assert environment["INE_OCR_NAME_MODEL_THRESHOLD"] == "0.9"
    assert environment["INE_OCR_CROSS_CHANNEL_NAME_CONSENSUS"] == "1"
    assert environment["INE_OCR_FRONT_ROI_SIDECAR"] == "0"
    assert environment["INE_OCR_EXPECTED_NAME_MODEL_SHA256"] == "a8e2afbbc28119ae4bab43abbe2cea526e359386030051d244f005f99e019eb5"
    assert environment["INE_OCR_PADDLE_TIER"] == "small"
    assert Path(environment["INE_OCR_YOLO_MODEL_PATH"]).parent == tmp_path / "card"


def test_r2_missing_corrupt_and_symlinked_assets_fail_closed(monkeypatch, tmp_path):
    content = b"synthetic-tensor-fixture"
    target = tmp_path / "fixture.onnx"
    monkeypatch.setattr(r2_profile, "contract", lambda: {"files": {target.name: entry(content)}})
    with pytest.raises(RuntimeError, match="r2_asset_identity_mismatch"):
        r2_profile.verify_models(tmp_path)
    target.write_bytes(content)
    assert r2_profile.verify_models(tmp_path) == {target.name: entry(content)["sha256"]}
    target.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="r2_asset_identity_mismatch"):
        r2_profile.verify_models(tmp_path)
    target.unlink()
    source = tmp_path / "other"
    source.write_bytes(content)
    target.symlink_to(source)
    with pytest.raises(RuntimeError, match="r2_asset_identity_mismatch"):
        r2_profile.verify_models(tmp_path)


@pytest.mark.parametrize("override", ["INE_OCR_NAME_MODEL_THRESHOLD", "INE_OCR_FRONT_ROI_SIDECAR", "INE_OCR_FIELD_SELECTOR", "INE_OCR_TEXT_DET_LIMIT_SIDE_LEN"])
def test_r2_runtime_rejects_recipe_drift(monkeypatch, tmp_path, override):
    for key, value in r2_profile.runtime_environment(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(r2_profile, "verify_models", lambda: {"verified": True})
    assert r2_profile.verify_runtime() == {"verified": True}
    monkeypatch.setenv(override, "different")
    with pytest.raises(RuntimeError, match="r2_(configuration_mismatch|unsupported_override)"):
        r2_profile.verify_runtime()


def test_launcher_verifies_profile_before_serving(monkeypatch, tmp_path):
    import ine_ocr.api as api

    launcher = load_script("serve_r2")
    monkeypatch.setattr(r2_profile.os, "environ", dict(r2_profile.os.environ))
    monkeypatch.setenv("INE_OCR_R2_MODEL_DIR", str(tmp_path))
    calls = []
    monkeypatch.setattr(r2_profile, "verify_runtime", lambda: calls.append("verify"))
    monkeypatch.setattr(api, "main", lambda: calls.append("serve"))
    launcher.main()
    assert calls == ["verify", "serve"]
    assert api.os.environ["INE_OCR_NAME_MODEL_PATH"] == str(tmp_path / "name/ine-names.onnx")


def test_launcher_rejects_override_before_serving(monkeypatch, tmp_path):
    launcher = load_script("serve_r2")
    monkeypatch.setattr(r2_profile.os, "environ", dict(r2_profile.os.environ))
    monkeypatch.setenv("INE_OCR_R2_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("INE_OCR_NAME_MODEL_THRESHOLD", "0.1")
    with pytest.raises(RuntimeError, match="r2_configuration_mismatch"):
        launcher.main()


def test_package_and_docker_entrypoints_share_the_frozen_gate():
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert package["project"]["scripts"]["ine-ocr-serve"] == "ine_ocr.r2_profile:serve"
    assert load_script("serve_r2").main is r2_profile.serve


@pytest.mark.parametrize("defect", ["missing", "hash", "configuration"])
def test_shared_entrypoint_rejects_bad_release_before_listening(monkeypatch, tmp_path, defect):
    import ine_ocr.api as api

    monkeypatch.setattr(r2_profile.os, "environ", dict(r2_profile.os.environ))
    monkeypatch.setenv("INE_OCR_R2_MODEL_DIR", str(tmp_path))
    profile = r2_profile.contract()
    content = b"synthetic-model"
    for name in profile["files"]:
        profile["files"][name] = entry(content)
        if defect != "missing":
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    monkeypatch.setattr(r2_profile, "contract", lambda: profile)
    if defect == "hash":
        (tmp_path / "name/ine-names.onnx").write_bytes(b"X" * len(content))
    if defect == "configuration":
        monkeypatch.setenv("INE_OCR_CROSS_CHANNEL_NAME_CONSENSUS", "0")
    calls = []
    monkeypatch.setattr(api, "main", lambda: calls.append("listen"))
    with pytest.raises(RuntimeError, match="r2_(asset_identity|configuration)_mismatch"):
        r2_profile.serve()
    assert not calls


def test_verified_archive_unpacks_only_expected_files(tmp_path):
    downloader = load_script("fetch_r2_models")
    content = b"synthetic-model"
    archive = archive_file(tmp_path, [("name/model.onnx", content, tarfile.REGTYPE)])
    destination = tmp_path / "unpacked"
    downloader.unpack_verified(archive, destination, {"name/model.onnx": entry(content)})
    assert (destination / "name/model.onnx").read_bytes() == content


@pytest.mark.parametrize("name,kind", [("../escaped", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE), ("model", tarfile.SYMTYPE)])
def test_archive_rejects_unsafe_member_even_if_allowlisted(tmp_path, name, kind):
    downloader = load_script("fetch_r2_models")
    archive = archive_file(tmp_path, [(name, b"", kind)])
    with pytest.raises(RuntimeError, match="r2_archive_member_invalid"):
        downloader.unpack_verified(archive, tmp_path / "unpacked", {name: entry(b"")})


@pytest.mark.parametrize("defect", ["duplicate", "unexpected", "hash", "size"])
def test_archive_rejects_integrity_failures(tmp_path, defect):
    downloader = load_script("fetch_r2_models")
    content = b"synthetic-model"
    members = [("model", content, tarfile.REGTYPE)]
    expected = {"model": entry(content)}
    if defect == "duplicate":
        members *= 2
    elif defect == "unexpected":
        members.append(("extra", content, tarfile.REGTYPE))
    elif defect == "hash":
        expected["model"]["sha256"] = "0" * 64
    elif defect == "size":
        expected["model"]["bytes"] += 1
    archive = archive_file(tmp_path, members)
    with pytest.raises(RuntimeError, match="r2_archive_"):
        downloader.unpack_verified(archive, tmp_path / "unpacked", expected)


def test_model_download_retries_network_failure_with_verified_bytes(monkeypatch, tmp_path):
    downloader = load_script("fetch_r2_models")
    content = b"synthetic-model"
    calls = []

    def open_response(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise URLError("temporary")
        return io.BytesIO(content)

    monkeypatch.setattr(downloader, "urlopen", open_response)
    monkeypatch.setattr(downloader.time, "sleep", lambda seconds: None)
    target = tmp_path / "download"
    downloader.download(target, {**entry(content), "url": "https://example.invalid/model"})
    assert len(calls) == 2
    assert target.read_bytes() == content


def test_model_download_never_accepts_wrong_hash(monkeypatch, tmp_path):
    downloader = load_script("fetch_r2_models")
    monkeypatch.setattr(downloader, "urlopen", lambda *args, **kwargs: io.BytesIO(b"changed"))
    with pytest.raises(RuntimeError, match="r2_download_hash_mismatch"):
        downloader.download(tmp_path / "download", {**entry(b"correct"), "url": "https://example.invalid/model"})

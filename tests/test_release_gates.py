import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("failed_command", ["status", "rev-parse"])
def test_git_errors_stop_before_docker(tmp_path, failed_command):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "deploy.sh").write_text((ROOT / "scripts/deploy.sh").read_text())
    (tmp_path / ".env").write_text("")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text('#!/bin/sh\nif [ "$1" = "$FAIL_COMMAND" ]; then exit 128; fi\nif [ "$1" = "rev-parse" ]; then printf "1111111111111111111111111111111111111111\\n"; fi\nexit 0\n')
    git.chmod(0o700)
    docker = binaries / "docker"
    docker.write_text('#!/bin/sh\ntouch "$DOCKER_MARKER"\nexit 0\n')
    docker.chmod(0o700)
    marker = tmp_path / "docker_called"
    environment = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"], "FAIL_COMMAND": failed_command, "DOCKER_MARKER": str(marker)}
    result = subprocess.run(["bash", str(scripts / "deploy.sh")], env=environment, capture_output=True)
    assert result.returncode != 0
    assert not marker.exists()


def test_optimized_python_keeps_runtime_verification():
    program = '''
import runpy
verify = runpy.run_path("scripts/deployment_receipt.py")["verify_container"]
container = {
    "Image": "sha256:wrong", "State": {"Health": {"Status": "healthy"}},
    "HostConfig": {"ReadonlyRootfs": True, "PortBindings": {"8100/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8100"}]}, "Memory": 6 * 1024 ** 3, "NanoCpus": 2000000000, "CapDrop": ["ALL"]},
    "Config": {"User": "ocr", "Labels": {"org.opencontainers.image.revision": "synthetic"}},
}
verify(container, {"release_commit": "synthetic"}, commit="synthetic", image_sha256="expected")
'''
    result = subprocess.run([sys.executable, "-O", "-c", program], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "deployment_verification_failed:image_identity" in result.stderr


def test_optimized_python_keeps_smoke_requests_and_failure_gate():
    program = '''
import io
import runpy
main = runpy.run_path("scripts/smoke_test.py")["main"]
class Response(io.BytesIO):
    status = 500
def fake_request(*args, **kwargs):
    print("SYNTHETIC_HTTP_PROBE")
    return Response(b"{}")
main.__globals__["urlopen"] = fake_request
main.__globals__["local_token"] = lambda: "synthetic" * 8
main()
'''
    result = subprocess.run([sys.executable, "-O", "-c", program], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "SYNTHETIC_HTTP_PROBE" in result.stdout
    assert "smoke_failed:health" in result.stderr

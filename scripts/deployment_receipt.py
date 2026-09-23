import datetime
import json
import os
from pathlib import Path
import subprocess


def verify_container(container, manifest, *, commit, image_sha256):
    checks = {
        "image_identity": container["Image"] == "sha256:" + image_sha256,
        "healthy": container["State"]["Health"]["Status"] == "healthy",
        "read_only": container["HostConfig"]["ReadonlyRootfs"],
        "non_root": container["Config"]["User"] == "ocr",
        "loopback_binding": container["HostConfig"]["PortBindings"] == {"8100/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8100"}]},
        "memory_limit": container["HostConfig"]["Memory"] == 6 * 1024 ** 3,
        "cpu_limit": container["HostConfig"]["NanoCpus"] == 2_000_000_000,
        "dropped_capabilities": container["HostConfig"]["CapDrop"] == ["ALL"],
        "revision_label": container["Config"]["Labels"]["org.opencontainers.image.revision"] == commit,
        "manifest_commit": manifest["release_commit"] == commit,
    }
    for name, passed in checks.items():
        if not passed:
            raise RuntimeError("deployment_verification_failed:" + name)


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts"
    output.mkdir(mode=0o700, exist_ok=True)
    container_id = subprocess.check_output(["docker", "compose", "ps", "-q", "ocr"], text=True).strip()
    container = json.loads(subprocess.check_output(["docker", "inspect", container_id], text=True))[0]
    manifest = json.loads(subprocess.check_output(["docker", "exec", container_id, "cat", "/opt/ine-ocr/runtime-manifest.json"], text=True))
    verify_container(container, manifest, commit=os.environ["RELEASE_COMMIT"], image_sha256=os.environ["RELEASE_ARTIFACT_SHA256"])
    receipt = {
        "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit": os.environ["RELEASE_COMMIT"], "artifact_sha256": os.environ["RELEASE_ARTIFACT_SHA256"],
        "container_id": container_id, "health": "healthy", "bind": "127.0.0.1:8100", "runtime": manifest,
        "rollback": "Previous verified image ID and corresponding receipt; never rebuild a moving tag for rollback",
    }
    descriptor = os.open(output / (os.environ["RELEASE_TAG"] + ".json"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({"status": "verified", "commit": receipt["commit"], "artifact_sha256": receipt["artifact_sha256"]}))


if __name__ == "__main__":
    main()

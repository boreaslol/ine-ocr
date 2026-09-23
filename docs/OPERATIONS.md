# Operating a deployment

## Release and verification

Build on the target host from a clean checkout of the public `main` branch, using `bash scripts/deploy.sh`. Do not transfer hand-edited runtime directories. Keep the generated `.env` only on that host or in your secret manager, never in an image or Git. The Docker build context is an explicit allowlist and excludes `.git`, `.env`, tests and local artifacts.

`scripts/deployment_receipt.py` verifies the healthy container's immutable image ID, revision label, non-root/read-only configuration, loopback binding and asset manifest. It writes a local owner-only receipt under ignored `artifacts/`. Keep the previous receipt and image before upgrading. Image ID is the local content-addressed Docker image identity, not a registry manifest digest or an archive checksum.

The default image includes the complete R2 models. The runtime manifest records dependency versions, every model hash and frozen R2 settings. Check it with `docker exec <container-id> cat /opt/ine-ocr/runtime-manifest.json`. Authenticated readiness must report `profile: r2-v1`, a configured name model and enabled cross-channel consensus. Missing assets or changed settings fail rather than silently reducing functionality. Do not publish raw `docker inspect` or `docker compose config` output: it contains your token.

## Remote callers

The repository intentionally supplies no shared endpoint, DNS name, certificates, cloud firewall rules or organization credentials. Complete these steps on your infrastructure:

1. Put an HTTPS reverse proxy on the same host in front of `127.0.0.1:8100`, or use a private authenticated tunnel.
2. Restrict permitted backend egress addresses and protect both proxy and service access. Do not send the server token to browsers/mobile apps.
3. Preserve `Authorization`; configure body limits no greater than 44 MiB, finite upload/read timeouts, rate limiting and bounded retry budgets.
4. Disable request/response body logging. Avoid query strings carrying secrets or user data.
5. Verify valid caller success, missing/wrong token rejection, unauthorized network rejection, certificate validation and renewal monitoring from the real caller.

A successful localhost smoke test does not certify any external network path. Capacity-test representative consented inputs before increasing concurrency; raising worker count duplicates model memory and may increase CPU contention. The default limits deliberately prioritize predictable single-inference memory use, not maximum throughput.

## Retention and rollback

OCR uses temporary PNGs on container `/tmp`, removes them after each read, and has no persistent image volume. A forced process termination can leave an intermediate in tmpfs until container removal. Logs are rotated but request IDs and metadata persist until rotation. Control host swap/crash dumps/backups and application-side retention as part of your deployment policy.

Before an upgrade, retain the previous verified image ID and its local receipt. If acceptance fails, stop the new container and redeploy the previous image **by that immutable ID**, preserving the existing token, loopback binding, tmpfs, resource limits and non-root/read-only settings. Re-run readiness and `scripts/smoke_test.py`, and write a new receipt matching the rollback image and source commit. Do not rebuild a moving tag as a substitute for the saved image. Automatic cross-version rollback is not provided.

For a smoke test against an already running service, activate the host venv and run `python scripts/smoke_test.py`. It reads the local `.env` privately, creates only synthetic image content in memory and prints statuses/timing, never the token or OCR fields.

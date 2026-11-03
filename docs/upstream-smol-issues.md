# Smol Machines upstream observations

These are minimal local reproductions for behavior that may be useful to report
upstream. They are not Monopole product guarantees.

Environment: Apple Silicon macOS, local `.smolmachine` image, observed
2026-07-26.

## CLI 1.6.6 delays completed guest commands

```bash
smol machine create --name latency-smoke --from image.smolmachine --net
smol machine start --name latency-smoke --local
time smol machine exec --name latency-smoke --local -- true
```

The guest command completed immediately, but the CLI returned after 61.93
seconds. With CLI 1.6.13, the same command returned in 0.09 seconds.

Status: fixed by upgrading the required CLI version to 1.6.13.

## CLI deletion leaves the local image mounted

With both CLI 1.6.6 and 1.6.13:

```bash
smol machine rm --name latency-smoke --yes --local
```

The command exited successfully but warned:

```text
Warning: failed to clean up data directory: Resource busy (os error 16)
```

`smol machine ls --local --json` was empty, while `hdiutil info` still showed
the VM's `layers-cs.sparseimage` attached and its APFS volume remained mounted.
The cache directory also remained. Smol SDK 1.6.13 `Machine.delete()` cleaned
the OCI/base machines it created, but the SDK cannot create from this portable
artifact for an equivalent comparison.

Status: Monopole detaches and removes only the exact deleted machine directory
returned by `smol machine data-dir`.

## CLI 1.6.13 interactive stdin upload reports false success

```bash
printf archive |
  smol machine exec --name latency-smoke --local --interactive -- \
    dd of=/tmp/workspace.tar status=none
smol machine exec --name latency-smoke --local -- \
  test -f /tmp/workspace.tar
```

The `dd` invocation exited successfully, but the file did not exist. The
equivalent SDK `Machine.write_file()` call created the file correctly with the
requested mode.

Status: Monopole passes bounded base64 through `--secret-env` indirection to
`machine exec`; payload bytes are not present in process arguments.

## CLI 1.6.13 copy reports success without updating a running guest

```bash
smol machine cp --local proof.txt latency-smoke:/tmp/proof.txt
smol machine exec --name latency-smoke --local -- test -f /tmp/proof.txt
```

The copy command printed `Uploaded ...` and exited successfully. The running
guest did not contain the file. Copying to a stopped machine is rejected because
the machine must be running.

Status: Monopole does not use `machine cp`.

## CLI 1.6.13 can intermittently fail an immediate exec without diagnostics

Across repeated create/start/exec/delete preflights, one bounded file-transfer
`machine exec` returned nonzero with empty stdout and stderr. The same preflight
passed immediately before and after, and no guest command or input changed.

Status: Monopole retries only its idempotent fixed-offset file-transfer commands
up to three times. Agent execution is never retried by this workaround.

## SDK 1.6.13 local creation silently ignores `MachineConfig.image`

```python
from smol import Machine, MachineConfig

machine = Machine.create(
    MachineConfig(
        name="sdk-image-smoke",
        image="registry.example/runtime@sha256:complete-digest",
        persistent=True,
    )
)
try:
    result = machine.exec(["test", "-x", "/usr/local/bin/codex"])
    assert result.exit_code == 0, result
finally:
    machine.delete()
```

The local SDK transport's `_native_config` omits `MachineConfig.image`, so
creation succeeded but the guest contained the base filesystem rather than the
pinned image. The same happened with an absolute `.smolmachine` path, and the
expected harness executable was absent. Connecting the SDK to a CLI-created
portable machine with the same name exposed the base filesystem and stopped the
CLI-created packaged VM.

Status: Monopole does not use the SDK until it can create a local machine from
the pinned runtime image.

## SDK 1.6.13 execution cannot be interrupted from another Python thread

```python
import threading
import time

from smol import Machine

machine = Machine.create()
observed = []


def watchdog():
    time.sleep(1)
    observed.append(time.monotonic())


thread = threading.Thread(target=watchdog)
thread.start()
started = time.monotonic()
machine.exec(["sleep", "10"])
thread.join()
machine.delete()
assert observed[0] - started < 2
```

A native `machine.exec(["sleep", "10"])` call prevented the Python thread
requesting cancellation from running until the guest command completed. The
equivalent CLI subprocess could be terminated immediately by the host.

Status: Monopole uses a CLI subprocess for long-running agent execution.

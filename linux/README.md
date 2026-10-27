# The agent's computer

Agent mode gives the model a real computer: a small 32-bit Linux that boots inside the page in the [v86](https://github.com/copy/v86) PC emulator (`machine.js`). The program types the model's commands into its serial console, and what Linux prints goes back to the model.

`bzimage.bin` is that Linux: one file holding the kernel and, built into it, a tiny root filesystem with BusyBox (the shell and the usual commands). It has no network driver, so nothing on it can reach the internet, and the page keeps a copy of its state just after boot, so Reset puts it back exactly.

## What is in it

| Part | Version | Licence |
|---|---|---|
| Linux kernel, configured from `tinyconfig` with only what the page needs (`kernel.config`) | the version in `buildroot.config` | GPL-2.0 |
| BusyBox | Buildroot's default | GPL-2.0 |
| musl C library (Bootlin's prebuilt i686 toolchain) | Buildroot's default | MIT |
| Buildroot's skeleton, plus `overlay/` (a root shell on the serial port, the `~% ` prompt) | Buildroot 2025.02.19 | GPL-2.0 |

The emulator itself (v86, BSD-2-Clause) and its BIOS (SeaBIOS, LGPL-3.0) are loaded from jsDelivr, not stored here.

## Source code

The complete corresponding source for `bzimage.bin` (every package's source tarball, its licence texts, and the exact Buildroot configuration, made with Buildroot's `make legal-info`) is attached to the GitHub release named in `SOURCE.txt`.

## Rebuilding it

With Docker, from the repository root:

```
docker run --rm -v "$PWD/linux:/work/linux" -v "$PWD/out:/work/out" debian:12 sh /work/linux/build.sh
```

`build.sh` downloads Buildroot, builds with `defconfig`, `kernel.config` and `overlay/`, and writes `out/bzimage.bin`, `out/buildroot.config` and `out/legal-info.tar.gz`.

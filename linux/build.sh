#!/bin/sh
# Builds linux/bzimage.bin (and the GPL source bundle) from this folder, inside a throwaway Debian container:
#   docker run --rm -v "$PWD/linux:/work/linux" -v "$PWD/out:/work/out" debian:12 sh /work/linux/build.sh
set -eu
export FORCE_UNSAFE_CONFIGURE=1 # the container runs as root
BR=2025.02.19
apt-get update -qq && apt-get install -y -qq build-essential wget cpio unzip rsync bc file python3 perl git xz-utils flex bison libelf-dev libssl-dev >/dev/null
cd /tmp && wget -q https://buildroot.org/downloads/buildroot-$BR.tar.xz && tar xf buildroot-$BR.tar.xz && cd buildroot-$BR
# the kernel: the version this Buildroot calls "latest", configured from tinyconfig with only what the page needs
KV=$(sed -n '/config BR2_LINUX_KERNEL_VERSION/,/^$/p' linux/Config.in | sed -n 's/.*default "\([0-9.]*\)" if BR2_LINUX_KERNEL_LATEST_VERSION.*/\1/p')
echo "kernel $KV"
if [ ! -f /work/linux/kernel.config ]; then
  wget -q https://cdn.kernel.org/pub/linux/kernel/v${KV%%.*}.x/linux-$KV.tar.xz && tar xf linux-$KV.tar.xz && cd linux-$KV
  make -s ARCH=i386 tinyconfig
  ./scripts/config --enable M686 --enable PRINTK --enable TTY --enable SERIAL_8250 --enable SERIAL_8250_CONSOLE \
    --enable BINFMT_ELF --enable BINFMT_SCRIPT --enable BLK_DEV_INITRD --enable RD_GZIP --enable DEVTMPFS \
    --enable PROC_FS --enable SYSFS --enable TMPFS --enable SHMEM --enable MULTIUSER --enable FUTEX --enable EPOLL \
    --enable SIGNALFD --enable TIMERFD --enable EVENTFD --enable POSIX_TIMERS --enable FILE_LOCKING --enable UNIX98_PTYS \
    --enable CMDLINE_BOOL --set-str CMDLINE "console=ttyS0" --disable NET
  make -s ARCH=i386 olddefconfig && cp .config /work/linux/kernel.config && cd ..
fi
sed "s/@KERNEL@/$KV/" /work/linux/defconfig | sed "s#\$(BR2_EXTERNAL_AGENT_PATH)#/work/linux#" > configs/agent_defconfig
make -s agent_defconfig
make -j8 2>&1 | tail -5
make legal-info >/dev/null 2>&1 || true
mkdir -p /work/out && cp output/images/bzImage /work/out/bzimage.bin
cp .config /work/out/buildroot.config && tar czf /work/out/legal-info.tar.gz -C output legal-info
ls -la /work/out

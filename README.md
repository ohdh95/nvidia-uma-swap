# nvidia-uma-swap

DGX Spark / Ubuntu 24.04 / aarch64용 NVIDIA open kernel modules 빌드·적용 절차다. 아래 명령은 같은 Bash 세션에서 순서대로 실행한다.

- 원본: <https://github.com/NVIDIA/open-gpu-kernel-modules>
- 태그: `580.173.02`
- 원본 커밋: `20e4e6e19cc26ba47b5cbe23130a396be100c427`
- NVIDIA 사용자 공간 드라이버와 GSP firmware도 **580.173.02**를 사용한다. 현재 설치된 같은 버전의 사용자 공간 구성은 유지한다.

## 1. 빌드 준비

```bash
sudo apt update
sudo apt install build-essential gcc-13 g++-13 libelf-dev libssl-dev kmod zstd openssl mokutil

cd ~/Downloads/dgx-spark-source/nvidia-uma-swap
UMA_DRIVER_SRC="$(pwd -P)"
UMA_KERNEL_SRC="$(dirname "$UMA_DRIVER_SRC")/linux-uma-swap"
UMA_KERNEL_OUT="$(dirname "$UMA_DRIVER_SRC")/build/linux-uma-swap"
UMA_SIGNING_DIR="$HOME/.local/share/uma-swap/keys"
UMA_KERNEL_RELEASE="$(cat "$UMA_KERNEL_OUT/include/config/kernel.release")"
test -s "$UMA_KERNEL_OUT/Module.symvers"
printf '대상 커널: %s\n' "$UMA_KERNEL_RELEASE"
```

위 설정은 형제 디렉터리 `linux-uma-swap`의 README에 따라 커널 전체 빌드와 패키지 설치를 완료한 경우다. `CONFIG_MODVERSIONS=y`이므로 `modules_prepare`만 실행한 트리로 대체하지 않는다. 서명 키도 커널 README의 절차로 생성·등록한다.

현재 배포판 커널에서 드라이버만 바꾸려면 커널 관련 변수는 다음으로 대체한다. 해당 커널의 headers 패키지가 필요하다.

```bash
UMA_KERNEL_RELEASE="$(uname -r)"
UMA_KERNEL_SRC="$(readlink -f "/lib/modules/$UMA_KERNEL_RELEASE/build")"
UMA_KERNEL_OUT="$UMA_KERNEL_SRC"
test -s "$UMA_KERNEL_OUT/Module.symvers"
```

## 2. 컴파일

```bash
make -C "$UMA_DRIVER_SRC" -j"$(nproc)" modules \
    TARGET_ARCH=aarch64 CC=gcc-13 \
    SYSSRC="$UMA_KERNEL_SRC" SYSOUT="$UMA_KERNEL_OUT"

modinfo -F version "$UMA_DRIVER_SRC/kernel-open/nvidia.ko"
modinfo -F vermagic "$UMA_DRIVER_SRC/kernel-open/nvidia-uvm.ko"
test "$(modinfo -F vermagic "$UMA_DRIVER_SRC/kernel-open/nvidia.ko" | awk '{print $1}')" \
    = "$UMA_KERNEL_RELEASE"
```

버전은 `580.173.02`, vermagic의 첫 항목은 대상 커널이어야 한다. 다른 커널로 빌드 대상을 바꾸면 먼저 같은 `SYSSRC`·`SYSOUT` 인자로 `make clean`을 실행한다.

## 3. 서명·설치

현재 Spark는 Secure Boot가 켜져 있으므로 등록한 MOK로 서명한다. 빌드한 NVIDIA 모듈을 모두 같은 버전으로 설치한다. 서명 후 모듈을 strip하지 않는다.

```bash
test -f "$UMA_SIGNING_DIR/MOK.key"
test -f "$UMA_SIGNING_DIR/MOK.der"
mokutil --test-key "$UMA_SIGNING_DIR/MOK.der"
test -f "$UMA_DRIVER_SRC/kernel-open/nvidia.ko" || exit 1

for module in "$UMA_DRIVER_SRC"/kernel-open/*.ko; do
    test -f "$module" || break
    "$UMA_KERNEL_OUT/scripts/sign-file" sha512 \
        "$UMA_SIGNING_DIR/MOK.key" "$UMA_SIGNING_DIR/MOK.der" "$module" || exit 1
    sudo install -D -m 644 "$module" \
        "/lib/modules/$UMA_KERNEL_RELEASE/updates/uma-swap/$(basename "$module")" || exit 1
done

sudo depmod -a "$UMA_KERNEL_RELEASE"
modinfo -k "$UMA_KERNEL_RELEASE" -n nvidia
modinfo -k "$UMA_KERNEL_RELEASE" -n nvidia_uvm
modinfo -k "$UMA_KERNEL_RELEASE" -F signer nvidia_uvm
```

두 모듈 경로가 모두 `/lib/modules/<대상 커널>/updates/uma-swap/` 아래인지 확인한다. 다른 경로가 나오면 모듈 선택 문제를 해결한 뒤 진행한다.

```bash
sudo update-initramfs -u -k "$UMA_KERNEL_RELEASE"
sudo update-grub
sudo reboot
```

직접 빌드한 커널을 사용한다면 재부팅 전에 커널 README의 커널 이미지 서명까지 완료해야 한다. 실행 중인 드라이버의 교체는 재부팅으로 적용한다.

## 4. 적용 확인

```bash
uname -r
cat /proc/driver/nvidia/version
nvidia-smi
sudo modprobe nvidia_uvm
modinfo -n nvidia_uvm
modinfo -F signer nvidia_uvm
sudo journalctl -k -b --no-pager | tail -n 80
```

대상 커널과 NVIDIA `580.173.02`인지 확인한다. `modinfo`는 디스크에 설치된 모듈을 표시하므로 실제 적용 확인은 재부팅 후에 한다.

## 5. 수동 설치 모듈 제거

대상 커널을 확인하고 수동 설치 디렉터리만 제거한다. 같은 커널용 모듈이 다른 위치에 있으면 그 모듈을 다시 선택한다. 직접 빌드한 커널에 대체 모듈이 없다면 GRUB에서 원래 `7.0.0-1019-nvidia`로 부팅한다.

```bash
test -n "$UMA_KERNEL_RELEASE" || exit 1
sudo rm -rf "/lib/modules/$UMA_KERNEL_RELEASE/updates/uma-swap"
sudo depmod -a "$UMA_KERNEL_RELEASE"
sudo update-initramfs -u -k "$UMA_KERNEL_RELEASE"
sudo reboot
```

## 참고

- [NVIDIA 빌드 설명](https://github.com/NVIDIA/open-gpu-kernel-modules/tree/580.173.02)
- [Linux 모듈 서명](https://docs.kernel.org/admin-guide/module-signing.html)
- [Ubuntu Secure Boot](https://documentation.ubuntu.com/security/security-features/platform-protections/secure-boot/)
- [DGX Spark 커널 빌드 설정](https://github.com/ohdh95/linux-uma-swap/blob/main/configs/spark.config)

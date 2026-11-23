# GPU 메모리 실험 소스

기존 `gpu-pte-analysis`에서 소스·빌드 스크립트·README·bpftrace 입력 템플릿만 보관한 것이다. 원래 실험 폴더와 기존 측정 결과는 상위 작업 폴더에 그대로 남아 있다.

## 빌드·실행

저장소 루트에서 실행한다. CUDA는 `/usr/local/cuda`에 설치된 환경을 사용한다.

```bash
# CPU/GPU managed fault 최소 실험: 내부에서 nvcc로 빌드하고 실행한다.
bash experiments/gpu-pte-analysis/gpu-managed-fault-minimal/run.sh

# malloc/cudaMalloc의 할당 크기 비교: run.py가 C++ probe를 빌드한다.
python3 experiments/gpu-pte-analysis/folio-size-comparison-20261120/run.py --sizes 1024

# 같은 비교에 커널 추적을 추가한다. bpftrace와 root 권한이 필요하다.
sudo python3 experiments/gpu-pte-analysis/folio-size-comparison-20261120/run.py --trace --sizes 1024
```

`reproduce.sh`, `run.sh`, `build.sh`가 있는 실험은 해당 스크립트의 빌드 명령을 사용한다. 별도 분석 스크립트는 해당 실험에서 생성한 로컬 출력이 필요하다. 과거 audit 보고서와 측정 결과는 이 저장소에 포함하지 않는다.

`prefetch-uma-20261123/probe.cpp`만 빌드·실행하려면:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -I/usr/local/cuda/include \
    experiments/gpu-pte-analysis/prefetch-uma-20261123/probe.cpp \
    -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart \
    -o /tmp/uma-prefetch-probe
/tmp/uma-prefetch-probe
```

커널 소스도 조사하는 `backing-manager-audit-20261120`은 `linux-uma-swap`과 `nvidia-uma-swap`이 형제 디렉터리에 있다고 가정한다. 각 실험의 바이너리·baseline/trace 디렉터리·로그·CSV/JSON/TSV 측정 결과는 Git 추적에서 제외된다.

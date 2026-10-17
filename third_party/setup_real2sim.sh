#!/usr/bin/env bash
# KIWI's tested Linux / CUDA 12.8 installation; does not modify the KIWI env.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash third_party/setup_real2sim.sh [--backend PATH] [--cuda-home PATH]

Installs the pinned NVIDIA 3DGUT/3DGRT renderer and KIWI in BACKEND/.venv, for
`kiwi real2simviz`.
Default BACKEND: $KIWI_3DGRUT, otherwise <KIWI>/build/3dgrut.
Requires Linux x86_64, NVIDIA driver, CUDA toolkit 12.8, C++ compiler, git,
curl and uv. CUDA_HOME or nvcc on PATH locates the toolkit.

The script creates no service and never updates another environment. A backend
checkout at a different revision is left untouched and rejected.
EOF
}

kiwi_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
backend=${KIWI_3DGRUT:-"$kiwi_root/build/3dgrut"}
cuda_root=${CUDA_HOME:-}
while (($#)); do
    case "$1" in
        --backend|--cuda-home)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            if [[ $1 == --backend ]]; then backend=$2; else cuda_root=$2; fi
            shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo "This tested installation targets Linux x86_64." >&2; exit 1
fi
for tool in git curl uv c++; do
    command -v "$tool" >/dev/null || { echo "Required tool not found: $tool" >&2; exit 1; }
done
if [[ -z $cuda_root ]]; then
    nvcc_path=$(command -v nvcc || true)
    if [[ -n $nvcc_path ]]; then
        cuda_root=$(dirname "$(dirname "$(readlink -f "$nvcc_path")")")
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        cuda_root=/usr/local/cuda
    fi
fi
if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
    echo "CUDA toolkit required: pass --cuda-home /path/to/cuda-12.8" >&2; exit 1
fi
if ! "$cuda_root/bin/nvcc" --version | grep -q 'release 12.8'; then
    echo "This recipe pins PyTorch cu128; supply a CUDA 12.8 toolkit." >&2; exit 1
fi
export CUDA_HOME=$cuda_root
export PATH="$CUDA_HOME/bin:$PATH"
# Conda/Pixi toolkits store headers and libraries under targets; system CUDA
# uses include/lib64. Add either layout only when present.
for inc in "$CUDA_HOME/targets/x86_64-linux/include" "$CUDA_HOME/include"; do
    [[ ! -d $inc ]] || export CPATH="$inc${CPATH:+:$CPATH}"
done
for lib in "$CUDA_HOME/targets/x86_64-linux/lib" "$CUDA_HOME/lib64" "$CUDA_HOME/lib"; do
    if [[ -d $lib ]]; then
        export LIBRARY_PATH="$lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
        export LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
export MAX_JOBS=${MAX_JOBS:-6}

revision=a37ef721012dea0f29c0fcfff2d525023b4e854a
if [[ ! -d "$backend/.git" ]]; then
    if [[ -e $backend ]]; then
        echo "Destination already exists and is not a Git checkout: $backend" >&2; exit 1
    fi
    mkdir -p "$(dirname "$backend")"
    git init "$backend"
    git -C "$backend" remote add origin https://github.com/nv-tlabs/3dgrut.git
    git -C "$backend" fetch --depth 1 origin "$revision"
    git -C "$backend" checkout --detach FETCH_HEAD
fi
backend=$(cd "$backend" && pwd)
if [[ $(git -C "$backend" rev-parse HEAD) != "$revision" ]]; then
    echo "Backend revision differs from the tested commit; use a new --backend directory." >&2; exit 1
fi
# GUT's CUDA tracer uses tiny-cuda-nn headers even with SH; no Python binding
# build is needed. GRT also needs the OptiX headers from its submodule.
git -C "$backend" submodule update --init --recursive

if [[ ! -x "$backend/.venv/bin/python" ]]; then
    uv venv --python 3.11 "$backend/.venv"
fi
backend_python="$backend/.venv/bin/python"
export PATH="$backend/.venv/bin:$PATH"
constraints="$backend/.venv/kiwi-constraints.txt"
cat > "$constraints" <<'EOF'
torch==2.8.0+cu128
torchvision==0.23.0+cu128
nvidia-ncore==19.8.0
viser==1.1.0
EOF
uv pip install --python "$backend_python" --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128'
uv pip install --python "$backend_python" --constraint "$constraints" \
    --index pytorch=https://download.pytorch.org/whl/cu128 \
    -e "$backend[gui]" -e "$kiwi_root"
if [[ -z ${TORCH_CUDA_ARCH_LIST:-} ]]; then
    export TORCH_CUDA_ARCH_LIST=$("$backend_python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('A visible NVIDIA GPU is required, or set TORCH_CUDA_ARCH_LIST explicitly.')
print('.'.join(map(str, torch.cuda.get_device_capability())))
PY
)
fi
# Pin the same Slang compiler used by the tested 3DGUT/3DGRT runtime.
slang_archive="$backend/.venv/slang-2026.5.2-linux-x86_64.tar.gz"
slang_sha=d4f93f5e541a42b6e5a75b82ac5f787077ec74d80227dbfc5963324a4e8efcf1
if [[ ! -f $slang_archive ]]; then
    curl --fail --location --retry 3 \
        https://github.com/shader-slang/slang/releases/download/v2026.5.2/slang-2026.5.2-linux-x86_64.tar.gz \
        --output "$slang_archive.part"
    mv "$slang_archive.part" "$slang_archive"
fi
printf '%s  %s\n' "$slang_sha" "$slang_archive" | sha256sum --check
tar -xzf "$slang_archive" -C "$backend/.venv"

"$backend_python" - <<'PY'
import torch
import ncore
import viser
from threedgrut.model.model import MixtureOfGaussians
if not torch.cuda.is_available():
    raise SystemExit('Installation completed, but PyTorch cannot access CUDA.')
print('Runtime ready:', torch.__version__, torch.cuda.get_device_name(), 'Viser', viser.__version__)
PY
# Activation is standard shell environment, not a separate KIWI config system.
runtime_env="$backend/.venv/kiwi-cuda-env.sh"
{
    printf 'export CUDA_HOME=%q\n' "$CUDA_HOME"
    # Preserve the compiler/toolkit pairing used during installation, including
    # the host compiler bundled with Conda/Pixi CUDA distributions.
    printf 'export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:$PATH"\n'
    printf 'export CPATH=%q\n' "${CPATH:-}"
    printf 'export LIBRARY_PATH=%q\n' "${LIBRARY_PATH:-}"
    printf 'export LD_LIBRARY_PATH=%q\n' "${LD_LIBRARY_PATH:-}"
    printf 'export TORCH_CUDA_ARCH_LIST=%q\n' "$TORCH_CUDA_ARCH_LIST"
    printf 'export MAX_JOBS=%q\n' "$MAX_JOBS"
} > "$runtime_env"
if ! grep -q 'kiwi-cuda-env.sh' "$backend/.venv/bin/activate"; then
    printf '\n. "$VIRTUAL_ENV/kiwi-cuda-env.sh"\n' >> "$backend/.venv/bin/activate"
fi
printf '\nReady. Activate with: source %q\n' "$backend/.venv/bin/activate"
printf 'Backend: %s\n' "$backend"
printf 'Python: %s\n' "$backend_python"
printf 'kiwi real2simviz <asset> --backend-repo %q (or export KIWI_3DGRUT=%q).\n' "$backend" "$backend"

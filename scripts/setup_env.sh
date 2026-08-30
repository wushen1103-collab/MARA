#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/env

VENV_DIR="${VENV_DIR:-.venv}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_FALLBACK_INDEX_URLS="${PIP_FALLBACK_INDEX_URLS:-https://pypi.tuna.tsinghua.edu.cn/simple https://pypi.org/simple}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="*"
export no_proxy="*"

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v /usr/bin/python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="/usr/bin/python3.10"
  else
    PYTHON_BIN="python3"
  fi
fi

if [ -x "${VENV_DIR}/bin/python" ]; then
  existing_version="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [ "${existing_version}" != "3.10" ]; then
    echo "Existing ${VENV_DIR} uses Python ${existing_version}; rebuilding for Python 3.10"
    rm -rf "${VENV_DIR}"
  fi
fi

if [ -d "${VENV_DIR}" ] && { [ ! -x "${VENV_DIR}/bin/python" ] || [ ! -f "${VENV_DIR}/bin/activate" ]; }; then
  echo "Existing ${VENV_DIR} is incomplete; rebuilding"
  rm -rf "${VENV_DIR}"
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m ensurepip --upgrade >/dev/null 2>&1 || true

if ! python -m pip --version >/dev/null 2>&1; then
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/mara_get_pip.py
  get_pip_ok=0
  for index_url in "${PIP_INDEX_URL}" ${PIP_FALLBACK_INDEX_URLS}; do
    echo "Trying get-pip with ${index_url}"
    if python /tmp/mara_get_pip.py --index-url "${index_url}"; then
      get_pip_ok=1
      PIP_INDEX_URL="${index_url}"
      break
    fi
  done
  if [ "${get_pip_ok}" != "1" ]; then
    echo "get-pip failed for all configured indexes" >&2
    exit 1
  fi
fi

python -m pip config set global.index-url "${PIP_INDEX_URL}" || true

pip_install() {
  local ok=0
  for index_url in "${PIP_INDEX_URL}" ${PIP_FALLBACK_INDEX_URLS}; do
    echo "Trying pip install from ${index_url}: $*"
    if python -m pip install --proxy "" -i "${index_url}" "$@"; then
      ok=1
      PIP_INDEX_URL="${index_url}"
      python -m pip config set global.index-url "${PIP_INDEX_URL}" || true
      break
    fi
  done
  if [ "${ok}" != "1" ]; then
    echo "pip install failed for all configured indexes: $*" >&2
    exit 1
  fi
}

pip_install -U pip setuptools wheel
pip_install -r requirements.txt

if [ "${INSTALL_OPTIONAL:-${INSTALL_OPTIONAL_TDC:-0}}" = "1" ]; then
  echo "Installing optional dataset and scalability dependencies"
  if ! timeout "${OPTIONAL_TDC_TIMEOUT:-900}" python -m pip install --proxy "" -i "${PIP_INDEX_URL}" -r requirements-optional.txt; then
    echo "Optional dependency installation failed or timed out; continuing with core dependencies" >&2
  fi
fi

if [ "${INSTALL_GRAPH:-0}" = "1" ]; then
  echo "Installing optional neural graph and molecular-backbone dependencies"
  pip_install -r requirements-graph.txt
fi

python scripts/env_witness.py | tee logs/env/env_witness.log

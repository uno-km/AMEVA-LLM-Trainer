#!/usr/bin/env bash
# ==============================================================================
# AMEVA-LLM-Trainer Unix/Linux/macOS Setup Script
# ==============================================================================

set -e

echo "============================================================"
echo "   AMEVA-LLM-Trainer Unix Environment Setup"
echo "============================================================"
echo ""

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment (venv)..."
    python3 -m venv venv
else
    echo "[INFO] Virtual environment already exists."
fi

# 2. Activate venv
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# 3. Upgrade pip
echo "[INFO] Upgrading pip..."
pip install --upgrade pip

# 4. Install dependencies
echo "[INFO] Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# 5. Create necessary directories
for d in dataset outputs logs models/gguf configs; do
    mkdir -p "$ROOT/$d"
    echo "[INFO] Ensured directory: $d"
done

# 6. Create HF_HOME cache directory
HF_HOME_DIR="$HOME/.ameva/models/llm"
mkdir -p "$HF_HOME_DIR"
echo "[INFO] Created HF_HOME cache directory: $HF_HOME_DIR"

# 7. Set environment variable in shell profile
SHELL_RC="$HOME/.bashrc"
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

if ! grep -q "HF_HOME" "$SHELL_RC" 2>/dev/null; then
    echo "export HF_HOME=\"$HF_HOME_DIR\"" >> "$SHELL_RC"
    echo "[INFO] HF_HOME added to $SHELL_RC"
fi

echo ""
echo "[SUCCESS] Setup process completed successfully!"
echo "To run the application, execute: ./run_cli.sh"
echo ""

export UV_PROJECT_ENVIRONMENT=/pvc/SemShift/.venv
export VIRTUAL_ENV=/pvc/SemShift/.venv
export UV_CACHE_DIR=/pvc/.cache/uv
export UV_PYTHON_INSTALL_DIR=/pvc/.cache/uv/pythons
export UV_PYTHON_CACHE_DIR=/pvc/.cache/uv/python
export UV_LINK_MODE=copy
export NLTK_DATA='/pvc/data/nltk'
mkdir -p $NLTK_DATA

export PYTHONUNBUFFERED=1 # print everything immediately

echo "Installing NLTK"
uv run python -c "import nltk; nltk.download('wordnet')"
# echo "Installing SpaCy"
# uv run python -m spacy download en_core_web_sm

cd /workspace/SemShift

uv run python build_comprehensive_kwic_dataset.py -i stimuli.csv -o /pvc/SemShift/output.csv -j /pvc/SemShift/output.json --checkpoint-dir /pvc/SemShift/checkpoints --checkpoint-freq 10 --verbs ""

echo "Done!"

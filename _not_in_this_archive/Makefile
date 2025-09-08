# MonoX Makefile

.PHONY: help test smoke-test install clean

help:
	@echo "MonoX Training Commands"
	@echo "======================"
	@echo "make test          - Run smoke test (config validation)"
	@echo "make smoke-test    - Alias for test"
	@echo "make install       - Install dependencies"
	@echo "make clean         - Clean output directories"
	@echo ""
	@echo "Training Examples:"
	@echo "  python3 train.py dataset.path=/data/videos dataset=ffs"
	@echo "  python3 train.py dataset=ffs training.steps=10 launcher=local"

test:
	@echo "🚀 Running MonoX smoke test..."
	./scripts/test_train.sh

smoke-test: test

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

clean:
	@echo "🧹 Cleaning output directories..."
	rm -rf logs/ previews/ checkpoints/
	@echo "✅ Clean complete"
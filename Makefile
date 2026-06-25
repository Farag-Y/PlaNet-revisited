.PHONY: train-vast train-vast-py

train-vast:
	@bash scripts/train_vastai.sh $(ARGS)

train-vast-py:
	@uv run --group vastai python scripts/train_vastai.py $(ARGS)

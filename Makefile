.PHONY: train-vast

train-vast:
	@uv run --group vastai python scripts/train_vastai.py $(ARGS)

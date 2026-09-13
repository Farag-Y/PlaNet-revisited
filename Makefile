.PHONY: train-vast r2-cleanup

train-vast:
	@uv run --group vastai python scripts/train_vastai.py $(ARGS)

r2-cleanup:
	@uv run python scripts/r2_cleanup.py $(ARGS)

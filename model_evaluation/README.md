  # 1. Process dataset
  python test_ms_marco.py process --config example_config.json --limit-rows 100

  # 2. Evaluate (saves to example_config_evaluation_results.txt)
  python test_ms_marco.py evaluate --config example_config.json --limit-rows 100

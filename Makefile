bias_prob: 
	python scripts/probing.py --seed 42 --name="vanilla-linear" --task_config_file="hate_identifier_words.json" --model_name_or_path="/home/zakizadeh/contextualizing-hate-speech-models-with-explanations/runs/majority_gab_es_vanilla_bal_seed_0" --overwrite_cache

debiased_bias_prob:
	python scripts/probing.py --seed 42 --name="debiased-linear" --task_config_file="hate_identifier_words.json" --model_name_or_path="/home/zakizadeh/contextualizing-hate-speech-models-with-explanations/runs/majority_gab_es_reg_nb5_h5_is_bal_pos_seed_0" --overwrite_cache

mlp_bias_prob: 
	python scripts/probing.py --seed 42 --name="vanilla-mlp" --task_config_file="hate_identifier_words.json" --model_name_or_path="/home/zakizadeh/contextualizing-hate-speech-models-with-explanations/runs/majority_gab_es_vanilla_bal_seed_0" --overwrite_cache --probe_type="mlp"

mlp_debiased_bias_prob:
	python scripts/probing.py --seed 42 --name="debiased-mlp" --task_config_file="hate_identifier_words.json" --model_name_or_path="/home/zakizadeh/contextualizing-hate-speech-models-with-explanations/runs/majority_gab_es_reg_nb5_h5_is_bal_pos_seed_0" --overwrite_cache --probe_type="mlp"

mnli:
	python scripts/probing.py --seed 42 --name="baseline" --task_config_file="mnli_lex_class_sub.json" --model_name_or_path="bert-base-cased" --overwrite_cache 

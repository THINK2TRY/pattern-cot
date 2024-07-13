python refine_response.py \
--input_file ../data/math/math_train.jsonl \
--output_file ../data/math/math_train_refined.jsonl \
--sampler_url http://172.18.192.72:8080/generate \
--refine_mode pattern \
--num_process 50 \
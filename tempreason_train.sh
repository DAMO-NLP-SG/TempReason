sudo python run_seq2seq_qa.py   --model_name_or_path t5-base --context_column fact_context --question_column question   --answer_column text_answers   --per_device_eval_batch_size 2    --per_device_train_batch_size 8 --save_steps 300000   --learning_rate 2e-5   --num_train_epochs 3    --max_seq_length 768 --max_answer_length 32   --doc_stride 128   --output_dir checkpoints/t5-base-ft --eval_accumulation_steps 1 --predict_with_generate     --eval_steps 300000  --logging_steps 300000    --cache_dir "./cache/"     --train_file "temp_reason/train_l2.json"  --test_file "temp_reason/test_l2.json"     --overwrite_output_dir --dlc False --num_return_sequences 1 --do_predict --do_train      



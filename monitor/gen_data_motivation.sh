# 这是motivation的实验设计
# python monitor/my_motivation.py --output final_motivation.tsv --exp-name motivation_goodput_vs_num_devices_cv --workload=synthetic --policy mp-search-sep --num-devices 8 --rate 10 --model-type=mixed --num_devices_per_node 4
# python monitor/my_motivation.py --output final_motivation.tsv --exp-name motivation_goodput_vs_num_devices_cv --workload=synthetic --policy heuristic-dynamic --num-devices 8 --rate 10 --model-type=mixed --num_devices_per_node 4
# python monitor/my_motivation.py --output final_motivation.tsv --exp-name motivation_goodput_vs_num_devices_cv --workload=synthetic --policy sr-replace-60 --num-devices 8 --rate 10 --model-type=mixed --num_devices_per_node 4

# python monitor/my_motivation.py --output final_motivation.tsv --exp-name motivation_goodput_vs_num_devices_cv --workload=mixed --policy mp-search-sep --num-devices 8 --cv_scale 9 --rate 10 --mixed_ratio 0.4 --model-type=mixed --num_devices_per_node 8
# python monitor/my_motivation.py --output final_motivation.tsv --exp-name motivation_goodput_vs_num_devices_cv --workload=mixed --policy dqn-dynamic --num-devices 8 --cv_scale 9 --rate 10 --mixed_ratio 0.4 --model-type=mixed --num_devices_per_node 8
# python monitor/my_motivation.py --output final_motivation.tsv --exp-name motivation_goodput_vs_num_devices_cv --workload=mixed --policy sr-replace-60 --num-devices 8 --cv_scale 9 --rate 10 --mixed_ratio 0.4 --model-type=mixed --num_devices_per_node 8

python monitor/my_motivation.py --output final_motivation.tsv --exp-name motivation_goodput_vs_num_devices_cv --workload=azure_v1 --policy mp-search-sep --num-devices 20 --cv_scale 9 --model-type=mixed --num_devices_per_node 8
# python monitor/my_motivation.py --output final_motivation.tsv --exp-name motivation_goodput_vs_num_devices_cv --workload=azure_v1 --policy dqn-dynamic --num-devices 20 --cv_scale 9 --model-type=mixed --num_devices_per_node 8
# python monitor/my_motivation.py --output final_motivation.tsv --exp-name motivation_goodput_vs_num_devices_cv --workload=azure_v1 --policy sr-replace-60 --num-devices 20 --cv_scale 9 --model-type=mixed --num_devices_per_node 8


#echo torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py --config_path ./config_feb15_2013.yaml
#torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py --config_path ./config_feb15_2013.yaml --wandb

echo torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py --config_path ./config_feb16_2013.yaml
torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py --config_path ./config_feb16_2013.yaml --wandb

echo torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py --config_path ./config_feb17_2013.yaml
torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py --config_path ./config_feb17_2013.yaml --wandb

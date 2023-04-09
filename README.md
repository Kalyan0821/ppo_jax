### PPO implemented in Jax.
To train:
python train.py --config ./configs/config.json

To train with multiple hparams in parallel:
python train_parallel.py --config ./configs/config.json

To ensure vmap/non-vmap consistency:
JAX_ENABLE_X64=True python train.py --config ./configs/config.json
JAX_ENABLE_X64=True python train_parallel.py --config configs/config.json


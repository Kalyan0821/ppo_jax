import gymnax
import jax
import optax
import jax.numpy as jnp
import numpy as np
from flax.training.checkpoints import save_checkpoint
from model import NN
from learning import learn_policy


# env_name = "CartPole-v1"
# env_name = "SpaceInvaders-MinAtar"
env_name = "MountainCar-v0"

SEED = 0
total_experience = 500000

lr_begin = 5e-3
# lr_end = 0
lr_end = 5e-4

n_agents = 16
horizon = 32
n_epochs = 64
minibatch_size = 128
# minibatch_size = n_agents*horizon  # for 1 minibatch per epoch
hidden_layer_sizes = (64, 64)
normalize_advantages = True
anneal = True

permute_batches = True
clip_epsilon = 0.2
# entropy_coeff = 0.01
entropy_coeff = 0.003

val_loss_coeff = 0.5
discount = 0.99
gae_lambda = 0.95
n_eval_agents = 164
eval_discount = 1.0
eval_iter = 40
checkpoint_iter = 40
checkpoint_dir = "./checkpoints"


assert minibatch_size <= n_agents*horizon

env, env_params = gymnax.make(env_name)
key = jax.random.PRNGKey(SEED)
key, subkey_env, subkey_model = jax.random.split(key, 3)
state_feature, _ = env.reset(subkey_env)
n_actions = env.action_space().n

model = NN(hidden_layer_sizes=hidden_layer_sizes, 
           n_actions=n_actions, 
           single_input_shape=state_feature.shape)
model_params = model.init(subkey_model, jnp.zeros(state_feature.shape))

n_outer_iters = total_experience // (n_agents * horizon)
n_iters_per_epoch = n_agents*horizon // minibatch_size  # num_minibatches
n_inner_iters = n_epochs * n_iters_per_epoch 

print("\nState feature shape:", state_feature.shape)
print("Action space:", n_actions)
print("Minibatches per epoch:", n_iters_per_epoch)
print("Outer steps:", n_outer_iters, '\n')


if anneal:
    lr = optax.linear_schedule(init_value=lr_begin, 
                               end_value=lr_end, 
                               transition_steps=n_outer_iters*n_inner_iters)
optimizer = optax.adam(lr)

evals = learn_policy(env,
                     key,
                     model_params, 
                     model,
                     optimizer,
                     n_actions,
                     n_outer_iters,
                     horizon,
                     n_epochs,
                     n_agents,
                     minibatch_size,
                     val_loss_coeff,
                     entropy_coeff,
                     anneal,
                     normalize_advantages,
                     permute_batches,
                     clip_epsilon,
                     discount,
                     gae_lambda,
                     eval_discount,
                     n_eval_agents,
                     eval_iter)

print("Summary:")
for experience in evals:
    avg_return = np.mean(evals[experience])
    std_return = np.std(evals[experience])
    print(f"Experience: {experience}. Avg. return = {avg_return}, std={std_return}")



# if (outer_iter+1) % checkpoint_iter == 0:
#     experience = (outer_iter+1) * n_agents * horizon
#     checkpoint = {"model": model, 
#                   "params": model_params}
#     print("Saving ...")
#     save_checkpoint(checkpoint_dir, checkpoint, experience, 
#                     prefix=env_name+"_ckpt_", 
#                     keep_every_n_steps=checkpoint_iter, 
#                     overwrite=True)

# print('-------------')




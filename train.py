import gymnax
import jax
import optax
import jax.numpy as jnp
import numpy as np
from flax.training.checkpoints import save_checkpoint
from model import NN
from learning import learn_policy


env_name = "CartPole-v1"

SEED = 0
total_experience = 200000
lr = 1e-2
n_agents = 8

# horizon = 128
horizon = 4

n_epochs = 16
minibatch_size = 16
# minibatch_size = n_agents*horizon  # for 1 minibatch per epoch
hidden_layer_sizes = (64, 64)
normalize_advantages = True
anneal = True
permute_batches = True
clip_epsilon = 0.2
entropy_coeff = 0.01
val_loss_coeff = 0.5
discount = 0.99
gae_lambda = 0.95

n_eval_agents = 16
eval_discount = 1.0
eval_iter = 20
checkpoint_iter = 20
checkpoint_dir = "./checkpoints"


assert minibatch_size <= n_agents*horizon
env, env_params = gymnax.make(env_name)

key = jax.random.PRNGKey(SEED)
key, subkey_env, subkey_model = jax.random.split(key, 3)
state_feature, _ = env.reset(subkey_env)
n_features = state_feature.size
n_actions = env.action_space().n


model = NN(hidden_layer_sizes=hidden_layer_sizes, n_actions=n_actions)
model_params = model.init(subkey_model, jnp.zeros(n_features))

n_outer_iters = total_experience // (n_agents * horizon)  # loop_steps
n_iters_per_epoch = n_agents*horizon // minibatch_size
n_inner_iters = n_epochs * n_iters_per_epoch

if anneal:
    lr = optax.linear_schedule(init_value=lr, 
                               end_value=0, 
                               transition_steps=n_outer_iters*n_inner_iters)
optimizer = optax.adam(lr)

evals = dict()
print("Outer steps:", n_outer_iters)


learn_policy(env,
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
             n_eval_agents,
             eval_iter)




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




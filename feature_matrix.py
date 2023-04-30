import gymnax
import jax
import optax
import jax.numpy as jnp
import numpy as np
import argparse
import json
from model import NN, SeparateNN
from learning import sample_batch
from flax.training.checkpoints import restore_checkpoint
from test import evaluate
from functools import partial
from jax.config import config as cfg
cfg.update("jax_enable_x64", True)  # to ensure vmap/non-vmap consistency


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='JSON file path')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = json.load(f)

env_name = config["env_name"]
SEED = config["SEED"]
N_SEEDS = config["N_SEEDS"]
total_experience = int(config["total_experience"])
n_agents = config["n_agents"]
horizon = config["horizon"]
n_epochs = config["n_epochs"]
minibatch_size = config["minibatch_size"]
# minibatch_size = n_agents*horizon  # for 1 minibatch per epoch
assert minibatch_size <= n_agents*horizon
hidden_layer_sizes = tuple(config["hidden_layer_sizes"])
architecture = config["architecture"]
activation = config["activation"]
n_eval_agents = config["n_eval_agents"]
eval_iter = config["eval_iter"]

env, env_params = gymnax.make(env_name)
vecEnv_reset = jax.vmap(env.reset, in_axes=(0,))
vecEnv_step = jax.vmap(env.step, in_axes=(0, 0, 0))
example_state_feature, _ = env.reset(jax.random.PRNGKey(0))
n_actions = env.action_space().n

n_outer_iters = total_experience // (n_agents * horizon)
n_iters_per_epoch = n_agents*horizon // minibatch_size  # num_minibatches
n_inner_iters = n_epochs * n_iters_per_epoch 

print("\nState feature shape:", example_state_feature.shape)
print("Action space:", n_actions)
print("Minibatches per epoch:", n_iters_per_epoch)
print("Outer steps:", n_outer_iters, '\n')

################# CAN VMAP OVER CHOICES OF: #################
clip_epsilon = config["clip_epsilon"]
entropy_coeff = config["entropy_coeff"]
val_loss_coeff = config["val_loss_coeff"]
clip_grad = float(config["clip_grad"])
gae_lambda = config["gae_lambda"]
lr_begin = config["lr_begin"]
lr_end = config["lr_end"]
anneal = config["anneal_epsilon"]
normalize_advantages = config["normalize_advantages"]
discount = config["discount"]
eval_discount = config["eval_discount"]
#############################################################

@jax.jit
@partial(jax.vmap, in_axes=(0, None, 0))
@partial(jax.vmap, in_axes=(None, None, 0))
@partial(jax.vmap, in_axes=(None, None, 0))
def compute_features(key, optimal_params, model_params):
    """ To vmap over a hparam, include it as an argument and 
    modify the decorators appropriately """

    if architecture == "shared":
        model = NN(hidden_layer_sizes=hidden_layer_sizes, 
                n_actions=n_actions, 
                single_input_shape=example_state_feature.shape,
                activation=activation,
                return_feature=True)
    elif architecture == "separate":
        model = SeparateNN(hidden_layer_sizes=hidden_layer_sizes, 
                        n_actions=n_actions, 
                        single_input_shape=example_state_feature.shape,
                        activation=activation,
                        return_feature=True)

    key, *agents_subkeyReset = jax.random.split(key, n_agents+1)
    agents_subkeyReset = jnp.asarray(agents_subkeyReset)
    agents_stateFeature, agents_state = vecEnv_reset(agents_subkeyReset)  # (n_agents, n_features), (n_agents, .)

    _, _, batch, key = sample_batch(agents_stateFeature,
                                    agents_state,
                                    vecEnv_step,
                                    key,
                                    optimal_params, 
                                    model,
                                    n_actions,
                                    horizon,
                                    n_agents,
                                    discount,
                                    gae_lambda)
    assert batch["states"].shape[:2] == (horizon, n_agents)  # (horizon, n_agents, n_features)

    reshaped_batch = jax.tree_map(lambda x: jnp.reshape(x, (-1,)+x.shape[2:]), batch)  # each: (horizon*n_agents, ...)
    states = reshaped_batch["states"]  # (horizon*n_agents, n_features)
    assert states.shape[0] == horizon * n_agents

    _, _, model_features = model.apply(model_params, states)  # (horizon*n_agents, final_hidden_layer_size)

    return states, model_features
    

if __name__ == "__main__":
    key0 = jax.random.PRNGKey(SEED)
    keys = jnp.array([key0, *jax.random.split(key0, N_SEEDS-1)])

    optimal_params = restore_checkpoint(f"./saved_models/{architecture}", None, 0, prefix=env_name+'_')
    model_params = restore_checkpoint(f"./saved_models/{architecture}", None, 0, prefix=env_name+"-vmap_")


    states, features = compute_features(keys, optimal_params, model_params)
    print(states.shape, features.shape)

    # print("Done. Result shape:", result["avg_returns"].shape, '\n')

    # # Save for plotting
    # save_indices = result["std_returns"][(0,)*len(hparams)] > -0.5
    # with open(f"./plotting/{architecture}/{env_name}.npy", 'wb') as f:
    #     np.save(f, result["avg_returns"][..., save_indices])
    # with open(f"./plotting/{architecture}/{env_name}_exps.npy", 'wb') as f:
    #     np.save(f, result["experiences"][(0,)*len(hparams)+(save_indices,)])



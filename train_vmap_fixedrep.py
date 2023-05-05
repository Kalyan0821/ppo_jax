import gymnax
import jax
import optax
import jax.numpy as jnp
import numpy as np
import argparse
import json
from model import NN, SeparateNN, SoftMaxLayer
from learning import sample_batch, batch_epoch
from flax.training.checkpoints import restore_checkpoint
from test import evaluate
from functools import partial
from collections import OrderedDict
import wandb
import datetime
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
# @partial(jax.vmap, in_axes=(0,))
# def train_once(key):
@partial(jax.vmap, in_axes=(0, None, None, None))
@partial(jax.vmap, in_axes=(None, 0, None, None))
@partial(jax.vmap, in_axes=(None, None, 0, None))
def train_once(key, entropy_coeff, clip_epsilon, base_params):
    """ To vmap over a hparam, include it as an argument and 
    modify the decorators appropriately """

    if architecture == "shared":
        model = NN(hidden_layer_sizes=hidden_layer_sizes, 
                n_actions=n_actions, 
                single_input_shape=example_state_feature.shape,
                activation=activation,
                freeze_representation=True)
    elif architecture == "separate":
        model = SeparateNN(hidden_layer_sizes=hidden_layer_sizes, 
                        n_actions=n_actions, 
                        single_input_shape=example_state_feature.shape,
                        activation=activation,
                        freeze_representation=True)
    softmax_model = SoftMaxLayer(n_actions=n_actions)

    key, subkey_model = jax.random.split(key)
    softmax_params = softmax_model.init(subkey_model, jnp.zeros(hidden_layer_sizes[-1]))
    model_params = base_params
    model_params["params"]["logits"] = softmax_params["params"]["z"]

    lr = optax.linear_schedule(init_value=lr_begin, 
                               end_value=lr_end, 
                               transition_steps=n_outer_iters*n_inner_iters)
    
    max_norm = jnp.where(clip_grad > 0, clip_grad, jnp.inf).astype(jnp.float32)
    optimizer = optax.chain(optax.clip_by_global_norm(max_norm=max_norm), 
                            optax.adam(lr))

    key, *agents_subkeyReset = jax.random.split(key, n_agents+1)
    agents_subkeyReset = jnp.asarray(agents_subkeyReset)
    agents_stateFeature, agents_state = vecEnv_reset(agents_subkeyReset)  # (n_agents, n_features), (n_agents, .)
    optimizer_state = optimizer.init(model_params)

    initial_carry = {"key": key,
                     "model_params": model_params,
                     "optimizer_state": optimizer_state,
                     "agents_stateFeature": agents_stateFeature,
                     "agents_state": agents_state}
    def scan_function(carry, idx):
        def f_true(carry):
            key, key_eval = jax.random.split(carry["key"])
            returns = evaluate(env, key_eval, carry["model_params"], model, n_actions, n_eval_agents, eval_discount)
            avg_return, std_return = jnp.mean(returns), jnp.std(returns)
            return avg_return, std_return, key            
        def f_false(carry):
            if "Acrobot" in env_name or "CartPole" in env_name or "Mountain" in env_name:
                return -1.0, -1.0, carry["key"]
            return jnp.array(-1.0, dtype=jnp.float32), jnp.array(-1.0, dtype=jnp.float32), carry["key"]


        append_to = dict()
        append_to["steps"], append_to["experiences"] = idx, idx*n_agents*horizon
        append_to["avg_returns"], append_to["std_returns"], key = jax.lax.cond(idx % eval_iter == 0, f_true, f_false, carry)

        carry["agents_stateFeature"], carry["agents_state"], batch, key = sample_batch(carry["agents_stateFeature"],
                                                                    carry["agents_state"],
                                                                    vecEnv_step,
                                                                    key,
                                                                    carry["model_params"], 
                                                                    model,
                                                                    n_actions,
                                                                    horizon,
                                                                    n_agents,
                                                                    discount,
                                                                    gae_lambda)
        # (agents_stateFeature, agents_state) returned were the last ones seen in this step, and will initialize the next step
        assert carry["agents_stateFeature"].shape[0] == n_agents

        alpha = jnp.where(anneal, (1-idx/n_outer_iters), 1.0)

        for _ in range(n_epochs):
            key, permutation_key = jax.random.split(key)

            (carry["model_params"], carry["optimizer_state"], minibatch_losses, 
            ppo_losses, val_losses, ent_bonuses, clip_trigger_fracs, approx_kls) = batch_epoch(
                                                        batch,
                                                        permutation_key,
                                                        carry["model_params"], 
                                                        model,
                                                        carry["optimizer_state"],
                                                        optimizer,
                                                        n_actions,
                                                        horizon,
                                                        n_agents,
                                                        minibatch_size,
                                                        val_loss_coeff,
                                                        entropy_coeff,
                                                        normalize_advantages,
                                                        clip_epsilon*alpha)

        carry["key"] = key
        append_to["minibatch_losses"] = jnp.mean(minibatch_losses)
        append_to["ppo_losses"] = jnp.mean(ppo_losses)
        append_to["val_losses"] = jnp.mean(val_losses)
        append_to["ent_bonuses"] = jnp.mean(ent_bonuses)
        append_to["clip_trigger_fracs"] = jnp.mean(clip_trigger_fracs)
        append_to["approx_kls"] = jnp.mean(approx_kls)

        return carry, append_to

    carry, result = jax.lax.scan(scan_function, initial_carry, xs=jnp.arange(n_outer_iters))
    
    # One more eval
    _, key_eval = jax.random.split(carry["key"])
    returns = evaluate(env, key_eval, carry["model_params"], model, n_actions, n_eval_agents, eval_discount)
    avg_return, std_return = jnp.mean(returns), jnp.std(returns)
    result["steps"] = jnp.append(result["steps"], result["steps"][-1] + 1)    
    result["experiences"] = jnp.append(result["experiences"], result["experiences"][-1] + n_agents*horizon)
    result["avg_returns"] = jnp.append(result["avg_returns"], avg_return)
    result["std_returns"] = jnp.append(result["std_returns"], std_return)

    return result, carry
    

if __name__ == "__main__":
    key0 = jax.random.PRNGKey(SEED)
    keys = jnp.array([key0, *jax.random.split(key0, N_SEEDS-1)])


    ################# VMAP OVER: #################
    # hparams = OrderedDict({"keys": keys})
    hparams = OrderedDict({"keys": keys, 
                           "ent": jnp.array( [0.0, 0.01, 0.05, 0.1, 0.4, 0.8] ),
                           "clip": jnp.array( [0.005, 0.02, 0.08, 0.2, 0.5, 0.8, 1e6] )})
    ##############################################
    SAVE_ARRAY = True
    NOREG_BASE = False

    hparam_names = list(hparams.keys())
    assert hparam_names[0] == "keys"
    
    model_params_vmap = restore_checkpoint(f"./saved_models/{architecture}", None, 0, prefix=env_name+"-vmap_")
    if NOREG_BASE:
        base_params = jax.tree_map(lambda x: x[0, 0, -1], model_params_vmap)
    else:
        base_params = jax.tree_map(lambda x: x[0, 1, 2], model_params_vmap)

    # Train:
    result, carry = train_once(*hparams.values(), base_params)
    print("Done. Result shape:", result["avg_returns"].shape, '\n')

    # Save for plotting
    if SAVE_ARRAY and len(hparams) == 3:
        save_indices = result["std_returns"][(0,)*len(hparams)] > -0.5
        if NOREG_BASE:
            name = env_name+"_noregbase"
        else:
            name = env_name+"_optimalbase"
            
        with open(f"./plotting/{architecture}/fixed_rep/{name}.npy", 'wb') as f:
            np.save(f, result["avg_returns"][..., save_indices])
        with open(f"./plotting/{architecture}/fixed_rep/{name}_exps.npy", 'wb') as f:
            np.save(f, result["experiences"][(0,)*len(hparams)+(save_indices,)])

import gymnax
import jax
import optax
import jax.numpy as jnp
import argparse
import json
from typing import Callable
from model import NN, SeparateNN
from flax.core.frozen_dict import FrozenDict
from flax.training.checkpoints import restore_checkpoint
from test import evaluate
from model import SoftMaxLayer
import flax.linen as nn
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
# n_agents = config["n_agents"]
# horizon = config["horizon"]
n_epochs = config["n_epochs"]
# minibatch_size = config["minibatch_size"]
# minibatch_size = n_agents*horizon  # for 1 minibatch per epoch
# assert minibatch_size <= n_agents*horizon
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

# n_outer_iters = total_experience // (n_agents * horizon)
# n_iters_per_epoch = n_agents*horizon // minibatch_size  # num_minibatches
# n_inner_iters = n_epochs * n_iters_per_epoch 

print("\nState feature shape:", example_state_feature.shape)
print("Action space:", n_actions)
# print("Minibatches per epoch:", n_iters_per_epoch)
# print("Outer steps:", n_outer_iters, '\n')

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


def softmax_loss_function(softmax_params: FrozenDict,
                          minibatch: dict[str, jnp.array],
                          softmax_model: SoftMaxLayer,
                          n_actions: int,
                          minibatch_size: int):
    
    model_features = minibatch["model_features"]  # (minibatch_size, n_features)
    optimal_probs = minibatch["optimal_probs"]

    log_probs = softmax_model.apply(softmax_params, model_features)
    assert log_probs.shape == (minibatch_size, n_actions)

    cross_entropy_losses = -1 * jnp.sum(optimal_probs*log_probs, axis=1)
    assert cross_entropy_losses.shape == (minibatch_size,)
    loss = jnp.mean(cross_entropy_losses)
    return loss


val_and_grad_function = jax.jit(jax.value_and_grad(softmax_loss_function, argnums=0),
                                static_argnums=(2, 3, 4))


@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0))
def greedy_action(n_actions, probs):
    assert probs.shape == (n_actions,)
    action = jnp.argmax(probs)
    return action


@partial(jax.jit, static_argnums=(2, 5, 6, 7, 8))
def sample_batch(agents_stateFeature: jnp.array,
                 agents_state: jnp.array,
                 vecEnv_step: Callable,
                 key: jax.random.PRNGKey,
                 optimal_params: FrozenDict,
                 model: NN,
                 n_actions: int,
                 horizon: int,
                 n_agents: int):
    
    initial_carry = {"agents_stateFeature": agents_stateFeature, 
                     "agents_state": agents_state, 
                     "key": key}
    def scan_function(carry, x=None):
        # (n_agents, n_actions), (n_agents, 1)
        agents_logProbs, _, _ = model.apply(optimal_params, carry["agents_stateFeature"])
        agents_probs = jnp.exp(agents_logProbs)
        assert agents_probs.shape == (n_agents, n_actions)

        # int, (n_agents,n_actions) --> (n_agents,)
        agents_action = greedy_action(n_actions, agents_probs)
        assert agents_action.shape == (n_agents,)

        append_to = {"states": carry["agents_stateFeature"],
                     "actions": agents_action,
                     "optimal_probs": agents_probs}

        ################### MDP TRANSITION ###################        
        carry["key"], *agents_subkeyMDP = jax.random.split(carry["key"], n_agents+1)
        agents_subkeyMDP = jnp.asarray(agents_subkeyMDP)  # (n_agents, ...)
        carry["agents_stateFeature"], carry["agents_state"], append_to["rewards"], _, _ = vecEnv_step(
                                                                agents_subkeyMDP, 
                                                                carry["agents_state"], 
                                                                agents_action)
        return carry, append_to

    carry, batch = jax.lax.scan(scan_function, initial_carry, xs=None, length=horizon)
    
    return_keys = ["states", "actions", "rewards", "optimal_probs"]
    return_batch = {k: batch[k] for k in return_keys}

    return carry["agents_stateFeature"], carry["agents_state"], return_batch, carry["key"]


@jax.jit
@partial(jax.vmap, in_axes=(0,    None, 0))
@partial(jax.vmap, in_axes=(None, None, 0))
@partial(jax.vmap, in_axes=(None, None, 0))
def behaviour_clone(key, optimal_params, model_params):
    lr = 1e-4
    n_agents = 4
    n_iters = 50
    horizon = env_params.max_steps_in_episode

    softmax_model = SoftMaxLayer(n_actions=n_actions)

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
    
    key, subkey_model = jax.random.split(key)
    softmax_params = softmax_model.init(subkey_model, jnp.zeros(hidden_layer_sizes[-1]))
    softmax_optimizer = optax.adam(lr)
    
    key, *agents_subkeyReset = jax.random.split(key, n_agents+1)
    agents_subkeyReset = jnp.asarray(agents_subkeyReset)
    agents_stateFeature, agents_state = vecEnv_reset(agents_subkeyReset)  # (n_agents, n_features), (n_agents, .)
    softmax_optimizer_state = softmax_optimizer.init(softmax_params)

    for _ in range(n_iters):
        agents_stateFeature, agents_state, batch, key = sample_batch(agents_stateFeature,
                                                                        agents_state,
                                                                        vecEnv_step,
                                                                        key,
                                                                        optimal_params, 
                                                                        model,
                                                                        n_actions,
                                                                        horizon,
                                                                        n_agents)
        assert batch["states"].shape[:2] == (horizon, n_agents)  # (horizon, n_agents, n_features)
        assert agents_stateFeature.shape[0] == n_agents

        reshaped_batch = jax.tree_map(lambda x: jnp.reshape(x, (-1,)+x.shape[2:]), batch)  # each: (horizon*n_agents, ...)
        assert reshaped_batch["states"].shape[0] == horizon*n_agents

        _, _, reshaped_batch["model_features"] = model.apply(model_params, reshaped_batch["states"])  # (horizon*n_agents, final_hidden_layer_size)
        assert reshaped_batch["model_features"].shape[0] == horizon*n_agents

        n_epochs = 5

        for _ in range(n_epochs):
            loss, gradient = val_and_grad_function(softmax_params, 
                                                    reshaped_batch,
                                                    softmax_model,
                                                    n_actions,
                                                    horizon*n_agents)
                
            softmax_param_updates, softmax_optimizer_state = softmax_optimizer.update(gradient,
                                                                                      softmax_optimizer_state,
                                                                                      softmax_params)            
            softmax_params = optax.apply_updates(softmax_params, softmax_param_updates)

        jax.debug.print("L {}", loss)
    
    # Eval
    _, key_eval = jax.random.split(key)
    model.return_feature = False
    composite_params = model_params
    composite_params["params"]["logits"] = softmax_params["params"]["z"]
    
    returns = evaluate(env, key_eval, composite_params, model, n_actions, n_eval_agents, eval_discount)
    avg_return = jnp.mean(returns)
    return softmax_params, softmax_optimizer_state, loss, avg_return


@jax.jit
@partial(jax.vmap, in_axes=(0,    None, 0))
@partial(jax.vmap, in_axes=(None, None, 0))
@partial(jax.vmap, in_axes=(None, None, 0))
def compute_features(key, optimal_params, model_params):
    """ To vmap over a hparam, include it as an argument and 
    modify the decorators appropriately """

    horizon = env_params.max_steps_in_episode
    n_agents = 1

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

    _, _, batch, _ = sample_batch(agents_stateFeature,
                                  agents_state,
                                  vecEnv_step,
                                  key,
                                  optimal_params, 
                                  model,
                                  n_actions,
                                  horizon,
                                  n_agents)
    assert batch["states"].shape[:2] == (horizon, n_agents)  # (horizon, n_agents, n_features)
    
    reshaped_batch = jax.tree_map(lambda x: jnp.reshape(x, (-1,)+x.shape[2:]), batch)  # each: (horizon*n_agents, ...)
    states = reshaped_batch["states"]  # (horizon*n_agents, n_features)
    actions = reshaped_batch["actions"]  # (horizon*n_agents,)
    assert states.shape[0] == actions.shape[0] == horizon*n_agents

    _, _, model_features = model.apply(model_params, states)  # (horizon*n_agents, final_hidden_layer_size)

    _, _, bad_batch, _ = sample_batch(agents_stateFeature,
                                      agents_state,
                                      vecEnv_step,
                                      key,
                                      model_params, 
                                      model,
                                      n_actions,
                                      horizon,
                                      n_agents)
    reshaped_bad_batch = jax.tree_map(lambda x: jnp.reshape(x, (-1,)+x.shape[2:]), bad_batch)  # each: (horizon*n_agents, ...)
    bad_rewards = reshaped_bad_batch["rewards"]  # (horizon*n_agents,)

    return states, model_features, actions, bad_rewards
    

if __name__ == "__main__":
    key0 = jax.random.PRNGKey(SEED)
    keys = jnp.array([key0, *jax.random.split(key0, N_SEEDS-1)])

    model_params = restore_checkpoint(f"./saved_models/{architecture}", None, 0, prefix=env_name+"-vmap_")
    optimal_params = jax.tree_map(lambda x: x[0, 1, 2], model_params)

    states, features, actions, model_rewards = compute_features(keys, optimal_params, model_params)
    print("Done.", states.shape, features.shape, actions.shape, model_rewards.shape)
    # print(jnp.cumsum(model_rewards[0, 1, 1]))
    # print(jnp.cumsum(model_rewards[0, -1, -1]))

    # # Save for plotting
    # with open(f"./plotting/{architecture}/feature_matrix/{env_name}_features.npy", 'wb') as f:
    #     np.save(f, features)
    # with open(f"./plotting/{architecture}/feature_matrix/{env_name}_actions.npy", 'wb') as f:
    #     np.save(f, actions)


    softmax_params, softmax_optimizer_state, loss, avg_return = behaviour_clone(keys, optimal_params, model_params)
    print(loss.shape, avg_return.shape)
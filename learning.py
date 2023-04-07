import jax
import optax
import jax.numpy as jnp
from functools import partial
from flax.core.frozen_dict import FrozenDict
from model import NN
from typing import Callable


def loss_function(model_params: FrozenDict,
                  minibatch: dict[str, jnp.array],
                  model: NN,
                  n_actions: int,
                  minibatch_size: int,
                  val_loss_coeff: float,
                  entropy_coeff: float,
                  normalize_advantages: bool,
                  clip_epsilon: float):
        
    states = minibatch["states"]  # (minibatch_size, n_features)
    actions = minibatch["actions"]  # (minibatch_size,)
    old_policy_log_likelihoods = minibatch["old_policy_log_likelihoods"]  # (minibatch_size,)
    advantages = minibatch["advantages"]  # (minibatch_size,)
    bootstrap_returns = minibatch["bootstrap_returns"]  # (minibatch_size,)
    assert states.shape[0] == minibatch_size

    # (minibatch_size, n_actions), (minibatch_size, 1)
    policy_log_probs, values = model.apply(model_params, states)
    values = jnp.squeeze(values, axis=-1)
    assert policy_log_probs.shape == (minibatch_size, n_actions)
    assert values.shape == (minibatch_size,)

    # (m, a) x (m,) => (m,) 
    @partial(jax.vmap, in_axes=0)
    def get_element(vector, idx):
        return vector[idx]    
    policy_log_likelihoods = get_element(policy_log_probs, actions)  # (minibatch_size,)
    assert policy_log_likelihoods.shape == old_policy_log_likelihoods.shape

    log_likelihood_ratios = policy_log_likelihoods - old_policy_log_likelihoods
    likelihood_ratios = jnp.exp(log_likelihood_ratios)
    clip_likelihood_ratios = jnp.clip(likelihood_ratios, 
                                         a_min=1-clip_epsilon, a_max=1+clip_epsilon)
    clip_trigger_frac = jnp.mean(jnp.abs(likelihood_ratios-1) > clip_epsilon)
    # Approximate avg. KL(old || new), from John Schulman blog
    approx_kl = jnp.mean(-log_likelihood_ratios + likelihood_ratios-1)
    
    if normalize_advantages:
        advantages = (advantages-jnp.mean(advantages)) / (jnp.std(advantages)+1e-8)

    policy_gradient_losses = likelihood_ratios * advantages  # (minibatch_size,)
    clip_losses = clip_likelihood_ratios * advantages

    ppo_loss = -1. * jnp.mean(jnp.minimum(policy_gradient_losses, clip_losses))
    val_loss = 0.5 * jnp.mean((values-bootstrap_returns)**2)    
    entropy_bonus = jnp.mean(-jnp.exp(policy_log_probs)*policy_log_probs) * n_actions

    loss = ppo_loss + val_loss_coeff*val_loss - entropy_coeff*entropy_bonus
    return loss, (ppo_loss, val_loss, entropy_bonus, clip_trigger_frac, approx_kl)


val_and_grad_function = jax.jit(jax.value_and_grad(loss_function, argnums=0, has_aux=True),
                                static_argnums=(2, 3, 4, 5, 6, 7))


@jax.jit
def permute(batch, key):
    """ batch: each jnp.array: (horizon, n_agents, ...) """

    _, key0, key1 = jax.random.split(key, 3)

    batch = jax.tree_map(lambda x: jax.random.permutation(key0, x, axis=0),
                         batch)
    batch = jax.tree_map(lambda x: jax.random.permutation(key1, x, axis=1),
                         batch)
    return batch


@partial(jax.jit, static_argnums=(3, 5, 6, 7, 8, 9, 10, 11, 12, 14))
def batch_epoch(batch: dict[str, jnp.array],
                permutation_key: jax.random.PRNGKey,
                model_params: FrozenDict,
                model: NN,
                optimizer_state: optax.OptState,
                optimizer: optax.GradientTransformation,
                n_actions: int,
                horizon: int,
                n_agents: int,
                minibatch_size: int,
                val_loss_coeff: float,
                entropy_coeff: float,
                normalize_advantages: bool,
                clip_epsilon: float,
                permute_batch: bool):
    """ batch: each jnp.array: (horizon, n_agents, ...) """

    if permute_batch:
        batch = permute(batch, permutation_key)
    assert batch["states"].shape[:2] == (horizon, n_agents)

    n_iters = horizon*n_agents // minibatch_size  # number of minibatches
    assert n_iters*minibatch_size == horizon*n_agents

    def reshape(x):
        new_shape = (n_iters, minibatch_size) + x.shape[2:]
        return jnp.reshape(x, new_shape)
    reshaped_batch = jax.tree_map(reshape, batch)  # each: (n_iters, ...)
    assert reshaped_batch["states"].shape[0] == n_iters


    initial_carry = {"model_params": model_params,
                     "optimizer_state": optimizer_state}
    def scan_function(carry, idx):
        minibatch = jax.tree_map(lambda x: x[idx], 
                                 reshaped_batch)
        assert minibatch["states"].shape[0] == minibatch_size

        (minibatch_loss, loss_info), gradient = val_and_grad_function(carry["model_params"],
                                                                      minibatch,
                                                                      model,
                                                                      n_actions,
                                                                      minibatch_size,
                                                                      val_loss_coeff,
                                                                      entropy_coeff,
                                                                      normalize_advantages,
                                                                      clip_epsilon)
        
        param_updates, carry["optimizer_state"] = optimizer.update(gradient,
                                                                   carry["optimizer_state"],
                                                                   carry["model_params"])
        carry["model_params"] = optax.apply_updates(carry["model_params"], param_updates)

        ppo_loss, val_loss, entropy_bonus, clip_trigger_frac, approx_kl = loss_info
        append = (minibatch_loss, ppo_loss, val_loss, entropy_bonus, clip_trigger_frac, approx_kl)
        return carry, append

    carry, result = jax.lax.scan(scan_function, initial_carry, xs=jnp.arange(n_iters))
    
    return (carry["model_params"], carry["optimizer_state"], *result)


@partial(jax.jit, static_argnums=(3, 4, 5))
@partial(jax.vmap, in_axes=(1, 1, 1, None, None, None), out_axes=(1, 1))
def batch_advantages_and_returns(values: jnp.array,
                                 rewards: jnp.array,
                                 next_is_terminal: jnp.array,
                                 horizon: int,
                                 discount: float,
                                 gae_lambda: float):
    
    assert rewards.shape == next_is_terminal.shape == (horizon,)
    assert values.shape == (horizon + 1,)

    initial_carry = {"next_advantage": 0.}
    def scan_function(carry, t):
        next_value = jnp.where(next_is_terminal[t], 0., values[t+1])
        next_advantage = jnp.where(next_is_terminal[t], 0., carry["next_advantage"])

        td_error = rewards[t] + discount*next_value - values[t]
        advantage = td_error + (discount*gae_lambda)*next_advantage
        bootstrap_return = advantage + values[t]

        append_to = {"advantages": advantage, 
                     "bootstrap_returns": bootstrap_return}
        
        carry["next_advantage"] = advantage

        return carry, append_to

    _, result = jax.lax.scan(scan_function, initial_carry, xs=jnp.arange(horizon), reverse=True)

    return result["advantages"], result["bootstrap_returns"]


@partial(jax.jit, static_argnums=(1,))
@partial(jax.vmap, in_axes=(0, None, 0, 0), out_axes=(0, 0))
def sample_action_and_logLikelihood(key, n_actions, probs, logProbs):
    assert probs.shape == logProbs.shape == (n_actions,)

    action = jax.random.choice(key, n_actions, p=probs)
    logLikelihood = logProbs[action]
    return (action, logLikelihood)


@partial(jax.jit, static_argnums=(2, 5, 6, 7, 8, 9, 10))
def sample_batch(agents_stateFeature: jnp.array,
                 agents_state: jnp.array,
                 vecEnv_step: Callable,
                 key: jax.random.PRNGKey,
                 model_params: FrozenDict,
                 model: NN,
                 n_actions: int,
                 horizon: int,
                 n_agents: int,
                 discount: float,
                 gae_lambda: float):
    
    initial_carry = {"agents_stateFeature": agents_stateFeature, 
                     "agents_state": agents_state, 
                     "key": key}
    def scan_function(carry, x=None):
        # (n_agents, n_actions), (n_agents, 1)
        agents_logProbs, agents_value = model.apply(model_params, carry["agents_stateFeature"])
        agents_probs = jnp.exp(agents_logProbs)
        agents_value = jnp.squeeze(agents_value, axis=-1)  # (n_agents,)
        assert agents_probs.shape == (n_agents, n_actions)

        key, *agents_subkeyPolicy = jax.random.split(carry["key"], n_agents+1)
        agents_subkeyPolicy = jnp.asarray(agents_subkeyPolicy)  # (n_agents, ...)
        # (n_agents,.), int, (n_agents,n_actions), (n_agents,n_actions) --> (n_agents,), (n_agents,)
        agents_action, agents_logLikelihood = sample_action_and_logLikelihood(
                                                                agents_subkeyPolicy, 
                                                                n_actions, 
                                                                agents_probs,
                                                                agents_logProbs)
        assert agents_action.shape == (n_agents,)
        assert agents_logLikelihood.shape == (n_agents,)

        append_to = {"states": carry["agents_stateFeature"],
                     "values": agents_value,
                     "actions": agents_action,
                     "old_policy_log_likelihoods": agents_logLikelihood}

        ################### MDP TRANSITION ###################        
        carry["key"], *agents_subkeyMDP = jax.random.split(key, n_agents+1)
        agents_subkeyMDP = jnp.asarray(agents_subkeyMDP)  # (n_agents, ...)
        carry["agents_stateFeature"], carry["agents_state"], append_to["rewards"], append_to["nextTerminals"], _ = vecEnv_step(
                                                                agents_subkeyMDP, 
                                                                carry["agents_state"], 
                                                                agents_action)
        return carry, append_to

    carry, batch = jax.lax.scan(scan_function, initial_carry, xs=None, length=horizon)

    # Finally, also get v(S_horizon) for advantage calculation
    _, agents_value = model.apply(model_params, carry["agents_stateFeature"])
    agents_value = jnp.squeeze(agents_value, axis=-1)  # (n_agents,)
    batch["values"] = jnp.row_stack((batch["values"],
                                     agents_value))  # (horizon + 1, n_agents), where column = [v_0, v_1, v_2, ..., v_H]
    
    # Both: (horizon, n_agents)
    batch["advantages"], batch["bootstrap_returns"] = batch_advantages_and_returns(
                                                                batch["values"],
                                                                batch["rewards"],
                                                                batch["nextTerminals"],
                                                                horizon,
                                                                discount,
                                                                gae_lambda)
    assert batch["advantages"].shape == batch["bootstrap_returns"].shape == (horizon, n_agents)

    return_keys = ["states", "actions", "old_policy_log_likelihoods", "advantages", "bootstrap_returns"]
    return_batch = {k: batch[k] for k in return_keys}

    return carry["agents_stateFeature"], carry["agents_state"], return_batch, carry["key"]


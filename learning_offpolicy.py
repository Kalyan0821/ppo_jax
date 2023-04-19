import jax
import optax
import jax.numpy as jnp
from functools import partial
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
from model import NN, PerturbedModel
from typing import Callable
from jax.config import config as cfg
cfg.update("jax_enable_x64", True)  # to ensure vmap/non-vmap consistency


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
    behaviour_log_likelihoods = minibatch["behaviour_log_likelihoods"]  # (minibatch_size,)
    corrected_advantages = minibatch["corrected_advantages"]  # (minibatch_size,)
    corrected_bootstrap_returns = minibatch["corrected_bootstrap_returns"]  # (minibatch_size,)
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
    
    # if normalize_advantages:
    #     advantages = (advantages-jnp.mean(advantages)) / (jnp.std(advantages)+1e-8)
    corrected_advantages = jnp.where(normalize_advantages, 
                           (corrected_advantages-jnp.mean(corrected_advantages)) / (jnp.std(corrected_advantages)+1e-8), 
                           corrected_advantages)

    policy_by_behaviour_likelihood_ratios = jnp.exp(policy_log_likelihoods - behaviour_log_likelihoods)    
    old_by_behaviour_likelihood_ratios = jnp.exp(old_policy_log_likelihoods - behaviour_log_likelihoods)    

    policy_gradient_losses = policy_by_behaviour_likelihood_ratios * corrected_advantages  # (minibatch_size,)
    reg_losses = old_by_behaviour_likelihood_ratios * nn.relu((likelihood_ratios-clip_likelihood_ratios) * corrected_advantages)
    
    ppo_loss = -1. * jnp.mean(policy_gradient_losses - reg_losses)
    val_loss = 0.5 * jnp.mean((values-corrected_bootstrap_returns)**2)    
    entropy_bonus = jnp.mean(-jnp.exp(policy_log_probs)*policy_log_probs) * n_actions

    loss = ppo_loss + val_loss_coeff*val_loss - entropy_coeff*entropy_bonus
    return loss, (ppo_loss, val_loss, entropy_bonus, clip_trigger_frac, approx_kl)


val_and_grad_function = jax.jit(jax.value_and_grad(loss_function, argnums=0, has_aux=True),
                                static_argnums=(2, 3, 4))


@jax.jit
def permute(batch, key):
    """ batch: each jnp.array: (horizon, n_agents, ...) """

    _, key0, key1 = jax.random.split(key, 3)
    batch = jax.tree_map(lambda x: jax.random.permutation(key0, x, axis=0),
                         batch)
    batch = jax.tree_map(lambda x: jax.random.permutation(key1, x, axis=1),
                         batch)
    return batch


@partial(jax.jit, static_argnums=(3, 5, 6, 7, 8, 9))
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
                clip_epsilon: float):
    """ batch: each jnp.array: (horizon, n_agents, ...) """

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


@partial(jax.jit, static_argnums=(4, 5))
def batch_advantages_and_returns(values: jnp.array,
                                 rewards: jnp.array,
                                 next_is_terminal: jnp.array,
                                 old_by_behaviour_likelihood_ratios: jnp.array,
                                 horizon: int,
                                 n_agents: int,
                                 discount: float,
                                 gae_lambda: float):
    
    assert rewards.shape == next_is_terminal.shape == (horizon, n_agents)
    assert values.shape == (horizon + 1, n_agents)

    initial_carry = {"next_advantage": jnp.zeros(n_agents)}
    def scan_function(carry, t):
        next_value = jnp.where(next_is_terminal[t, :], jnp.zeros(n_agents), values[t+1, :])
        next_advantage = jnp.where(next_is_terminal[t, :], jnp.zeros(n_agents), carry["next_advantage"])

        # RHO = 1.0
        RHO = old_by_behaviour_likelihood_ratios[t, :]

        # td_error = RHO*(rewards[t] + discount*next_value) - values[t]
        # advantage = td_error + (discount*gae_lambda)*RHO*next_advantage
        # bootstrap_return = advantage + values[t]
        # corrected_advantage = bootstrap_return/(RHO + 1e-8) - values[t]

        temp = rewards[t, :] + discount*next_value + (discount*gae_lambda)*next_advantage
        bootstrap_return = RHO*temp
        advantage = bootstrap_return - values[t, :]
        corrected_advantage = temp - values[t, :]


        append_to = {"corrected_advantages": corrected_advantage,
                     "bootstrap_returns": bootstrap_return}
        
        carry["next_advantage"] = advantage

        return carry, append_to

    _, result = jax.lax.scan(scan_function, initial_carry, xs=jnp.arange(horizon), reverse=True)

    return result["corrected_advantages"], result["bootstrap_returns"]


@partial(jax.jit, static_argnums=(1,))
@partial(jax.vmap, in_axes=(0, None, 0, 0), out_axes=(0, 0))
def sample_action_and_logLikelihood(key, n_actions, probs, logProbs):
    assert probs.shape == logProbs.shape == (n_actions,)

    action = jax.random.choice(key, n_actions, p=probs)
    logLikelihood = logProbs[action]
    return (action, logLikelihood)


@partial(jax.jit, static_argnums=(2, 5, 7, 8, 9))
def sample_batch(agents_stateFeature: jnp.array,
                 agents_state: jnp.array,
                 vecEnv_step: Callable,
                 key: jax.random.PRNGKey,
                 model_params: FrozenDict,
                 model: NN,
                 behaviour_params: FrozenDict,
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
        agents_value = jnp.squeeze(agents_value, axis=-1)  # (n_agents,)
        assert agents_logProbs.shape == (n_agents, n_actions)
        
        # (n_agents, n_actions)
        behaviours_logProbs, behaviours_value = model.apply(behaviour_params, carry["agents_stateFeature"])
        behaviours_probs = jnp.exp(behaviours_logProbs)
        behaviours_value = jnp.squeeze(behaviours_value, axis=-1)  # (n_agents,)        
        assert behaviours_probs.shape == (n_agents, n_actions)


        key, *behaviours_subkeyPolicy = jax.random.split(carry["key"], n_agents+1)
        behaviours_subkeyPolicy = jnp.asarray(behaviours_subkeyPolicy)  # (n_agents, ...)
        # (n_agents,.), int, (n_agents,n_actions), (n_agents,n_actions) --> (n_agents,), (n_agents,)
        behaviours_action, behaviours_logLikelihood = sample_action_and_logLikelihood(
                                                                behaviours_subkeyPolicy, 
                                                                n_actions, 
                                                                behaviours_probs,
                                                                behaviours_logProbs)
        
        # (m, a) x (m,) => (m,) 
        @partial(jax.vmap, in_axes=0)
        def get_element(vector, idx):
            return vector[idx]
        agents_logLikelihood = get_element(agents_logProbs, behaviours_action)
        
        assert behaviours_action.shape == (n_agents,)
        assert agents_logLikelihood.shape == behaviours_logLikelihood.shape == (n_agents,)

        append_to = {"states": carry["agents_stateFeature"],
                     "values": agents_value,
                     "behaviour_values": behaviours_value,
                     "actions": behaviours_action,
                     "old_policy_log_likelihoods": agents_logLikelihood,
                     "behaviour_log_likelihoods": behaviours_logLikelihood}

        ################### MDP TRANSITION ###################        
        carry["key"], *behaviours_subkeyMDP = jax.random.split(key, n_agents+1)
        behaviours_subkeyMDP = jnp.asarray(behaviours_subkeyMDP)  # (n_agents, ...)
        carry["agents_stateFeature"], carry["agents_state"], append_to["rewards"], append_to["nextTerminals"], _ = vecEnv_step(
                                                                behaviours_subkeyMDP, 
                                                                carry["agents_state"], 
                                                                behaviours_action)
        return carry, append_to

    carry, batch = jax.lax.scan(scan_function, initial_carry, xs=None, length=horizon)

    # Finally, also get v(S_horizon) for advantage calculation
    _, behaviours_value = model.apply(behaviour_params, carry["agents_stateFeature"])
    behaviours_value = jnp.squeeze(behaviours_value, axis=-1)  # (n_agents,)
    batch["behaviour_values"] = jnp.row_stack((batch["behaviour_values"],
                                               behaviours_value))  # (horizon + 1, n_agents), where column = [v_0, v_1, v_2, ..., v_H]
    
    _, agents_value = model.apply(model_params, carry["agents_stateFeature"])
    agents_value = jnp.squeeze(agents_value, axis=-1)  # (n_agents,)
    batch["values"] = jnp.row_stack((batch["values"],
                                     agents_value))  # (horizon + 1, n_agents), where column = [v_0, v_1, v_2, ..., v_H]
    


    old_by_behaviour_likelihood_ratios = jnp.exp(batch["old_policy_log_likelihoods"] - batch["behaviour_log_likelihoods"])
    assert old_by_behaviour_likelihood_ratios.shape == (horizon, n_agents)

    # Both: (horizon, n_agents)
    batch["corrected_advantages"], batch["corrected_bootstrap_returns"] = batch_advantages_and_returns(
                                                                batch["values"],
                                                                batch["rewards"],
                                                                batch["nextTerminals"],
                                                                old_by_behaviour_likelihood_ratios,
                                                                horizon,
                                                                n_agents,
                                                                discount,
                                                                gae_lambda)

    batch["behaviour_advantages"], batch["behaviour_bootstrap_returns"] = batch_advantages_and_returns(
                                                                batch["behaviour_values"],
                                                                batch["rewards"],
                                                                batch["nextTerminals"],
                                                                jnp.ones((horizon, n_agents)),
                                                                horizon,
                                                                n_agents,
                                                                discount,
                                                                gae_lambda)

    assert batch["corrected_advantages"].shape == batch["corrected_bootstrap_returns"].shape == (horizon, n_agents)

    return_keys = ["states", "actions", "old_policy_log_likelihoods", "behaviour_log_likelihoods", "corrected_advantages", "behaviour_advantages", "corrected_bootstrap_returns", "behaviour_bootstrap_returns"]
    return_batch = {k: batch[k] for k in return_keys}

    return carry["agents_stateFeature"], carry["agents_state"], return_batch, carry["key"]


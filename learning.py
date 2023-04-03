import jax
import optax
import jax.numpy as jnp
from functools import partial
from flax.core.frozen_dict import FrozenDict
from model import NN
from typing import Callable


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
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


@jax.jit
def permute(batch, key):
    """ batch: each jnp.array: (horizon, n_agents, ...) """

    key0, key1 = jax.random.split(key)

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
                permute_batch: bool,
                ):
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

    val_and_grad_function = jax.value_and_grad(loss_function, argnums=0, has_aux=True)

    minibatch_losses = []
    ppo_losses = []
    val_losses = []
    ent_bonuses = []
    clip_trigger_fracs = []
    approx_kls = []
    for minibatch_idx in range(n_iters):
        minibatch = jax.tree_map(lambda x: x[minibatch_idx],
                                 reshaped_batch)
        assert minibatch["states"].shape[0] == minibatch_size

        (minibatch_loss, loss_info), gradient = val_and_grad_function(model_params,
                                                         minibatch,
                                                         model,
                                                         n_actions,
                                                         minibatch_size,
                                                         val_loss_coeff,
                                                         entropy_coeff,
                                                         normalize_advantages,
                                                         clip_epsilon)
        
        param_updates, optimizer_state = optimizer.update(gradient,
                                                          optimizer_state,
                                                          model_params)
        model_params = optax.apply_updates(model_params, param_updates)

        minibatch_losses.append(minibatch_loss)
        ppo_loss, val_loss, entropy_bonus, clip_trigger_frac, approx_kl = loss_info
        ppo_losses.append(ppo_loss)
        val_losses.append(val_loss)
        ent_bonuses.append(entropy_bonus)
        clip_trigger_fracs.append(clip_trigger_frac)
        approx_kls.append(approx_kl)

    return (model_params, optimizer_state, jnp.asarray(minibatch_losses), 
            jnp.asarray(ppo_losses), jnp.asarray(val_losses), jnp.asarray(ent_bonuses), jnp.asarray(clip_trigger_fracs), jnp.asarray(approx_kls))


@partial(jax.jit, static_argnums=(1,))
@partial(jax.vmap, in_axes=(0, None, 0, 0), out_axes=(0, 0))
def sample_action_and_logLikelihood(key, n_actions, probs, logProbs):
    assert probs.shape == logProbs.shape == (n_actions,)

    action = jax.random.choice(key, n_actions, p=probs)
    logLikelihood = logProbs[action]
    return (action, logLikelihood)


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

    advantages = []
    bootstrap_returns = []
    next_advantage = 0.
    for t in reversed(range(horizon)):  # horizon-1, ..., 2, 1, 0
        # if next_is_terminal[t]:
        #     next_value = 0.
        #     next_advantage = 0.
        # else:
        #     next_value = values[t+1]
        next_value = jnp.where(next_is_terminal[t], 0., values[t+1])
        next_advantage = jnp.where(next_is_terminal[t], 0., next_advantage)

        td_error = rewards[t] + discount*next_value - values[t]
        advantage = td_error + (discount*gae_lambda)*next_advantage
        bootstrap_return = advantage + values[t]

        advantages.append(advantage)
        bootstrap_returns.append(bootstrap_return)
        next_advantage = advantage

    advantages = jnp.asarray(advantages[::-1])
    bootstrap_returns = jnp.asarray(bootstrap_returns[::-1])

    return advantages, bootstrap_returns


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
    
    agents_stateFeature_list = []  # (horizon, n_agents, n_features)
    agents_value_list = []   # (horizon + 1, n_agents), where column = [v_0, v_1, v_2, ..., v_H]
    agents_action_list = []  # (horizon, n_agents)
    agents_logLikelihood_list = []  # (horizon, n_agents)
    agents_reward_list = []  # (horizon, n_agents), where column = [R_1, R_2, ..., R_H]
    agents_nextTerminal_list = []  # (horizon, n_agents)

    for _ in range(horizon):
        # (n_agents, n_actions), (n_agents, 1)
        agents_logProbs, agents_value = model.apply(model_params, agents_stateFeature)
        agents_probs = jnp.exp(agents_logProbs)
        agents_value = jnp.squeeze(agents_value, axis=-1)  # (n_agents,)
        assert agents_probs.shape == (n_agents, n_actions)

        key, *agents_subkeyPolicy = jax.random.split(key, n_agents+1)
        agents_subkeyPolicy = jnp.asarray(agents_subkeyPolicy)  # (n_agents, ...)
        # (n_agents,.), int, (n_agents,n_actions), (n_agents,n_actions) --> (n_agents,), (n_agents,)
        agents_action, agents_logLikelihood = sample_action_and_logLikelihood(
                                                                agents_subkeyPolicy, 
                                                                n_actions, 
                                                                agents_probs,
                                                                agents_logProbs)
        assert agents_action.shape == (n_agents,)
        assert agents_logLikelihood.shape == (n_agents,)

        agents_stateFeature_list.append(agents_stateFeature)
        agents_value_list.append(agents_value)
        agents_action_list.append(agents_action)
        agents_logLikelihood_list.append(agents_logLikelihood)

        ################### MDP TRANSITION ###################        
        key, *agents_subkeyMDP = jax.random.split(key, n_agents+1)
        agents_subkeyMDP = jnp.asarray(agents_subkeyMDP)  # (n_agents, ...)
        agents_stateFeature, agents_state, agents_reward, agents_nextTerminal, _ = vecEnv_step(
                                                                agents_subkeyMDP, 
                                                                agents_state, 
                                                                agents_action)        
        agents_reward_list.append(agents_reward)
        agents_nextTerminal_list.append(agents_nextTerminal)

    # Finally, also get v(S_horizon) for advantage calculation
    _, agents_value = model.apply(model_params, agents_stateFeature)
    agents_value = jnp.squeeze(agents_value, axis=-1)  # (n_agents,)
    agents_value_list.append(agents_value)

    agents_value_array = jnp.asarray(agents_value_list)  # (horizon + 1, n_agents), where column = [v_0, v_1, v_2, ..., v_H]
    agents_reward_array = jnp.asarray(agents_reward_list)  # (horizon, n_agents), where column = [R_1, R_2, ..., R_H]
    agents_nextTerminal_array = jnp.asarray(agents_nextTerminal_list)  # (horizon, n_agents)

    # Both: (horizon, n_agents)
    agents_advantage_array, agents_bootstrapReturn_array = batch_advantages_and_returns(
                                                                agents_value_array,
                                                                agents_reward_array,
                                                                agents_nextTerminal_array,
                                                                horizon,
                                                                discount,
                                                                gae_lambda)
    assert agents_advantage_array.shape == agents_bootstrapReturn_array.shape == (horizon, n_agents)

    batch = dict()
    batch["states"] = jnp.asarray(agents_stateFeature_list)  # (horizon, n_agents, n_features)
    batch["actions"] = jnp.asarray(agents_action_list)  # (horizon, n_agents)
    batch["old_policy_log_likelihoods"] = jnp.asarray(agents_logLikelihood_list)  # (horizon, n_agents)
    batch["advantages"] = agents_advantage_array  # (horizon, n_agents)
    batch["bootstrap_returns"] = agents_bootstrapReturn_array  # (horizon, n_agents)

    return agents_stateFeature, agents_state, batch, key


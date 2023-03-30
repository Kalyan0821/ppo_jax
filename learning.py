import jax
import optax
import jax.numpy as jnp
from functools import partial
from gymnax.environments.environment import Environment
from flax.core.frozen_dict import FrozenDict
from model import NN


@partial(jax.jit, static_argnums=(1, 4, 5, 6, 7, 8))
def loss_function(model_params: FrozenDict, 
                  model: NN,
                  minibatch: dict[str, jnp.array],
                  clip_epsilon: float,
                  val_loss_coeff: float,
                  entropy_coeff: float,
                  minibatch_size: int,
                  n_actions: int,
                  normalize_advantages: bool):
        
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

    likelihood_ratios = jnp.exp(policy_log_likelihoods-old_policy_log_likelihoods)
    clip_likelihood_ratios = jnp.clip(likelihood_ratios, 
                                         a_min=1-clip_epsilon, a_max=1+clip_epsilon)
    
    if normalize_advantages:
        advantages = (advantages-jnp.mean(advantages)) / (jnp.std(advantages)+1e-8)

    policy_gradient_losses = likelihood_ratios * advantages  # (minibatch_size,)
    clip_losses = clip_likelihood_ratios * advantages

    ppo_loss = -1.*jnp.mean(jnp.minimum(policy_gradient_losses, clip_losses))
    val_loss = jnp.mean((values-bootstrap_returns)**2)    
    entropy_bonus = jnp.mean(-jnp.exp(policy_log_probs)*policy_log_probs) * n_actions

    loss = ppo_loss + val_loss_coeff*val_loss - entropy_coeff*entropy_bonus
    return loss


@jax.jit
def permute(batch, key):
    """ batch: each jnp.array: (n_agents, horizon, ...) """

    _, key0, key1 = jax.random.split(key, 3)

    batch = jax.tree_map(lambda x: jax.random.permutation(key0, x, axis=0),
                         batch)
    batch = jax.tree_map(lambda x: jax.random.permutation(key1, x, axis=1),
                         batch)
    return batch


@partial(jax.jit, static_argnums=(1, 2, 7, 8, 9, 10, 11, 12, 13, 14))
def batch_epoch(model_params: FrozenDict, 
                model: NN,
                optimizer: optax.GradientTransformation,
                optimizer_state: optax.OptState,
                batch: dict[str, jnp.array],
                permutation_key: jax.random.PRNGKey,
                clip_epsilon: float,
                val_loss_coeff: float,
                entropy_coeff: float,
                n_agents: int,
                horizon: int,
                minibatch_size: int,
                n_actions: int,
                permute_batch=True,
                normalize_advantages=False):
    """ batch: each jnp.array: (n_agents, horizon, ...) """

    assert batch["states"].shape[:-1] == (n_agents, horizon)

    if permute_batch:
        batch = permute(batch, permutation_key)
        assert batch["states"].shape[:-1] == (n_agents, horizon)

    n_iters = n_agents*horizon // minibatch_size  # number of minibatches
    assert n_iters*minibatch_size == n_agents*horizon

    def reshape(x):
        new_shape = (n_iters, minibatch_size) + x.shape[2:]
        return jnp.reshape(x, new_shape)
    reshaped_batch = jax.tree_map(reshape, batch)
    assert reshaped_batch["states"].shape[0] == n_iters

    val_and_grad_function = jax.value_and_grad(loss_function, argnums=0)

    minibatch_losses = []
    for minibatch_idx in range(n_iters):
        minibatch = jax.tree_map(lambda x: x[minibatch_idx],
                                 reshaped_batch)
        assert minibatch["states"].shape[0] == minibatch_size

        minibatch_loss, gradient = val_and_grad_function(model_params,
                                                         model,
                                                         minibatch,
                                                         clip_epsilon,
                                                         val_loss_coeff,
                                                         entropy_coeff,
                                                         minibatch_size,
                                                         n_actions,
                                                         normalize_advantages)

        param_updates, optimizer_state = optimizer.update(gradient,
                                                          optimizer_state,
                                                          model_params)
        model_params = optax.apply_updates(model_params, param_updates)


        minibatch_losses.append(minibatch_loss)

    return model_params, optimizer_state, minibatch_losses


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
@partial(jax.vmap, in_axes=(None, 0, None, None, None, None, None))  # run on several agents in parallel
def agent_trajectory(env: Environment,
                     key: jax.random.PRNGKey,
                     model_params: FrozenDict, 
                     model: NN,
                     horizon: int,
                     discount: float,
                     gae_lambda: float):

    key, subkey_reset = jax.random.split(key)
    state_array, state = env.reset(subkey_reset)  # state_array: (n_features,)

    state_arrays = []  # (horizon, n_features)
    actions = []  # (horizon,)
    policy_log_likelihoods = []  # (horizon,)


    rewards = []  # [R_1, R_2, ..., R_H]
    values = []   # [v_0, v_1, v_2, ..., v_H]
    currently_terminal = False
    for t in range(horizon):
        
        key, subkey_policy, subkey_mdp = jax.random.split(key, 3)

        # (n_actions), (1,)
        policy_log_probs, value = model.apply(model_params, state_array)
        value = value[0]
        policy_probs = jnp.exp(policy_log_probs)
        assert jnp.size(policy_probs) == env.action_space().n

        action = jax.random.choice(subkey_policy, 
                                   env.action_space().n, 
                                   p=policy_probs)
        policy_log_likelihood = policy_log_probs[action]

        state_arrays.append(state_array)
        actions.append(action)
        policy_log_likelihoods.append(policy_log_likelihood)
        
        state_array, state, reward, next_is_terminal, _ = env.step(subkey_mdp, state, action)
        
        # if currently_terminal:
        #     reward = 0  # R_t+1 = 0
        #     value = 0  # v(S_t) = 0
        reward = jnp.where(currently_terminal, 0, reward)
        value = jnp.where(currently_terminal, 0, value)

        rewards.append(reward)
        values.append(value)

        # if next_is_terminal:
        #     currently_terminal = True
        currently_terminal = jnp.where(next_is_terminal, True, currently_terminal)

    # # Also find v(S_horizon)
    # if currently_terminal:
    #     final_value = 0
    # else:
    #     _, final_value = model.apply(model_params, state_array)
    #     final_value = final_value[0]
    final_value = jnp.where(currently_terminal, 
                            0,
                            model.apply(model_params, state_array)[1][0])

    values.append(final_value)

    assert len(rewards) == horizon
    assert len(values) == horizon + 1

    advantages = []  # (horizon,)
    bootstrap_returns = []  # (horizon,)

    next_advantage = 0.
    for t in reversed(range(horizon)):  # horizon-1, ..., 2, 1, 0
        td_error = rewards[t] + discount*values[t+1] - values[t]
        advantage = td_error + (discount*gae_lambda)*next_advantage
        bootstrap_return = advantage + values[t]
        advantages.append(advantage)
        bootstrap_returns.append(bootstrap_return)

        next_advantage = advantage

    advantages = advantages[::-1]
    bootstrap_returns = bootstrap_returns[::-1]

    batch = dict()
    batch["states"] = jnp.asarray(state_arrays)  # (horizon, n_features)
    batch["actions"] = jnp.asarray(actions)  # (horizon,)
    batch["old_policy_log_likelihoods"] = jnp.asarray(policy_log_likelihoods)  # (horizon,)
    batch["advantages"] = jnp.asarray(advantages)  # (horizon,)
    batch["bootstrap_returns"] = jnp.asarray(bootstrap_returns)  # (horizon,)

    return batch


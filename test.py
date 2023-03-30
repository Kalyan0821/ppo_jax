import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments.environment import Environment
from flax.core.frozen_dict import FrozenDict
from model import NN


@partial(jax.jit, static_argnums=(0, 3, 4))
@partial(jax.vmap, in_axes=(None, 0, None, None, None))  # run on several agents in parallel
def full_return(env: Environment,
                key: jax.random.PRNGKey,
                model_params: FrozenDict, 
                model: NN,
                discount: float):

    key, subkey_reset = jax.random.split(key)
    state_feature, state = env.reset(subkey_reset)  # state_feature: (n_features,)

    # next_is_terminal = False
    # t = 0
    # discounted_return = 0

    # while not next_is_terminal:
    #     key, subkey_policy, subkey_mdp = jax.random.split(key, 3)

    #     # (n_actions), (1,)
    #     policy_log_probs, _ = model.apply(model_params, state_feature)
    #     policy_probs = jnp.exp(policy_log_probs)
    #     action = jax.random.choice(subkey_policy, 
    #                                env.action_space().n, 
    #                                p=policy_probs)

    #     state_feature, state, reward, next_is_terminal, _ = env.step(subkey_mdp, state, action)
    #     discounted_return += (discount**t) * reward
        
    #     t += 1
    initial_val = {"next_is_terminal": False,
                   't': 0,
                   "discounted_return": 0,
                   "key": key,
                   "model_params": model_params,
                   "state_feature": state_feature,
                   "state": state}

    def condition_function(val):
        return jnp.logical_not(val["next_is_terminal"])

    def body_function(val):
        val["key"], subkey_policy, subkey_mdp = jax.random.split(val["key"], 3)

        # (n_actions), (1,)
        policy_log_probs, _ = model.apply(val["model_params"], val["state_feature"])
        policy_probs = jnp.exp(policy_log_probs)
        action = jax.random.choice(subkey_policy, 
                                    env.action_space().n, 
                                    p=policy_probs)

        val["state_feature"], val["state"], reward, val["next_is_terminal"], _ = env.step(subkey_mdp, val["state"], action)
        val["discounted_return"] += (discount**val['t']) * reward
        val['t'] += 1
        
        return val

    val = jax.lax.while_loop(condition_function, body_function, initial_val)
    return val["discounted_return"]






import jax
import jax.numpy as jnp
from functools import partial
from gymnax.environments.environment import Environment
from flax.core.frozen_dict import FrozenDict
from flax.training.checkpoints import restore_checkpoint
import numpy as np
import flax.linen as nn
from model import NN, SeparateNN
from jax.config import config as cfg
cfg.update("jax_enable_x64", True)  # to ensure vmap/non-vmap consistency


@partial(jax.jit, static_argnums=(0, 3, 4))
@partial(jax.vmap, in_axes=(None, 0, None, None, None, None))  # run on several agents in parallel
def full_return(env: Environment,
                key: jax.random.PRNGKey,
                model_params: FrozenDict, 
                model: NN,
                n_actions: int,
                discount: float):

    key, subkey_reset = jax.random.split(key)
    state_feature, state = env.reset(subkey_reset)  # state_feature: (n_features,)

    initial_val = {"next_is_terminal": False,
                   't': 0,
                   "discounted_return": 0,
                   "key": key,
                   "state_feature": state_feature,
                   "state": state}

    def condition_function(val):
        return jnp.logical_not(val["next_is_terminal"])

    def body_function(val):
        val["key"], subkey_policy, subkey_mdp = jax.random.split(val["key"], 3)

        # (n_actions), (1,)
        policy_log_probs, _ = model.apply(model_params, val["state_feature"])
        policy_probs = jnp.exp(policy_log_probs)
        assert policy_probs.shape == (n_actions,), f"{policy_probs.shape}, {n_actions}"
        action = jax.random.choice(subkey_policy, n_actions, p=policy_probs)
        
        val["state_feature"], val["state"], reward, val["next_is_terminal"], _ = env.step(subkey_mdp, val["state"], action)
        val["discounted_return"] += (discount**val['t']) * reward
        val['t'] += 1
        return val

    val = jax.lax.while_loop(condition_function, body_function, initial_val)
    return val["discounted_return"], val['t']


@partial(jax.jit, static_argnums=(0, 3, 4, 5, 7))
def evaluate(env: Environment,
             key: jax.random.PRNGKey,
             model_params: FrozenDict, 
             model: NN,
             n_actions: int,
             n_eval_agents: int,
             discount: float,
             return_durations=False):
    
    agents_subkeyEval = jax.random.split(key, n_eval_agents)
    returns, durations = full_return(env,
                                     agents_subkeyEval,
                                     model_params,
                                     model,
                                     n_actions,
                                     discount)
    assert returns.shape == durations.shape == (n_eval_agents,)
    if return_durations: 
        return durations
    
    return returns


class TestablePerturbedModel(nn.Module):
    model: NN
    alpha: float
    
    def apply(self, params: FrozenDict, x: jnp.array):
        assert 0 <= self.alpha <= 1

        log_probs, _ = self.model.apply(params, x)
        probs = jnp.exp(log_probs)
        assert probs.shape[-1] == self.model.n_actions

        uniform_probs = jnp.ones(probs.shape) / self.model.n_actions

        new_probs = self.alpha*probs + (1-self.alpha)*uniform_probs
        new_log_probs = jnp.log(new_probs)

        return new_log_probs, None
    

if __name__ == "__main__":
    from train_parallel_offpolicy import env_name, SEED, N_SEEDS, hidden_layer_sizes, architecture, activation, n_eval_agents, env, example_state_feature, n_actions, eval_discount, offpolicy_alpha
    key0 = jax.random.PRNGKey(SEED)
    keys = jnp.array([key0, *jax.random.split(key0, N_SEEDS-1)])

    if architecture == "shared":
        model = NN(hidden_layer_sizes=hidden_layer_sizes, 
                n_actions=n_actions, 
                single_input_shape=example_state_feature.shape,
                activation=activation)
    elif architecture == "separate":
        model = SeparateNN(hidden_layer_sizes=hidden_layer_sizes, 
                        n_actions=n_actions, 
                        single_input_shape=example_state_feature.shape,
                        activation=activation)

    model_params = restore_checkpoint("./saved_models", None, 0, prefix=env_name+'_')
    perturbed_model = TestablePerturbedModel(model, alpha=offpolicy_alpha)
    print(perturbed_model, '\n')

    vmap_evaluate = jax.vmap(evaluate, in_axes=(None, 0, None, None, None, None, None, None))
    returns = vmap_evaluate(env, keys, model_params, perturbed_model, n_actions, n_eval_agents, eval_discount, 
                            False)
    
    avg_return, std_return = np.mean(returns), np.std(returns)
    print(f"Returns avg ± std: {avg_return} ± {std_return}")
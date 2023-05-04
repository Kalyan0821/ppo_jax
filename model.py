import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.core.frozen_dict import FrozenDict
from jax.config import config as cfg
cfg.update("jax_enable_x64", True)  # to ensure vmap/non-vmap consistency


class NN(nn.Module):
    """ Shared-param model for policy and value function """

    hidden_layer_sizes: tuple[int]
    n_actions: int
    single_input_shape: tuple[int]
    activation: str
    return_feature: bool = False
    freeze_representation: bool = False

    @nn.compact
    def __call__(self, x: jnp.array):
        """ x: (n_features,) """

        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else: 
            raise NotImplementedError

        # flatten input features if required
        if x.shape == self.single_input_shape:
            x = jnp.ravel(x)
        elif x.shape[1:] == self.single_input_shape:
            x = jnp.reshape(x, (x.shape[0], -1))
        else: 
            raise NotImplementedError

        # Shared layers
        for l, size in enumerate(self.hidden_layer_sizes):
            x = nn.Dense(features=size, name=f"dense_{l+1}")(x)
            x = activation(x)

        if self.freeze_representation:
            x = jax.lax.stop_gradient(x)

        # Output layers
        policy_logits = nn.Dense(features=self.n_actions, name="logits")(x)
        policy_log_probs = nn.log_softmax(policy_logits)

        value = nn.Dense(features=1, name="value")(x)

        if self.return_feature:
            return policy_log_probs, value, jax.lax.stop_gradient(x)  # (n_actions,), (1,), (final_hidden_layer_size,)

        return policy_log_probs, value  # (n_actions,), (1,)
    


class SeparateNN(nn.Module):
    """ Separate models for policy and value function """

    hidden_layer_sizes: tuple[int]
    n_actions: int
    single_input_shape: tuple[int]
    activation: str
    return_feature: bool = False

    @nn.compact
    def __call__(self, x: jnp.array):
        """ x: (n_features,) """

        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else: 
            raise NotImplementedError
        
        # Flatten input features if required
        if x.shape == self.single_input_shape:
            x = jnp.ravel(x)
        elif x.shape[1:] == self.single_input_shape:
            x = jnp.reshape(x, (x.shape[0], -1))
        else: 
            raise NotImplementedError

        actor_x = x
        critic_x = x

        # Actor
        for l, size in enumerate(self.hidden_layer_sizes):
            actor_x = nn.Dense(features=size, name=f"dense_{l+1}_policy")(actor_x)
            actor_x = activation(actor_x)

        policy_logits = nn.Dense(features=self.n_actions, name="logits")(actor_x)
        policy_log_probs = nn.log_softmax(policy_logits)

        # Critic
        for l, size in enumerate(self.hidden_layer_sizes):
            critic_x = nn.Dense(features=size, name=f"dense_{l+1}_value")(critic_x)
            critic_x = activation(critic_x)

        value = nn.Dense(features=1, name="value")(critic_x)

        # Return
        if self.return_feature:
            return policy_log_probs, value, jax.lax.stop_gradient(actor_x)  # (n_actions,), (1,), (final_hidden_layer_size,)
        
        return policy_log_probs, value  # (n_actions,), (1,)
    

class PerturbedModel(nn.Module):
    model: NN
    
    def apply(self, params: FrozenDict, x: jnp.array, alpha: float):
        
        log_probs, _ = self.model.apply(params, x)
        probs = jnp.exp(log_probs)
        assert probs.shape[-1] == self.model.n_actions

        uniform_probs = jnp.ones(probs.shape) / self.model.n_actions

        new_probs = alpha*probs + (1-alpha)*uniform_probs
        new_log_probs = jnp.log(new_probs)

        return new_log_probs, None
    

class SoftMaxLayer(nn.Module):
    n_actions: int
    
    @nn.compact
    def __call__(self, x: jnp.array):
        logits = nn.Dense(features=self.n_actions, name='z')(x)
        log_probs = nn.log_softmax(logits)
        return log_probs
import flax.linen as nn
import jax.numpy as jnp

class NN(nn.Module):
    """ Shared-param model for policy and value function """

    hidden_layer_sizes: tuple[int]
    n_actions: int

    @nn.compact
    def __call__(self, x: jnp.array):
        """ x: (n_features,) """

        # Shared layers
        for l, size in enumerate(self.hidden_layer_sizes):
            x = nn.Dense(features=size, name=f"dense_{l+1}")(x)
            x = nn.relu(x)

        # Output layers
        policy_logits = nn.Dense(features=self.n_actions, name="logits")(x)
        policy_log_probs = nn.log_softmax(policy_logits)
        
        value = nn.Dense(features=1, name="value")(x)

        return policy_log_probs, value  # (n_actions,), (1,)
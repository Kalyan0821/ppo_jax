import flax.linen as nn
import jax.numpy as jnp

class NN(nn.Module):
    """ Shared-param model for policy and value function """

    hidden_layer_sizes: tuple[int]
    n_actions: int
    single_input_shape: tuple[int]

    @nn.compact
    def __call__(self, x: jnp.array):
        """ x: (n_features,) """

        # flatten input features if required
        if x.shape == self.single_input_shape:
            x = jnp.ravel(x)
        elif x.shape[1:] == self.single_input_shape:
            x = jnp.reshape(x, (x.shape[0], -1))
        else: 
            raise NotImplementedError


        # Shared layers
        for l, size in enumerate(self.hidden_layer_sizes):
            x = nn.Dense(features=size, 
                         kernel_init=nn.initializers.orthogonal(scale=jnp.sqrt(2)),
                         name=f"dense_{l+1}")(x)
            
            x = nn.activation.tanh(x)

        # Output layers
        policy_logits = nn.Dense(features=self.n_actions, 
                                 kernel_init=nn.initializers.orthogonal(scale=jnp.array(0.01)),
                                 name="logits")(x)
        policy_log_probs = nn.log_softmax(policy_logits)
        
        value = nn.Dense(features=1, 
                         kernel_init=nn.initializers.orthogonal(scale=1.),
                         name="value")(x)

        return policy_log_probs, value  # (n_actions,), (1,)
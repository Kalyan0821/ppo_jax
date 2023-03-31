import gymnax
import jax
import optax
import jax.numpy as jnp
import numpy as np
from flax.training.checkpoints import save_checkpoint
from model import NN
from learning import batch_epoch, agent_trajectory
from test import full_return


env_name = "CartPole-v1"

SEED = 0
total_experience = 200000
lr = 1e-2
n_agents = 8

horizon = 128
horizon = 8

n_epochs = 16
# minibatch_size = 128
minibatch_size = n_agents*horizon  # for 1 minibatch per epoch
hidden_layer_sizes = (64, 64)
normalize_advantages = True
anneal = True
permute_batches = True
clip_epsilon = 0.2
entropy_coeff = 0.01
val_loss_coeff = 0.5
discount = 0.99
gae_lambda = 0.95

n_eval_agents = 16
eval_discount = 1.0
eval_iter = 20
checkpoint_iter = 20
checkpoint_dir = "./checkpoints"


env, env_params = gymnax.make(env_name)
vecEnv_reset = jax.vmap(env.reset, in_axes=(0, None))
vecEnv_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

key = jax.random.PRNGKey(SEED)
key, subkey_env, subkey_model = jax.random.split(key, 3)
state_feature, _ = env.reset(subkey_env)
n_features = state_feature.size
n_actions = env.action_space().n

model = NN(hidden_layer_sizes=hidden_layer_sizes, n_actions=n_actions)
model_params = model.init(subkey_model, jnp.zeros(n_features))

n_outer_iters = total_experience // (n_agents * horizon)  # loop_steps
n_iters_per_epoch = n_agents*horizon // minibatch_size
n_inner_iters = n_epochs * n_iters_per_epoch

if anneal:
    lr = optax.linear_schedule(init_value=lr, 
                               end_value=0, 
                               transition_steps=n_outer_iters*n_inner_iters)
    
optimizer = optax.adam(lr)
optimizer_state = optimizer.init(model_params)
evals = dict()
print("Outer steps:", n_outer_iters)
for outer_iter in range(n_outer_iters):

    # if outer_iter % eval_iter == 0:
    #     experience = outer_iter * n_agents * horizon
    #     print("Evaluating ...")
    #     key, *eval_agent_keys = jax.random.split(key, n_eval_agents+1)
    #     eval_agent_keys = jnp.asarray(eval_agent_keys)
    #     returns = full_return(env,
    #                           eval_agent_keys,
    #                           model_params,
    #                           model,
    #                           eval_discount)
    #     evals[experience] = returns
    #     print(f"Experience: {experience}. Returns: {returns}\n")


    print(f"Step {outer_iter+1}:")
    key, *agent_keys = jax.random.split(key, n_agents+1)
    agent_keys = jnp.asarray(agent_keys)

    batch = agent_trajectory(env,
                             agent_keys,
                             model_params, 
                             model,
                             horizon,
                             discount,
                             gae_lambda)  # each: (n_agents, horizon, ...)
    
    alpha = (1-outer_iter/n_outer_iters) if anneal else 1

    for epoch in range(n_epochs):

        model_params, optimizer_state, minibatch_losses = batch_epoch(
                                                    model_params, 
                                                    model,
                                                    optimizer,
                                                    optimizer_state,
                                                    batch,
                                                    permutation_key,
                                                    clip_epsilon * alpha,
                                                    val_loss_coeff,
                                                    entropy_coeff,
                                                    n_agents,
                                                    horizon,
                                                    minibatch_size,
                                                    n_actions,
                                                    permute_batches,
                                                    normalize_advantages)
        print(f"Epoch {epoch+1}: avg. loss = {np.mean(minibatch_losses)}, change = ({minibatch_losses[0]} --> {minibatch_losses[-1]})")


    # if (outer_iter+1) % checkpoint_iter == 0:
    #     experience = (outer_iter+1) * n_agents * horizon
    #     checkpoint = {"model": model, 
    #                   "params": model_params}
    #     print("Saving ...")
    #     save_checkpoint(checkpoint_dir, checkpoint, experience, 
    #                     prefix=env_name+"_ckpt_", 
    #                     keep_every_n_steps=checkpoint_iter, 
    #                     overwrite=True)
    
    print('-------------')


print("Summary:")
for experience in evals:
    avg_return = np.mean(evals[experience])
    std_return = np.std(evals[experience])
    print(f"Experience: {experience}. Returns: avg={avg_return},  std={std_return}")

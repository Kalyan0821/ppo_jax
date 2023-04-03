import gymnax
import jax
import optax
import jax.numpy as jnp
from flax.training.checkpoints import save_checkpoint
from model import NN
from learning import sample_batch, batch_epoch
from test import evaluate


env_name = "CartPole-v1"
# env_name = "SpaceInvaders-MinAtar"
# env_name = "MountainCar-v0"

SEED = 0
total_experience = 200000

lr_begin = 2.5e-4
lr_end = 0

n_agents = 16
horizon = 32
n_epochs = 16
minibatch_size = 128
# minibatch_size = n_agents*horizon  # for 1 minibatch per epoch
hidden_layer_sizes = (64, 64)
normalize_advantages = True
anneal = True

permute_batches = True
clip_epsilon = 0.2
entropy_coeff = 0.01
# entropy_coeff = 0.003
val_loss_coeff = 0.5
clip_grad = 0.5

discount = 0.99
gae_lambda = 0.95
n_eval_agents = 32
eval_discount = 1.0
eval_iter = 40
checkpoint_iter = 40
checkpoint_dir = "./checkpoints"

assert minibatch_size <= n_agents*horizon

env, env_params = gymnax.make(env_name)
key = jax.random.PRNGKey(SEED)
key, subkey_env, subkey_model = jax.random.split(key, 3)
state_feature, _ = env.reset(subkey_env)
n_actions = env.action_space().n

model = NN(hidden_layer_sizes=hidden_layer_sizes, 
           n_actions=n_actions, 
           single_input_shape=state_feature.shape)
model_params = model.init(subkey_model, jnp.zeros(state_feature.shape))

n_outer_iters = total_experience // (n_agents * horizon)
n_iters_per_epoch = n_agents*horizon // minibatch_size  # num_minibatches
n_inner_iters = n_epochs * n_iters_per_epoch 

print("\nState feature shape:", state_feature.shape)
print("Action space:", n_actions)
print("Minibatches per epoch:", n_iters_per_epoch)
print("Outer steps:", n_outer_iters, '\n')

if anneal:
    lr = optax.linear_schedule(init_value=lr_begin, 
                               end_value=lr_end, 
                               transition_steps=n_outer_iters*n_inner_iters)
else:
    lr = lr_begin

if clip_grad:
    optimizer = optax.chain(optax.clip_by_global_norm(max_norm=clip_grad), 
                            optax.adam(lr, eps=1e-5))
else:
    optimizer = optax.adam(lr, eps=1e-5)


vecEnv_reset = jax.vmap(env.reset, in_axes=(0,))
vecEnv_step = jax.vmap(env.step, in_axes=(0, 0, 0))

key, *agents_subkeyReset = jax.random.split(key, n_agents+1)
agents_subkeyReset = jnp.asarray(agents_subkeyReset)
agents_stateFeature, agents_state = vecEnv_reset(agents_subkeyReset)  # (n_agents, n_features), (n_agents, .)
optimizer_state = optimizer.init(model_params)

evals = dict()
for outer_iter in range(n_outer_iters):
    experience = outer_iter * n_agents * horizon
    
    if outer_iter % eval_iter == 0:
        print(f"Evaluating at experience {experience}")
        key, key_eval = jax.random.split(key)
        evals[experience] = evaluate(env, key_eval, model_params, model, n_actions, n_eval_agents, eval_discount)
        print("Returns:", evals[experience])
        print('-------------')

    print(f"Step {outer_iter+1}:")
    agents_stateFeature, agents_state, batch, key = sample_batch(agents_stateFeature,
                                                                    agents_state,
                                                                    vecEnv_step,
                                                                    key,
                                                                    model_params, 
                                                                    model,
                                                                    n_actions,
                                                                    horizon,
                                                                    n_agents,
                                                                    discount,
                                                                    gae_lambda)  
    # (agents_stateFeature, agents_state) returned were the last ones seen in this step, and will initialize the next step
    assert agents_stateFeature.shape[0] == n_agents

    alpha = (1-outer_iter/n_outer_iters) if anneal else 1.

    for epoch in range(n_epochs):
        key, permutation_key = jax.random.split(key)

        (model_params, optimizer_state, minibatch_losses, 
         ppo_losses, val_losses, ent_bonuses, clip_trigger_fracs, approx_kls) = batch_epoch(
                                                    batch,
                                                    permutation_key,
                                                    model_params, 
                                                    model,
                                                    optimizer_state,
                                                    optimizer,
                                                    n_actions,
                                                    horizon,
                                                    n_agents,
                                                    minibatch_size,
                                                    val_loss_coeff,
                                                    entropy_coeff,
                                                    normalize_advantages,
                                                    clip_epsilon*alpha,
                                                    permute_batches)
        
        print(f"Epoch {epoch+1}: Loss = {jnp.mean(minibatch_losses):.2f}")
        print(f"ppo = {jnp.mean(ppo_losses):.5f}, val = {jnp.mean(val_losses):.2f}, ent = {jnp.mean(ent_bonuses):.2f}, % clip_trigger = {100*jnp.mean(clip_trigger_fracs):.2f}, approx_kl = {jnp.mean(approx_kls):.5f}")

        # print(f"Epoch {epoch+1}: Loss = {jnp.mean(minibatch_losses)}")
        # print(f"ppo = {jnp.mean(ppo_losses)}, val = {jnp.mean(val_losses)}, ent = {jnp.mean(ent_bonuses)}, % clip_trigger = {100*jnp.mean(clip_trigger_fracs)}, approx_kl = {jnp.mean(approx_kls)}")


    print('-------------')

print(f"Evaluating at experience {experience}")
key, key_eval = jax.random.split(key)
evals[experience] = evaluate(env, key_eval, model_params, model, n_actions, n_eval_agents, eval_discount)
print("Returns:", evals[experience])
print('-------------')

print("Summary:")
for experience in evals:
    avg_return = jnp.mean(evals[experience])
    std_return = jnp.std(evals[experience])
    print(f"Experience: {experience}. Avg. return = {avg_return}, std={std_return}")



# if (outer_iter+1) % checkpoint_iter == 0:
#     experience = (outer_iter+1) * n_agents * horizon
#     checkpoint = {"model": model, 
#                   "params": model_params}
#     print("Saving ...")
#     save_checkpoint(checkpoint_dir, checkpoint, experience, 
#                     prefix=env_name+"_ckpt_", 
#                     keep_every_n_steps=checkpoint_iter, 
#                     overwrite=True)

# print('-------------')




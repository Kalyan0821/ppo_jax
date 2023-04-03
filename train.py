import gymnax
import jax
import optax
import jax.numpy as jnp
from flax.training.checkpoints import save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from model import NN
from learning import sample_batch, batch_epoch
from test import evaluate


# env_name = "CartPole-v1"
env_name = "SpaceInvaders-MinAtar"
# env_name = "MountainCar-v0"

SEED = 0
total_experience = int(5e6)

lr_begin = 5e-4
lr_end = 5e-4
n_agents = 64
horizon = 128
n_epochs = 4
minibatch_size = 1024
# minibatch_size = n_agents*horizon  # for 1 minibatch per epoch
hidden_layer_sizes = (256, 256)
normalize_advantages = True
anneal = True
permute_batches = True
clip_epsilon = 0.2
entropy_coeff = 0.01
val_loss_coeff = 0.5
clip_grad = 0.5
discount = 0.999
gae_lambda = 0.95
n_eval_agents = 164
eval_discount = 1.0
eval_iter = 40
checkpoint_iter = 40
checkpoint_dir = "./checkpoints"
log_dir = "./logs"

assert minibatch_size <= n_agents*horizon
writer = SummaryWriter(log_dir=log_dir)

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
    # optimizer = optax.adam(lr, eps=1e-5)
    optimizer = optax.adam(lr)

vecEnv_reset = jax.vmap(env.reset, in_axes=(0,))
vecEnv_step = jax.vmap(env.step, in_axes=(0, 0, 0))

key, *agents_subkeyReset = jax.random.split(key, n_agents+1)
agents_subkeyReset = jnp.asarray(agents_subkeyReset)
agents_stateFeature, agents_state = vecEnv_reset(agents_subkeyReset)  # (n_agents, n_features), (n_agents, .)
optimizer_state = optimizer.init(model_params)

evals = dict()
for outer_iter in tqdm(range(n_outer_iters)):
    experience = outer_iter * n_agents * horizon
    steps = outer_iter
    # print(f"Step {steps+1}:")

    if outer_iter % eval_iter == 0:
        key, key_eval = jax.random.split(key)
        returns = evaluate(env, key_eval, model_params, model, n_actions, n_eval_agents, eval_discount)
        avg_return, std_return = np.average(returns), np.nanstd(returns)
        evals[experience, steps] = (avg_return, std_return)
        writer.add_scalars("Returns", {"avg": avg_return,
                                       "avg+std": avg_return + std_return,
                                       "avg-std": avg_return - std_return}, experience)

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
        
        # print(f"Epoch {epoch+1}: Loss = {np.mean(minibatch_losses):.2f}")
        # print(f"ppo = {np.mean(ppo_losses):.5f}, val = {np.mean(val_losses):.2f}, ent = {np.mean(ent_bonuses):.2f}, %clip_trigger = {100*np.mean(clip_trigger_fracs):.2f}, approx_kl = {np.mean(approx_kls):.5f}")
        
    new_experience = experience + (n_agents*horizon)
    writer.add_scalar("Losses/total", np.mean(minibatch_losses), new_experience)
    writer.add_scalar("Losses/ppo", np.mean(ppo_losses), new_experience)
    writer.add_scalar("Losses/val", np.mean(val_losses), new_experience)
    writer.add_scalar("Losses/ent", np.mean(ent_bonuses), new_experience)
    writer.add_scalar("Debug/%clip_trig", 100*np.mean(clip_trigger_fracs), new_experience)
    writer.add_scalar("Debug/approx_kl", np.mean(approx_kls), new_experience)


key, key_eval = jax.random.split(key)
returns = evaluate(env, key_eval, model_params, model, n_actions, n_eval_agents, eval_discount)
avg_return, std_return = np.average(returns), np.nanstd(returns)
evals[new_experience, steps+1] = (avg_return, std_return)
writer.add_scalars("Returns", {"avg": avg_return,
                                "avg+std": avg_return + std_return,
                                "avg-std": avg_return - std_return}, new_experience)

print("\nReturns avg ± std:")
for exp, steps in evals:
    avg_return, std_return = evals[exp, steps]
    print(f"(Exp {exp}, steps {steps}) --> {avg_return:.2f} ± {std_return:.2f}")
print()

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




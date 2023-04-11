import gymnax
import jax
import optax
import jax.numpy as jnp
import numpy as np
from flax.training.checkpoints import save_checkpoint
from tqdm import tqdm
import argparse
import json
import wandb
import datetime
from model import NN
from learning import sample_batch, batch_epoch
from test import evaluate
from jax.config import config
config.update("jax_enable_x64", True)  # to ensure vmap/non-vmap consistency

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='JSON file path')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = json.load(f)
env_name = config["env_name"]
SEED = config["SEED"]
total_experience = int(config["total_experience"])



lr_begin = config["lr_begin"]
lr_end = config["lr_end"]
n_agents = config["n_agents"]
horizon = config["horizon"]
n_epochs = config["n_epochs"]
minibatch_size = config["minibatch_size"]
# minibatch_size = n_agents*horizon  # for 1 minibatch per epoch
hidden_layer_sizes = tuple(config["hidden_layer_sizes"])
normalize_advantages = config["normalize_advantages"]
anneal = config["anneal_epsilon"]
clip_epsilon = config["clip_epsilon"]
entropy_coeff = config["entropy_coeff"]
val_loss_coeff = config["val_loss_coeff"]
clip_grad = config["clip_grad"]
discount = config["discount"]
gae_lambda = config["gae_lambda"]
n_eval_agents = config["n_eval_agents"]
eval_discount = config["eval_discount"]
eval_iter = config["eval_iter"]
assert minibatch_size <= n_agents*horizon
# wandb.init(project="ppo", 
#            config=config,
#            name=env_name+'-'+datetime.datetime.now().strftime("%d.%m-%H:%M"))

env, env_params = gymnax.make(env_name)
vecEnv_reset = jax.vmap(env.reset, in_axes=(0,))
vecEnv_step = jax.vmap(env.step, in_axes=(0, 0, 0))
example_state_feature, _ = env.reset(jax.random.PRNGKey(0))
n_actions = env.action_space().n

model = NN(hidden_layer_sizes=hidden_layer_sizes, 
           n_actions=n_actions, 
           single_input_shape=example_state_feature.shape)
print("\nState feature shape:", example_state_feature.shape)
print("Action space:", n_actions)

key = jax.random.PRNGKey(SEED)
key, subkey_model = jax.random.split(key)
model_params = model.init(subkey_model, jnp.zeros(example_state_feature.shape))

n_outer_iters = total_experience // (n_agents * horizon)
n_iters_per_epoch = n_agents*horizon // minibatch_size  # num_minibatches
n_inner_iters = n_epochs * n_iters_per_epoch 

print("Minibatches per epoch:", n_iters_per_epoch)
print("Outer steps:", n_outer_iters, '\n')

lr = optax.linear_schedule(init_value=lr_begin, 
                           end_value=lr_end, 
                           transition_steps=n_outer_iters*n_inner_iters)
if clip_grad:
    optimizer = optax.chain(optax.clip_by_global_norm(max_norm=clip_grad), 
                            optax.adam(lr, eps=1e-5))
else:
    optimizer = optax.adam(lr, eps=1e-5)

key, *agents_subkeyReset = jax.random.split(key, n_agents+1)
agents_subkeyReset = jnp.asarray(agents_subkeyReset)
agents_stateFeature, agents_state = vecEnv_reset(agents_subkeyReset)  # (n_agents, n_features), (n_agents, .)
optimizer_state = optimizer.init(model_params)

evals = dict()
for outer_iter in tqdm(range(n_outer_iters)):
    experience = outer_iter * n_agents * horizon
    steps = outer_iter
    
    if outer_iter % eval_iter == 0:
        key, key_eval = jax.random.split(key)
        returns = evaluate(env, key_eval, model_params, model, n_actions, n_eval_agents, eval_discount)
        avg_return, std_return = jnp.mean(returns), jnp.std(returns)
        evals[experience, steps] = (avg_return, std_return)
        # wandb.log({"Returns/avg": avg_return,
        #            "Returns/avg+std": avg_return + std_return,
        #            "Returns/avg-std": avg_return - std_return}, experience)
        
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

    for _ in range(n_epochs):
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
                                                    clip_epsilon*alpha)
                
    new_experience = experience + (n_agents*horizon)
    # wandb.log({"Losses/total": np.mean(minibatch_losses)}, new_experience)
    # wandb.log({"Losses/ppo": np.mean(ppo_losses)}, new_experience)
    # wandb.log({"Losses/val": np.mean(val_losses)}, new_experience)
    # wandb.log({"Losses/ent": np.mean(ent_bonuses)}, new_experience)
    # wandb.log({"Debug/%clip_trig": 100*np.mean(clip_trigger_fracs)}, new_experience)
    # wandb.log({"Debug/approx_kl": np.mean(approx_kls)}, new_experience)
    # wandb.log({"Debug/clip_epsilon": clip_epsilon*alpha}, new_experience)

# One more eval
_, key_eval = jax.random.split(key)
returns = evaluate(env, key_eval, model_params, model, n_actions, n_eval_agents, eval_discount)
avg_return, std_return = jnp.mean(returns), jnp.std(returns)
evals[new_experience, steps+1] = (avg_return, std_return)
# wandb.log({"Returns/avg": avg_return,
#            "Returns/avg+std": avg_return + std_return,
#            "Returns/avg-std": avg_return - std_return}, new_experience)


print("\nReturns avg ± std:")
for exp, steps in evals:
    avg_return, std_return = evals[exp, steps]
    print(f"(Exp {exp}, steps {steps}) --> {avg_return} ± {std_return}")
print()

# wandb.finish()




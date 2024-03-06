import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import numpy as np
import torch
from torch.optim import Adam
import time
import copy
import trpocvar_core as core
from utils.logx import EpochLogger, setup_logger_kwargs, colorize
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs, mpi_sum
from safe_rl_envs.envs.engine import Engine as safe_rl_envs_Engine
from utils.safe_rl_env_config import configuration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

class TRPOCVaRBuffer:
    """
    A buffer for storing trajectories experienced by a TRPOCVaR agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf          = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf          = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)

        self.rew_buf          = np.zeros(size, dtype=np.float32)
        self.ret_buf          = np.zeros(size, dtype=np.float32)
        self.val_buf          = np.zeros(size, dtype=np.float32)
        self.adv_buf          = np.zeros(size, dtype=np.float32)

        self.cost_buf         = np.zeros(size, dtype=np.float32)
        self.cost_ret_buf     = np.zeros(size, dtype=np.float32)

        self.aug_cost_ret_buf = np.zeros(size, dtype=np.float32)
        self.aug_cost_val_buf = np.zeros(size, dtype=np.float32)
        self.aug_cost_adv_buf = np.zeros(size, dtype=np.float32)

        self.logp_buf         = np.zeros(size, dtype=np.float32)
        self.mu_buf           = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.logstd_buf       = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, cost, aug_cost_val, mu, logstd):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr]          = obs
        self.act_buf[self.ptr]          = act
        self.rew_buf[self.ptr]          = rew
        self.val_buf[self.ptr]          = val
        self.logp_buf[self.ptr]         = logp
        self.cost_buf[self.ptr]         = cost
        self.aug_cost_val_buf[self.ptr] = aug_cost_val
        self.mu_buf[self.ptr]           = mu
        self.logstd_buf[self.ptr]       = logstd
        self.ptr += 1

    def finish_path(self, last_val=0, last_cost_val=0, last_aug_cost_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        costs = np.append(self.cost_buf[path_slice], last_cost_val)
        aug_costs = np.append(np.zeros_like(costs[:-1]), last_aug_cost_val)
        aug_cost_vals = np.append(self.aug_cost_val_buf[path_slice], last_aug_cost_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # augmented cost advantage calculation
        aug_cost_deltas = self.gamma * aug_cost_vals[1:] - aug_cost_vals[:-1]
        self.aug_cost_adv_buf[path_slice] = core.discount_cumsum(aug_cost_deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        # costs-to-go, targets for the cost value function
        self.cost_ret_buf[path_slice] = core.discount_cumsum(costs, self.gamma)[:-1]

        # augmented costs-to-go, targets for the augmented cost value function
        self.aug_cost_ret_buf[path_slice] = core.discount_cumsum(aug_costs, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        # center augmented cost advantage, but don't scale
        aug_cost_adv_mean, _ = mpi_statistics_scalar(self.aug_cost_adv_buf)
        self.aug_cost_adv_buf = (self.aug_cost_adv_buf - aug_cost_adv_mean)
        data = dict(obs=torch.FloatTensor(self.obs_buf).to(device), 
                    act=torch.FloatTensor(self.act_buf).to(device), 
                    ret=torch.FloatTensor(self.ret_buf).to(device),
                    adv=torch.FloatTensor(self.adv_buf).to(device),
                    cost_ret=torch.FloatTensor(self.cost_ret_buf).to(device),
                    aug_cost_ret=torch.FloatTensor(self.aug_cost_ret_buf).to(device),
                    aug_cost_adv=torch.FloatTensor(self.aug_cost_adv_buf).to(device),
                    logp=torch.FloatTensor(self.logp_buf).to(device),
                    mu=torch.FloatTensor(self.mu_buf).to(device),
                    logstd=torch.FloatTensor(self.logstd_buf).to(device))
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def get_net_param_np_vec(net):
    """
    Get the parameters of the network as numpy vector
    """
    return torch.cat([val.flatten() for val in net.parameters()], axis=0).detach().cpu().numpy()

def assign_net_param_from_flat(param_vec, net):
    param_sizes = [np.prod(list(val.shape)) for val in net.parameters()]
    ptr = 0
    for s, param in zip(param_sizes, net.parameters()):
        param.data.copy_(torch.from_numpy(param_vec[ptr:ptr+s]).reshape(param.shape))
        ptr += s

def cg(Ax, b, cg_iters=100):
    x = np.zeros_like(b)
    r = b.copy() # Note: should be 'b - Ax', but for x=0, Ax=0. Change if doing warm start.
    p = r.copy()
    r_dot_old = np.dot(r,r)
    for _ in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r,r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
        # early stopping 
        if np.linalg.norm(p) < EPS:
            break
    return x

def auto_grad(objective, net, to_numpy=True):
    """
    Get the gradient of the objective with respect to the parameters of the network
    """
    grad = torch.autograd.grad(objective, net.parameters(), create_graph=True)
    if to_numpy:
        return torch.cat([val.flatten() for val in grad], axis=0).detach().cpu().numpy()
    else:
        return torch.cat([val.flatten() for val in grad], axis=0)

def auto_hession_x(objective, net, x):
    """
    Returns 
    """
    jacob = auto_grad(objective, net, to_numpy=False)
    
    return auto_grad(torch.dot(jacob, x), net, to_numpy=True)

def trpocvar(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
             steps_per_epoch=4000, epochs=50, gamma=0.99, alpha=0.3, vf_lr=1e-3, vcf_lr=1e-3, 
             vc_aug_lr=1e-3, var_lr=1e-2, lam_lr=1e-2, train_v_iters=80, train_vc_iters=80,
             train_vc_aug_iters=80, lam=0.97, max_ep_len=1000, target_kl=0.01, 
             target_cost=1.5, logger_kwargs=dict(), save_freq=10, backtrack_coeff=0.8, 
             backtrack_iters=100, model_save=False, exp_name=None, lam_ratio=1.0):
    """
    Trust Region Policy Optimization - Conditional Value-at-Risk - Lagrangian Method
 
    Reference:
    Zhang, Q., Leng, S., Ma, X., Liu, Q., Wang, X., Liang, B., ... & Yang, J. (2024).
    CVaR-Constrained Policy Optimization for Safe Reinforcement Learning.
    IEEE Transactions on Neural Networks and Learning Systems.

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to TRPOCVaR.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        alpha (float): confidence level. (Always between 0 and 1. Below 0.5 in the reference paper.)

        vf_lr (float): Learning rate for value function optimizer.
        
        vcf_lr (float): Learning rate for cost value function optimizer.

        vc_aug_lr (float): Learning rate for augmented cost value function optimizer.

        lam_lr (float): Learning rate for Lagrangian multiplier.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
            
        train_vc_iters (int): Number of gradient descent steps to take on 
            cost value function per epoch.

        train_vc_aug_iters (int): Number of gradient descent steps to take on 
            augmented cost value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)
            
        target_cost (float): Cost limit that the agent should satisfy

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
            
        backtrack_coeff (float): Scaling factor for line search.
        
        backtrack_iters (int): Number of line search steps.
        
        model_save (bool): If saving model.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    assert len(env.observation_space.shape) == 1, 'Multi-dimensional observation space not supported!'
    obs_dim = (env.observation_space.shape[0] + 1,)
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = TRPOCVaRBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)


    def compute_kl_pi(data, cur_pi):
        """
        Return the sample average KL divergence between old and new policies
        """
        obs, mu_old, logstd_old = data['obs'], data['mu'], data['logstd']
        
        # Average KL Divergence  
        average_kl = cur_pi._d_kl(
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(mu_old, dtype=torch.float32),
            torch.as_tensor(logstd_old, dtype=torch.float32), device=device)
        
        return average_kl
            
    def compute_loss_pi(data, cur_pi):
        """
        The reward objective for TRPOCVaR (TRPOCVaR policy loss)
        """
        obs, act, adv, aug_cost_adv, logp_old = \
            data['obs'], data['act'], data['adv'], data['aug_cost_adv'], data['logp']
        
        # reward loss 
        pi, logp = cur_pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        loss_pi_reward = -(ratio * adv).mean()
        
        # lagrangian loss
        # get the Episode cost
        # Surrogate cost function 
        surr_cost = (ratio * aug_cost_adv).mean()        
        lag_term = ac.lmd * surr_cost * lam_ratio
        
        # total policy loss
        loss_pi = loss_pi_reward + lag_term
        
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)
        
        return loss_pi, pi_info
        
    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up function for computing cost loss 
    def compute_loss_vc(data):
        obs, cost_ret = data['obs'], data['cost_ret']
        return ((ac.vc(obs) - cost_ret)**2).mean()

    # Set up function for computing cost loss 
    def compute_loss_vc_aug(data):
        obs, aug_cost_ret = data['obs'], data['aug_cost_ret']
        return ((ac.vc_aug(obs) - aug_cost_ret)**2).mean()

    # Set up optimizers for policy and value function
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    vcf_optimizer = Adam(ac.vc.parameters(), lr=vcf_lr)
    vc_aug_optimizer = Adam(ac.vc_aug.parameters(), lr=vc_aug_lr)

    # Set up model saving
    if model_save:
        logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data, ac.pi)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # TRPOCVaR policy update core impelmentation 
        loss_pi, pi_info = compute_loss_pi(data, ac.pi)
        g = auto_grad(loss_pi, ac.pi) # get the flatten gradient evaluted at pi old 
        kl_div = compute_kl_pi(data, ac.pi)
        Hx = lambda x: auto_hession_x(kl_div, ac.pi, torch.FloatTensor(x).to(device))
        x_hat    = cg(Hx, g)             # Hinv_g = H \ g
        
        s = x_hat.T @ Hx(x_hat)
            
        x_direction = np.sqrt(2 * target_kl / (s+EPS)) * x_hat
        
        # copy an actor to conduct line search 
        actor_tmp = copy.deepcopy(ac.pi)
        def set_and_eval(step):
            new_param = get_net_param_np_vec(ac.pi) - step * x_direction
            assign_net_param_from_flat(new_param, actor_tmp)
            kl = compute_kl_pi(data, actor_tmp)
            pi_l, _ = compute_loss_pi(data, actor_tmp)
            
            return kl, pi_l
        
        # update the policy such that the KL diveragence constraints are satisfied and loss is decreasing
        for j in range(backtrack_iters):
            kl, pi_l_new = set_and_eval(backtrack_coeff**j)
            
            if (kl.item() <= target_kl and pi_l_new.item() <= pi_l_old):
                print(colorize(f'Accepting new params at step %d of line search.'%j, 'green', bold=False))
                # update the policy parameter 
                new_param = get_net_param_np_vec(ac.pi) - backtrack_coeff**j * x_direction
                assign_net_param_from_flat(new_param, ac.pi)
                loss_pi, pi_info = compute_loss_pi(data, ac.pi) # re-evaluate the pi_info for the new policy
                break
            if j==backtrack_iters-1:
                print(colorize(f'Line search failed! Keeping old params.', 'yellow', bold=False))
                
        # update lambda (the Lagrangian multiplier)
        def get_cost_violation():
            EpAugCostRet = logger.get_stats('EpAugCostRet')[0]
            c = EpAugCostRet - (1 - alpha) * (target_cost - ac.var)  # augmented cost violation
            return c

        ac.lmd = max(0, ac.lmd + lam_lr * get_cost_violation())

        # Value function learning
        for _ in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()
            
        # Cost value function learning
        for _ in range(train_vc_iters):
            vcf_optimizer.zero_grad()
            loss_vc = compute_loss_vc(data)
            loss_vc.backward()
            mpi_avg_grads(ac.vc)    # average grads across MPI processes
            vcf_optimizer.step()

        # Augmented cost value function learning
        for _ in range(train_vc_aug_iters):
            vc_aug_optimizer.zero_grad()
            loss_vc_aug = compute_loss_vc_aug(data)
            loss_vc_aug.backward()
            mpi_avg_grads(ac.vc_aug)    # average grads across MPI processes
            vc_aug_optimizer.step()

        # update value-at-risk
        p = (data['cost_ret'] >= ac.var).float().mean().item()
        ac.var = ac.var - var_lr * (1 - 1 / (1 - alpha) * p)

        # Log changes from update        
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, VaR=ac.var, AboveVaRRate=p,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    # o, ep_ret, ep_len = env.reset(), 0, 0
    while True:
        try:
            o, ep_ret, ep_len = env.reset(), 0, 0
            o = np.concatenate((o, [ac.var]))  # augmented observation
            break
        except:
            print('reset environment is wrong, try next reset')
    ep_cost_ret, ep_cost = 0, 0
    cum_cost = 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, _, vc_aug, logp, mu, logstd = ac.step(torch.as_tensor(o, dtype=torch.float32))

            try: 
                next_o, r, d, info = env.step(a)
                assert 'cost' in info.keys()
                next_o = np.concatenate((next_o, [(o[-1] - info['cost']) / gamma]))
            except: 
                # simulation exception discovered, discard this episode 
                next_o, r, d = o, 0, True # observation will not change, no reward when episode done 
                info['cost'] = 0 # no cost when episode done     
            # Track cumulative cost over training
            cum_cost += info['cost']
            ep_ret += r
            ep_cost_ret += info['cost'] * (gamma ** t)
            ep_cost += info['cost']
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp, info['cost'], vc_aug, mu, logstd)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, vc, vc_aug, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                    vc = 0
                    vc_aug = info['cost'] - o[-1]  # terminal cost in augmented CMDP
                buf.finish_path(v, vc, vc_aug)
                if terminal:
                    # only save EpRet / EpLen / EpCostRet if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost,
                                 EpAugCostRet=max(ep_cost_ret - ac.var, 0))

                while True:
                    try:
                        o, ep_ret, ep_len = env.reset(), 0, 0
                        o = np.concatenate((o, [ac.var]))  # augmented observation
                        break
                    except:
                        print('reset environment is wrong, try next reset')
                ep_cost_ret, ep_cost = 0, 0

        # Save model
        if ((epoch % save_freq == 0) or (epoch == epochs-1)) and model_save:
            logger.save_state({'env': env}, None)

        # Perform TRPOCVaR update!
        update()
        
        #=====================================================================#
        #  Cumulative cost calculations                                       #
        #=====================================================================#
        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('EpCost', average_only=True)
        logger.log_tabular('CumulativeCost', cumulative_cost)
        logger.log_tabular('CostRate', cost_rate)
        logger.log_tabular('VVals', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('VaR', average_only=True)
        logger.log_tabular('AboveVaRRate', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        
        
def create_env(args):
    env =  safe_rl_envs_Engine(configuration(args.task))
    return env

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', type=str, default='Goal_Point_8Hazards')
    parser.add_argument('--target_cost', type=float, default=0.) # the cost limit for the environment
    parser.add_argument('--target_kl', type=float, default=0.02) # the kl divergence limit for TRPOCVaR
    parser.add_argument('--lam_lr', type=float, default=0.005) # the learning rate for lambda
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='trpocvar')
    parser.add_argument('--model_save', action='store_true')
    parser.add_argument('--lam_ratio', type=float, default=1.0)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi
    
    exp_name = args.task + '_' + args.exp_name \
                + '_' 'lamratio' + str(args.lam_ratio) \
                + '_' + 'epochs' + str(args.epochs)
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    # whether to save model
    model_save = True if args.model_save else False

    trpocvar(lambda : create_env(args), actor_critic=core.MLPActorCritic,
             ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, alpha=args.alpha,
             seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, logger_kwargs=logger_kwargs,
             target_cost=args.target_cost, model_save=model_save, target_kl=args.target_kl,
             lam_lr=args.lam_lr, exp_name=args.exp_name, lam_ratio=args.lam_ratio)


import torch
import numpy as np
import random
from networks import Actor, Critic
import torch.optim as optim
import torch.nn.functional as F

class REDQ_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                 state_size,
                 action_size,
                 replay_buffer,
                 batch_size,
                 random_seed,
                 lr,
                 hidden_size,
                 gamma,
                 tau,
                 device,
                 action_prior="uniform",
                 N=2,
                 M=2,
                 G=1):
        """Initialize an Agent object

        Args:
            state_size (int):    State size
            action_size (int):   Action size
            replay_buffer:       Experience Replay Buffer
            batch_size:          Batch size when learning
            random_seed (int):   Random seed
            lr (float):          Learning rate
            hidden_size (int):   Number of hidden units per layer
            gamma (float):       Discount factor
            tau (float):         Tau, soft-update parameter
            device (torch device): Training Device cpu or cuda:0
            action_prior (str, optional): Action prior. Defaults to "uniform".
            N (int, optional):   Number of Q-Network Ensemble. Defaults to 2.
            M (int, optional):   Number of the subset of the Critic for update calculation. Defaults to 2.
            G (int, optional):   Critic Updates per step, UTD-raio. Defaults to 1.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
                
        self.target_entropy = -action_size  # -dim(A)
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=lr) 
        self._action_prior = action_prior
        self.alphas = []
        print("Using: ", device)
        self.device = device
        
        # REDQ parameter
        self.N = N # number of critics in the ensemble
        self.M = M # number of target critics that are randomly selected
        self.G = G # Updates per step ~ UTD-ratio
        
        # Actor Network 
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size=hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr)     
        
        # Critic Network (w/ Target Network)
        self.critics = []
        self.target_critics = []
        self.optims = []
        for i in range(self.N):
            critic = Critic(state_size, action_size, i, hidden_size=hidden_size).to(device)

            optimizer = optim.Adam(critic.parameters(), lr=lr, weight_decay=0)
            self.optims.append(optimizer)
            self.critics.append(critic)
            target = Critic(state_size, action_size, i, hidden_size=hidden_size).to(device)
            self.target_critics.append(target)


        # Replay memory
        self.memory = replay_buffer
        

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        for update in range(self.G):
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(update, experiences)
            
    
    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        action, _, _ = self.actor_local.sample(state)
        return action.detach().cpu()[0]
    
    def eval(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        _, _ , action = self.actor_local.sample(state)
        return action.detach().cpu()[0]
    
    def learn(self, step, experiences):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # sample target critics
        idx = np.random.choice(len(self.critics), self.M, replace=False) # replace=False so that not picking the same idx twice
        

        # ---------------------------- update critic ---------------------------- #

        with torch.no_grad():
            # Get predicted next-state actions and Q values from target models
            next_action, next_log_prob, _ = self.actor_local.sample(next_states)
            # TODO: make this variable for possible more than two target critics
            Q_target1_next = self.target_critics[idx[0]](next_states, next_action.squeeze(0))
            Q_target2_next = self.target_critics[idx[1]](next_states, next_action.squeeze(0))
            
            # take the min of both critics for updating
            Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * next_log_prob

        Q_targets = rewards.cpu() + (self.gamma * (1 - dones.cpu()) * Q_target_next.cpu())

        # Compute critic losses and update critics 
        for critic, optim, target in zip(self.critics, self.optims, self.target_critics):
            Q = critic(states, actions).cpu()
            Q_loss = 0.5*F.mse_loss(Q, Q_targets)
        
            # Update critic
            optim.zero_grad()
            Q_loss.backward()
            optim.step()
            # soft update of the targets
            self.soft_update(critic, target)
        
        # ---------------------------- update actor ---------------------------- #
        if step == self.G-1:

            actions_pred, log_prob, _ = self.actor_local.sample(states)             
            
            # TODO: make this variable for possible more than two critics
            Q1 = self.critics[idx[0]](states, actions_pred.squeeze(0)).cpu()
            Q2 = self.critics[idx[0]](states, actions_pred.squeeze(0)).cpu()
            Q = torch.min(Q1,Q2)

            actor_loss = (self.alpha * log_prob.cpu() - Q ).mean()
            # Optimize the actor loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Compute alpha loss 
            alpha_loss = - (self.log_alpha.exp() * (log_prob.cpu() + self.target_entropy).detach().cpu()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()
            self.alphas.append(self.alpha.detach())


    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

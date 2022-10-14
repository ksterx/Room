import torch
from room.train.agent import Agent


class PPO(Agent):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.actor = Actor(env, args)
        self.critic = Critic(env, args)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.memory = Memory()
        self.steps = 0
        self.episodes = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action, log_prob = self.actor.sample(state)
        return action.item(), log_prob

    def update(self):
        self.steps += 1
        state, action, reward, next_state, done, log_prob = self.memory.sample()
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(1 - done).to(device)
        log_prob = torch.FloatTensor(log_prob).to(device)

        # Critic update
        value = self.critic(state)
        next_value = self.critic(next_state)
        expected_value = reward + done * self.args.gamma * next_value
        critic_loss = F.mse_loss(value, expected_value.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        ratio = torch.exp(self.actor(state) - log_prob)
        advantage = expected_value - value
        actor_loss = -torch.min(
            ratio * advantage, torch.clamp(ratio, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantage
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def train(self):
        state = self.env.reset()
        score = 0
        while self.steps < self.args.max_steps:
            action, log_prob = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.memory.push(state, action, reward, next_state, done, log_prob)
            state = next_state
            score += reward
            if done:
                self.episodes += 1
                self.writer.add_scalar("train/score", score, self.episodes)

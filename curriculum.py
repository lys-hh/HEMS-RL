class CurriculumWrapper:
    def __init__(self, env):
        self.env = env
        self.curriculum_stage = 0  # 0~8阶段
        self.episode_count = 0  # 记录当前完成的episode数量

    def step(self, state, action):
        # 调用环境的 step 方法
        next_state, _, done = self.env.step(state, action)

        # 计算原始奖励
        original_reward = self.env.calculate_reward(state, action)

        # 计算惩罚项
        penalty = self.calculate_penalty()

        # 动态调整惩罚系数
        if self.curriculum_stage < 3:
            scaled_penalty = self.env.departure_penalty_scale * (self.curriculum_stage / 3) * penalty
            reward = original_reward + scaled_penalty
        else:
            reward = original_reward + self.env.departure_penalty_scale * penalty

        # 每50个 episode 提升难度
        self.episode_count += 1
        if self.episode_count % 50 == 0:
            self.curriculum_stage = min(4, self.curriculum_stage + 1)

        return next_state, reward, done

    def calculate_penalty(self):
        # # 示例：计算惩罚项
        # if self.env.data_interface.is_ev_departing_soon(self.env.current_time, self.env.current_time_index):
        #     required_charge = max(self.env.ev_min_charge - self.env.state['ev_battery_state'], 0)
        #     hours_remaining = self.env.data_interface.get_hours_until_departure(self.env.current_time,
        #                                                                         self.env.current_time_index)
        #     penalty = (required_charge / max(hours_remaining, 0.5)) ** 2  # 平方惩罚项
        # else:
        #     penalty = 0
        penalty = 0
        return penalty
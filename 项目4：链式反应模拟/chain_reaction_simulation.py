import numpy as np
import matplotlib.pyplot as plt


class AdvancedReactorSimulator:
    def __init__(self,
                 initial_neutrons=1e3,  # 初始中子数（蒙特卡罗模式为粒子数）
                 reproduction_factor=1.0,  # 有效增殖系数k
                 neutron_lifetime=1e-4,  # 中子平均寿命（秒）
                 simulation_time=0.1,  # 总模拟时间（秒）
                 time_step=1e-5,  # 时间步长
                 use_delayed_neutrons=False,  # 是否启用缓发中子
                 beta=0.0075,  # 缓发中子份额（占总中子比例）
                 delayed_lifetime=0.1  # 缓发中子先驱核平均寿命（秒）
                 ):
        self.N = initial_neutrons
        self.k = reproduction_factor
        self.lambda_ = 1 / neutron_lifetime  # 瞬发中子衰减常数
        self.dt = time_step
        self.t_end = simulation_time
        self.times = [0.0]
        self.neutron_counts = [self.N]
        self.use_delayed = use_delayed_neutrons
        self.beta = beta  # 缓发中子占比
        self.lambda_d = 1 / delayed_lifetime  # 缓发中子先驱核衰变常数
        self.C = 0.0  # 缓发中子先驱核浓度（仅在启用时使用）

    def _update_with_prompt_neutrons(self):
        """仅含瞬发中子的点堆方程"""
        dNdt = ((self.k - 1) / self.lambda_) * self.N
        self.N += dNdt * self.dt

    def _update_with_delayed_neutrons(self):
        """含瞬发+缓发中子的点堆方程（6组缓发简化为单组）"""
        # 瞬发中子项：k*(1-beta)为瞬发增殖系数
        prompt_term = (self.k * (1 - self.beta) - 1) / self.lambda_ * self.N
        # 缓发中子项：beta*k*N为产生的缓发中子先驱核，C*lambda_d为衰变产生的缓发中子
        delayed_term = (self.k * self.beta * self.N - self.C * self.lambda_d) / self.lambda_
        dNdt = prompt_term + delayed_term

        # 更新先驱核浓度：dC/dt = beta*k*N/lambda_ - lambda_d*C
        dCdt = (self.k * self.beta * self.N) / self.lambda_ - self.lambda_d * self.C
        self.C += dCdt * self.dt

        self.N += dNdt * self.dt

    def deterministic_simulation(self):
        """确定性模拟（微分方程解法）"""
        while len(self.times) * self.dt < self.t_end:
            if self.use_delayed:
                self._update_with_delayed_neutrons()
            else:
                self._update_with_prompt_neutrons()
            self.times.append(len(self.times) * self.dt)
            self.neutron_counts.append(self.N)
        return self.times, self.neutron_counts

    def monte_carlo_simulation(self, num_particles=None):
        """蒙特卡罗随机模拟（粒子数统计）"""
        if num_particles is None:
            num_particles = int(self.N)
        mc_times = [0.0]
        mc_counts = [num_particles]
        steps = int(self.t_end / self.dt)

        for _ in range(steps):
            # 每个时间步内的中子增殖服从泊松分布
            new_particles = np.random.poisson(self.k * num_particles)
            num_particles = new_particles
            mc_times.append((_ + 1) * self.dt)
            mc_counts.append(num_particles)

        self.mc_times = mc_times
        self.mc_counts = mc_counts
        return mc_times, mc_counts


def parameter_sweep_analysis():
    """参数扫描：对比k、中子寿命、是否启用缓发中子的影响"""
    # 基础参数
    base_config = {
        'initial_neutrons': 1e4,
        'simulation_time': 0.5,
        'time_step': 1e-5
    }

    # 场景1：增殖系数k的影响（确定性模拟）
    plt.figure(figsize=(12, 8))

    # 场景2：含缓发中子的临界控制（k=1.001，对比有无缓发）
    reactors = [
        AdvancedReactorSimulator(k=1.001, **base_config, use_delayed=False, label='瞬发中子(k=1.001)'),
        AdvancedReactorSimulator(k=1.001, **base_config, use_delayed=True, label='缓发中子(k=1.001)'),
        AdvancedReactorSimulator(k=0.999, **base_config, use_delayed=True, label='次临界(k=0.999)')
    ]

    # 场景3：中子寿命对比（k=1.01，寿命影响反应速率）
    lifetimes = [1e-5, 1e-4, 1e-3]
    for tau in lifetimes:
        reactor = AdvancedReactorSimulator(
            k=1.01,
            neutron_lifetime=tau,
            **base_config,
            label=f'τ={tau}s(k=1.01)'
        )
        reactor.deterministic_simulation()
        plt.plot(reactor.times, reactor.neutron_counts, linestyle='--')

    # 场景4：蒙特卡罗随机模拟（k=1.02，对比确定性结果）
    det_reactor = AdvancedReactorSimulator(k=1.02, **base_config, label='确定性(k=1.02)')
    det_reactor.deterministic_simulation()
    mc_reactor = AdvancedReactorSimulator(k=1.02, **base_config)
    mc_reactor.monte_carlo_simulation()

    # 统一绘图
    for reactor in reactors + [det_reactor]:
        plt.plot(reactor.times, reactor.neutron_counts, label=reactor.label)
    plt.plot(mc_reactor.mc_times, mc_reactor.mc_counts, 'o', label='蒙特卡罗(k=1.02)', alpha=0.5)

    plt.xlabel('时间（秒）')
    plt.ylabel('中子数（对数坐标）')
    plt.title('链式反应参数影响对比')
    plt.yscale('log')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parameter_sweep_analysis()

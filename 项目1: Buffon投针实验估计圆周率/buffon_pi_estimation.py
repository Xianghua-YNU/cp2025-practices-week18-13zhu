import numpy as np
import matplotlib.pyplot as plt


def buffon_experiment(n, l=1.0, d=2.0):
    """
    模拟Buffon投针实验

    参数:
    n (int): 实验次数
    l (float): 针的长度，默认为1.0
    d (float): 平行线间距，默认为2.0

    返回:
    float: π的估计值
    int: 相交次数
    """
    # 生成随机数
    y = np.random.uniform(0, d, n)  # 针的中心点位置
    theta = np.random.uniform(0, np.pi / 2, n)  # 针与平行线的夹角

    # 判断相交情况
    cross = (y <= (l / 2) * np.sin(theta)) | ((d - y) <= (l / 2) * np.sin(theta))
    m = np.sum(cross)  # 相交次数

    # 计算π的估计值
    if m == 0:
        return None, 0
    pi_estimate = (2 * l * n) / (m * d)
    return pi_estimate, m


def main():
    # 不同实验次数
    experiment_counts = [100, 1000, 10000, 100000, 1000000]
    pi_estimates = []
    crosses = []

    # 运行实验
    print("实验结果:")
    for n in experiment_counts:
        pi_est, m = buffon_experiment(n)
        pi_estimates.append(pi_est)
        crosses.append(m)
        print(f"实验次数: {n}, 相交次数: {m}, π估计值: {pi_est:.6f}")

    # 绘制实验结果图表
    plt.figure(figsize=(10, 6))
    plt.plot(experiment_counts, pi_estimates, 'o-', label='Estimated Value')
    plt.axhline(y=np.pi, color='r', linestyle='--', label='True Value')
    plt.xscale('log')
    plt.xlabel('Number of Experiments (log scale)')
    plt.ylabel('Estimated π Value')
    plt.title('Buffon\'s Needle Experiment: Effect of Experiment Count on π Estimation')
    plt.legend()
    plt.grid(True)
    plt.savefig('buffon_pi_results.png')
    plt.show()


if __name__ == "__main__":
    main()

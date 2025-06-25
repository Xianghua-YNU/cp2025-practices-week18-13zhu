import numpy as np

# 定义被积函数
def integrand(x):
    return x**(-1/2) * np.exp(x + 1)

# 生成满足 p(x) = 1/(2x) 分布的随机数
def generate_random_numbers(N):
    u1 = np.random.rand()
    u2 = np.random.rand()
    for _ in range(N):
        x = -np.log(u1) / (u2 - 1)
        yield x

# 估计积分值
def estimate_integral(N, random_numbers):
    return sum(integrand(x) for x in random_numbers) / N

# 计算方差
def calculate_variance(N, random_numbers, estimated_integral):
    f = lambda x: integrand(x)
    variance = sum((f(x) - estimated_integral)**2 for x in random_numbers) / (N - 1)
    return variance

# 估计误差
def estimate_error(N, random_numbers, estimated_integral):
    variance = calculate_variance(N, random_numbers, estimated_integral)
    return np.sqrt(variance / N)

# 主函数
if __name__ == "__main__":
    N = 1000000  # 样本数量

    # 生成随机数
    random_numbers = list(generate_random_numbers(N))

    # 估计积分值
    estimated_integral = estimate_integral(N, random_numbers)

    # 估计统计误差
    estimated_error = estimate_error(N, random_numbers, estimated_integral)

    # 输出结果
    print(f"Estimated Integral: {estimated_integral}")
    print(f"Estimated Error: {estimated_error}")

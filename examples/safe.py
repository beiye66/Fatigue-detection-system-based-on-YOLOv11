from __future__ import annotations
def safe_divide(numerator: float, denominator: float) -> float | None:
    """安全除法：除数为 0 时返回 None，避免 ZeroDivisionError。"""
    if denominator == 0:
        return None
    return numerator / denominator
def average(values: list[float]) -> float | None:
    """计算平均值；空列表返回 None。"""
    if not values:
        return None
    return sum(values) / len(values)

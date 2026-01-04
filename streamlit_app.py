import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. 获取ETF数据 =====
def get_etf_data(symbol):
    df = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="hfq")
    df = df[["日期", "收盘"]]
    df.rename(columns={"日期": "date", "收盘": symbol}, inplace=True)
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

# ===== 2. 计算因子和近期涨幅 =====
def calculate_factors_and_recent_return(data, window=21):
    for symbol in data.columns:
        data[f"{symbol}_daily_return"] = data[symbol].pct_change()
        data[f"{symbol}_factor"] = data[symbol].pct_change(periods=window)
        data[f"{symbol}_recent_return"] = data[symbol].pct_change(periods=window)
    return data.dropna()

# ===== 3. 生成信号 + 涨幅过快判断 =====
def generate_signal_with_fast_rise_filter(data, symbols, fast_rise_threshold=0.2):
    factor_columns = [f"{symbol}_factor" for symbol in symbols]
    data["signal"] = data[factor_columns].idxmax(axis=1).str.split("_").str[0]
    data["recent_return"] = data.apply(
        lambda row: row[f"{row['signal']}_recent_return"], axis=1
    )
    data["too_fast"] = data["recent_return"] > fast_rise_threshold
    return data.dropna()

# ===== 4. 计算策略收益 =====
def calculate_strategy_return(data, symbols):
    data["strategy_daily_return"] = data.apply(
        lambda row: 0 if row["too_fast"] else row[f"{row['signal']}_daily_return"], axis=1
    )
    data["strategy_cumulative"] = (1 + data["strategy_daily_return"]).cumprod()
    for symbol in symbols:
        data[f"{symbol}_cumulative"] = (1 + data[f"{symbol}_daily_return"]).cumprod()
    return data

# ===== 5. 计算绩效指标 =====
def calculate_metrics(data, symbols):
    metrics = {}
    for symbol in symbols + ["strategy"]:
        returns = data[f"{symbol}_daily_return"].dropna()
        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252
        annual_return = (1 + total_return) ** (1/years) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility != 0 else 0
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        metrics[symbol] = [annual_return, volatility, sharpe, max_drawdown]
    return pd.DataFrame(metrics, index=["年化收益率", "波动率", "夏普比率", "最大回撤"])

# ===== 6. 查看历史换仓信息 =====
def view_trade_history(data):
    trade_data = data.copy()
    trade_data["prev_signal"] = trade_data["signal"].shift(1)
    trade_data["prev_too_fast"] = trade_data["too_fast"].shift(1)
    trade_data["signal_change"] = trade_data["signal"] != trade_data["prev_signal"]
    trade_data["too_fast_change"] = trade_data["too_fast"] != trade_data["prev_too_fast"]
    trade_data["trade"] = trade_data["signal_change"] | trade_data["too_fast_change"]
    trades = trade_data[trade_data["trade"]].copy()
    trades["prev_position"] = trades.apply(
        lambda row: "空仓" if row["prev_too_fast"] else row["prev_signal"], axis=1
    )
    trades["curr_position"] = trades.apply(
        lambda row: "空仓" if row["too_fast"] else row["signal"], axis=1
    )
    return trades[["prev_position", "curr_position"]]

# ===== 7. 画图比较 =====
def plot_results(data, symbols):
    plt.figure(figsize=(12, 6))
    for symbol in symbols:
        plt.plot(data.index, data[f"{symbol}_cumulative"], label=symbol)
    plt.plot(data.index, data["strategy_cumulative"], label="轮动策略", linewidth=2)
    plt.xlabel("时间")
    plt.ylabel("累积收益")
    plt.legend()
    plt.title("ETF轮动策略 vs 单个ETF")
    plt.grid(True)
    return plt

# ===== Streamlit页面 =====
def main():
    st.set_page_config(page_title="ETF轮动策略回测", layout="wide")
    st.title("ETF轮动策略回测系统")

    # 参数设置
    etf_symbols = st.sidebar.multiselect(
        "选择ETF", ["518880", "513100", "159915", "510300"], default=["518880", "513100", "159915", "510300"]
    )
    window = st.sidebar.slider("滚动窗口", 5, 60, 21)
    fast_rise_threshold = st.sidebar.slider("涨幅过快阈值", 0.05, 0.5, 0.2)

    # 点击运行回测
    if st.sidebar.button("运行回测"):
        etf_data = [get_etf_data(s) for s in etf_symbols]
        data = pd.concat(etf_data, axis=1).dropna()
        data = calculate_factors_and_recent_return(data, window)
        data = generate_signal_with_fast_rise_filter(data, etf_symbols, fast_rise_threshold)
        data = calculate_strategy_return(data, etf_symbols)
        metrics = calculate_metrics(data, etf_symbols)
        trades = view_trade_history(data)

        # 直接显示结果
        st.subheader("策略绩效指标")
        st.dataframe(metrics, use_container_width=True)

        st.subheader("收益曲线图")
        fig = plot_results(data, etf_symbols)
        st.pyplot(fig, use_container_width=True)

        st.subheader("历史换仓记录")
        st.dataframe(trades, use_container_width=True)

if __name__ == "__main__":
    main()

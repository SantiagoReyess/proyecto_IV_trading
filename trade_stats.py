# trade_stats.py
import numpy as np

def compute_trade_stats(trades):
    """
    trades: lista de objetos Trade (definidos en backtesting)
            con atributos:
              - pnl_net
              - pnl_gross
              - return_pct
              - holding_period
              - side
              - commissions
    """
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "num_trades": 0,
            "win_rate": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "profit_factor": np.nan,
            "avg_return_pct": np.nan,
            "avg_holding_period_days": np.nan,
            "total_gross_profit": 0.0,
            "total_gross_loss": 0.0,
            "total_net_pnl": 0.0,
            "total_commissions": 0.0
        }

    pnls = np.array([t.pnl_net for t in trades])
    gross_pnls = np.array([t.pnl_gross for t in trades])
    returns = np.array([t.return_pct for t in trades])
    holding = np.array([t.holding_period for t in trades])
    comms = np.array([t.commissions for t in trades])

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    num_wins = len(wins)
    num_losses = len(losses)

    win_rate = num_wins / n_trades if n_trades > 0 else np.nan
    avg_win = wins.mean() if num_wins > 0 else 0.0
    avg_loss = losses.mean() if num_losses > 0 else 0.0

    total_gross_profit = wins.sum() if num_wins > 0 else 0.0
    total_gross_loss = abs(losses.sum()) if num_losses > 0 else 0.0
    profit_factor = (total_gross_profit / total_gross_loss) if total_gross_loss > 0 else np.nan

    total_net_pnl = pnls.sum()
    total_commissions = comms.sum()

    return {
        "num_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_return_pct": returns.mean() if len(returns) > 0 else np.nan,
        "avg_holding_period_days": holding.mean() if len(holding) > 0 else np.nan,
        "total_gross_profit": total_gross_profit,
        "total_gross_loss": total_gross_loss,
        "total_net_pnl": total_net_pnl,
        "total_commissions": total_commissions,
    }

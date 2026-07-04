# 台股/台指期 回測面板（單檔版，適合手機/雲端部署）
# 只需這個 app.py + requirements.txt 兩個檔即可運行。
from __future__ import annotations

# --- backtester/instrument.py ---
from dataclasses import dataclass, field


@dataclass
class Instrument:
    """商品基底類別。"""
    symbol: str
    name: str = ""
    is_future: bool = False
    multiplier: float = 1.0          # 契約乘數 (股票=1；大台=200)

    # 交易成本
    commission_rate: float = 0.0     # 依名目價金比例計收 (股票)
    commission_per_contract: float = 0.0  # 每口固定手續費 (期貨)
    min_commission: float = 0.0      # 最低手續費
    tax_rate: float = 0.0            # 交易稅率
    tax_on_sell_only: bool = True    # True=僅賣出課稅 (股票證交稅)；False=買賣皆課 (期交稅)

    # 期貨專用
    initial_margin: float = 0.0      # 每口原始保證金

    def notional(self, price: float, qty: float) -> float:
        """名目價金 = 價格 * |數量| * 乘數。"""
        return abs(qty) * price * self.multiplier

    def commission(self, price: float, qty: float) -> float:
        if self.is_future:
            fee = abs(qty) * self.commission_per_contract
        else:
            fee = self.notional(price, qty) * self.commission_rate
        return max(fee, self.min_commission) if abs(qty) > 0 else 0.0

    def tax(self, price: float, qty: float) -> float:
        """qty 帶符號：>0 買進、<0 賣出。"""
        if abs(qty) == 0:
            return 0.0
        is_sell = qty < 0
        if self.tax_on_sell_only and not is_sell:
            return 0.0
        return self.notional(price, qty) * self.tax_rate

    def equity_contribution(self, qty: float, avg_price: float, price: float) -> float:
        """部位對總權益的貢獻 (見模組說明)。"""
        if self.is_future:
            return (price - avg_price) * self.multiplier * qty
        return qty * price

    def margin_requirement(self, qty: float) -> float:
        return abs(qty) * self.initial_margin if self.is_future else 0.0


# ---------- 工廠函式：常見台灣商品 ----------

def tw_stock(symbol: str, name: str = "", *, fee_discount: float = 1.0,
             day_trade: bool = False) -> Instrument:
    """
    台股 (上市/上櫃)。
    - 手續費 0.1425%，可乘券商折數 fee_discount (例如 0.6 = 6 折)，最低 20 元。
    - 證交稅 賣出 0.3%；當沖 day_trade=True 時減半為 0.15%。
    - multiplier 用 1：以「股」為單位下單 (1 張 = 1000 股，數量請自行 *1000)。
    """
    return Instrument(
        symbol=symbol, name=name, is_future=False, multiplier=1.0,
        commission_rate=0.001425 * fee_discount,
        min_commission=20.0,
        tax_rate=0.0015 if day_trade else 0.003,
        tax_on_sell_only=True,
    )


def tx_future(symbol: str = "TX", name: str = "台指期(大台)", *,
              multiplier: float = 200.0, initial_margin: float = 167000.0,
              commission_per_contract: float = 40.0) -> Instrument:
    """台指期大台：每點 200 元；期交稅 十萬分之二 (0.00002) 買賣皆課。"""
    return Instrument(
        symbol=symbol, name=name, is_future=True, multiplier=multiplier,
        commission_per_contract=commission_per_contract,
        tax_rate=0.00002, tax_on_sell_only=False,
        initial_margin=initial_margin,
    )


def mtx_future(symbol: str = "MTX", name: str = "小型台指(小台)") -> Instrument:
    """小台：每點 50 元，保證金約 1/4。"""
    return tx_future(symbol, name, multiplier=50.0, initial_margin=41750.0,
                     commission_per_contract=30.0)

# --- backtester/data.py ---
import os
from typing import Optional
import numpy as np
import pandas as pd

_COLS = ["open", "high", "low", "close", "volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df[[c for c in _COLS if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


class YFinanceFeed:
    """以 yfinance 下載資料，並快取到 cache_dir。"""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get(self, symbol: str, start: str = "2018-01-01",
            end: Optional[str] = None, interval: str = "1d",
            use_cache: bool = True) -> pd.DataFrame:
        key = f"{symbol}_{start}_{end}_{interval}".replace("^", "IDX_").replace("/", "_")
        path = os.path.join(self.cache_dir, key + ".csv")
        if use_cache and os.path.exists(path):
            return _normalize(pd.read_csv(path, index_col=0))

        import yfinance as yf  # 延遲匯入，未安裝也不影響其他功能
        raw = yf.download(symbol, start=start, end=end, interval=interval,
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"yfinance 沒有抓到 {symbol} 的資料，請確認代碼/日期。")
        if isinstance(raw.columns, pd.MultiIndex):  # 單一標的時攤平欄位
            raw.columns = raw.columns.get_level_values(0)
        df = _normalize(raw)
        df.to_csv(path)
        return df


class CsvFeed:
    """讀取本地 CSV (需含 open/high/low/close[/volume] 欄與日期 index)。"""

    def __init__(self, directory: str = "."):
        self.directory = directory

    def get(self, symbol: str, **_) -> pd.DataFrame:
        path = symbol if os.path.exists(symbol) else os.path.join(self.directory, f"{symbol}.csv")
        return _normalize(pd.read_csv(path, index_col=0))


class SyntheticFeed:
    """模擬資料：幾何布朗運動 + 日內高低波動，僅供離線測試/示範。"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def get(self, symbol: str, start: str = "2020-01-01", periods: int = 1000,
            s0: float = 18000.0, mu: float = 0.08, sigma: float = 0.20,
            **_) -> pd.DataFrame:
        dt = 1 / 252
        rets = self.rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), periods)
        close = s0 * np.exp(np.cumsum(rets))
        opens = np.concatenate([[s0], close[:-1]])
        intraday = np.abs(self.rng.normal(0, sigma * np.sqrt(dt) * 0.7, periods)) * close
        high = np.maximum(opens, close) + intraday
        low = np.minimum(opens, close) - intraday
        vol = self.rng.integers(1000, 50000, periods).astype(float)
        idx = pd.bdate_range(start=start, periods=periods)
        return _normalize(pd.DataFrame(
            {"open": opens, "high": high, "low": low, "close": close, "volume": vol},
            index=idx))


# ---------- FinMind 純轉換函式 (可離線單元測試) ----------

# 已知的台指期貨代碼 (FinMind futures_id)；其餘純數字代碼視為股票
_KNOWN_FUTURES = {"TX", "MTX", "TMF", "TXF", "TE", "TF", "T5F", "XIF", "GTF"}


def _finmind_stock_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """TaiwanStockPrice -> 標準 OHLCV。欄位：open/max/min/close/Trading_Volume。"""
    out = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
    out = out.set_index("date")
    return _normalize(out)


def _finmind_select_front_month(df: pd.DataFrame, prefer_session: str = "position") -> pd.DataFrame:
    """
    期貨一天有多個到期月份與日/夜盤，這裡：
      1) 有 trading_session 欄時，優先取日盤 (prefer_session)，若濾完為空則退回全部
      2) 濾掉價差(spread)合約 (contract_date 含 '/')
      3) 每個交易日取「近月」= 最小且尚未到期(到期月 >= 當日年月)的合約，
         若無則退回該日最小到期月。
    產生近月連續序列 (月底換月，非回補式連續合約)。
    """
    d = df.copy()
    if "trading_session" in d.columns:
        reg = d[d["trading_session"].astype(str).str.lower() == prefer_session.lower()]
        if not reg.empty:
            d = reg
    d = d[~d["contract_date"].astype(str).str.contains("/")].copy()
    d["_ym"] = d["contract_date"].astype(str).str.slice(0, 6)
    d = d[d["_ym"].str.fullmatch(r"\d{6}")].copy()
    d["_ym"] = d["_ym"].astype(int)
    d["_dym"] = pd.to_datetime(d["date"]).dt.strftime("%Y%m").astype(int)

    # 每日取近月：優先「未到期(到期月>=當日年月)」中最小者，否則退回最小到期月。
    # 以排序 + 去重達成 (向量化，無 groupby.apply)：
    #   排序鍵 = (date, 未到期優先=0/已過期=1, 到期月由小到大)，每日取第一筆。
    d["_rank"] = (d["_ym"] < d["_dym"]).astype(int)
    d = d.sort_values(["date", "_rank", "_ym"])
    picked = d.drop_duplicates("date", keep="first")
    return picked.drop(columns=["_ym", "_dym", "_rank"])


def _finmind_futures_to_ohlcv(df: pd.DataFrame, prefer_session: str = "position") -> pd.DataFrame:
    """TaiwanFuturesDaily -> 標準 OHLCV (近月連續)。欄位：open/max/min/close/volume。"""
    picked = _finmind_select_front_month(df, prefer_session)
    out = picked.rename(columns={"max": "high", "min": "low"}).set_index("date")
    return _normalize(out)


class FinMindFeed:
    """
    FinMind 開源資料 (免開戶免費)。台股日線 + 真實台指期日線。

    代碼規則：
      - 純數字 (如 '2330'、'0050')         -> 台股日線
      - 期貨代碼 (如 'TX' 大台、'MTX' 小台) -> 期貨日線 (近月連續)
      - 也可用 asset='stock' / 'future' 明確指定

    token 非必填：未登入 300 次/小時；註冊後帶 token 提升到 600 次/小時。
    可用環境變數 FINMIND_TOKEN 提供。
    """

    def __init__(self, token: Optional[str] = None, cache_dir: str = ".cache",
                 prefer_session: str = "position"):
        self.token = token or os.getenv("FINMIND_TOKEN")
        self.cache_dir = cache_dir
        self.prefer_session = prefer_session
        os.makedirs(cache_dir, exist_ok=True)
        self._dl = None

    def _loader(self):
        if self._dl is None:
            from FinMind.data import DataLoader  # 延遲匯入
            self._dl = DataLoader()
            if self.token:
                self._dl.login_by_token(api_token=self.token)
        return self._dl

    @staticmethod
    def _is_future(symbol: str, asset: Optional[str]) -> bool:
        if asset:
            return asset.lower().startswith("fut")
        return not symbol.isdigit()          # 純數字=股票，其餘=期貨

    def get(self, symbol: str, start: str = "2018-01-01",
            end: Optional[str] = None, asset: Optional[str] = None,
            use_cache: bool = True) -> pd.DataFrame:
        end = end or ""
        is_fut = self._is_future(symbol, asset)
        kind = "fut" if is_fut else "stk"
        path = os.path.join(self.cache_dir, f"finmind_{kind}_{symbol}_{start}_{end}.csv")
        if use_cache and os.path.exists(path):
            return _normalize(pd.read_csv(path, index_col=0))

        dl = self._loader()
        if is_fut:
            raw = dl.taiwan_futures_daily(futures_id=symbol, start_date=start, end_date=end)
            if raw is None or raw.empty:
                raise ValueError(f"FinMind 沒有抓到期貨 {symbol} 的資料。")
            df = _finmind_futures_to_ohlcv(raw, self.prefer_session)
        else:
            raw = dl.taiwan_stock_daily(stock_id=symbol, start_date=start, end_date=end)
            if raw is None or raw.empty:
                raise ValueError(f"FinMind 沒有抓到股票 {symbol} 的資料。")
            df = _finmind_stock_to_ohlcv(raw)
        df.to_csv(path)
        return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """把日 K 聚合成週K/月K 等。rule 例：'W'(週)、'ME'(月)。"""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    cols = [c for c in agg if c in df.columns]
    out = df[cols].resample(rule).agg({c: agg[c] for c in cols})
    return out.dropna(subset=["open", "high", "low", "close"])

# --- backtester/portfolio.py ---
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd



@dataclass
class Position:
    qty: float = 0.0          # 帶符號：正=多、負=空
    avg_price: float = 0.0    # 平均進場價 (恆為正)

    def apply(self, dq: float, price: float, multiplier: float) -> float:
        """
        套用一筆成交 dq (帶符號)，回傳本次「已實現損益(以貨幣計)」。
        - 同方向加碼 -> 重算平均價，無已實現損益
        - 反方向 -> 平倉部分計算損益；若超量則反手建立新部位
        """
        realized = 0.0
        if self.qty == 0:
            self.qty, self.avg_price = dq, price
        elif (self.qty > 0) == (dq > 0):                      # 加碼
            total = self.qty + dq
            self.avg_price = (self.avg_price * abs(self.qty) + price * abs(dq)) / abs(total)
            self.qty = total
        else:                                                 # 反向：平倉/反手
            direction = 1 if self.qty > 0 else -1
            closing = min(abs(dq), abs(self.qty))
            realized = (price - self.avg_price) * closing * direction * multiplier
            self.qty += dq
            if self.qty == 0:
                self.avg_price = 0.0
            elif (self.qty > 0) != (direction > 0):           # 反手
                self.avg_price = price
        return realized


@dataclass
class Portfolio:
    initial_cash: float
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    instruments: Dict[str, Instrument] = field(default_factory=dict)
    equity_curve: List[dict] = field(default_factory=list)
    trades: List[dict] = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_cash

    def position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position())

    def register(self, inst: Instrument):
        self.instruments[inst.symbol] = inst

    # --- 評價 ---
    def equity(self, prices: Dict[str, float]) -> float:
        total = self.cash
        for sym, pos in self.positions.items():
            if pos.qty == 0 or sym not in prices:
                continue
            total += self.instruments[sym].equity_contribution(pos.qty, pos.avg_price, prices[sym])
        return total

    def margin_used(self) -> float:
        return sum(self.instruments[s].margin_requirement(p.qty)
                   for s, p in self.positions.items() if p.qty != 0)

    def record_equity(self, ts, prices: Dict[str, float]):
        self.equity_curve.append({"datetime": ts, "equity": self.equity(prices),
                                  "cash": self.cash, "margin_used": self.margin_used()})

    def record_trade(self, ts, symbol: str, qty: float, price: float,
                     fee: float, tax: float, realized: float):
        self.trades.append({"datetime": ts, "symbol": symbol, "qty": qty,
                            "price": price, "fee": fee, "tax": tax,
                            "realized_pnl": realized})

    def equity_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve).set_index("datetime") if self.equity_curve else pd.DataFrame()

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

# --- backtester/broker.py ---
from dataclasses import dataclass
from typing import Optional



@dataclass
class Order:
    instrument: Instrument
    quantity: float          # 帶符號
    note: str = ""


class Broker:
    def __init__(self, slippage: float = 0.0005, allow_margin_breach: bool = False,
                 verbose: bool = False):
        self.slippage = slippage          # 以成交價比例計
        self.allow_margin_breach = allow_margin_breach
        self.verbose = verbose

    def _fill_price(self, ref_price: float, qty: float) -> float:
        # 買進往上滑、賣出往下滑
        return ref_price * (1 + self.slippage) if qty > 0 else ref_price * (1 - self.slippage)

    def execute(self, order: Order, ref_price: float, portfolio: Portfolio, ts,
                force: bool = False) -> bool:
        """force=True 用於強制平倉(margin call)，跳過事前保證金/現金檢查。"""
        inst = order.instrument
        dq = order.quantity
        if dq == 0 or ref_price <= 0:
            return False
        price = self._fill_price(ref_price, dq)
        pos = portfolio.position(inst.symbol)

        # 預估成交後的現金與保證金，做事前檢查
        fee = inst.commission(price, dq)
        tax = inst.tax(price, dq)

        if not force and not self.allow_margin_breach:
            if inst.is_future:
                projected_qty = pos.qty + dq
                need = inst.margin_requirement(projected_qty) - inst.margin_requirement(pos.qty)
                # 反向減倉 need 可能為負(釋出保證金)，僅在加碼/反手需額外保證金時檢查
                if need > 0 and (portfolio.cash - portfolio.margin_used()) < need + fee + tax:
                    if self.verbose:
                        print(f"[拒單] {ts} {inst.symbol} 保證金不足 need={need:.0f}")
                    return False
            else:
                if dq > 0 and portfolio.cash < dq * price + fee + tax:
                    if self.verbose:
                        print(f"[拒單] {ts} {inst.symbol} 現金不足")
                    return False

        realized = pos.apply(dq, price, inst.multiplier)
        if inst.is_future:
            portfolio.cash += realized - fee - tax       # 期貨：僅損益與成本進出現金
        else:
            portfolio.cash += -dq * price - fee - tax     # 股票：名目價金進出現金

        portfolio.record_trade(ts, inst.symbol, dq, price, fee, tax, realized)
        if self.verbose:
            side = "買" if dq > 0 else "賣"
            print(f"[成交] {ts} {side} {inst.symbol} x{abs(dq):g} @ {price:.2f} "
                  f"fee={fee:.0f} tax={tax:.0f} realized={realized:.0f}")
        return True

# --- backtester/strategy.py ---
from typing import Dict, List, Optional
import numpy as np
import pandas as pd



class Strategy:
    def __init__(self, instrument: Instrument, lot_size: int = 1000, size_pct: float = 0.95,
                 contracts: int = 1, allow_short: Optional[bool] = None,
                 stop_loss: float = 0.0, take_profit: float = 0.0):
        self.instrument = instrument
        self.sym = instrument.symbol
        self.lot_size = lot_size
        self.size_pct = size_pct
        self.contracts = contracts
        self.allow_short = instrument.is_future if allow_short is None else allow_short
        self.stop_loss = stop_loss        # 0=關閉，0.05=5%
        self.take_profit = take_profit
        self._lock_dir = 0

    def prepare(self, data: Dict[str, pd.DataFrame]):
        pass

    def signal(self, ts, data, idx, portfolio) -> Optional[int]:
        return None

    # ---- 共用工具 ----
    def current_qty(self, portfolio: Portfolio) -> float:
        return portfolio.position(self.sym).qty

    def full_size(self, price: float, portfolio: Portfolio) -> float:
        if self.instrument.is_future:
            return self.contracts
        budget = portfolio.cash * self.size_pct
        return int(budget // (price * self.lot_size)) * self.lot_size

    def trade_to_target(self, target: int, price: float, portfolio: Portfolio,
                        note: str = "") -> List[Order]:
        if target < 0 and not self.allow_short:
            target = 0
        cur = self.current_qty(portfolio)
        if target > 0 and cur > 0:
            return []
        if target < 0 and cur < 0:
            return []
        if target == 0 and cur == 0:
            return []
        size = self.full_size(price, portfolio)
        if target != 0 and size == 0:
            return []
        dq = target * size - cur
        return [Order(self.instrument, dq, note)] if dq != 0 else []

    def on_bar(self, ts, data, idx, portfolio) -> List[Order]:
        i = idx.get(self.sym)
        if i is None:
            return []
        price = float(data[self.sym]["close"].iloc[i])
        pos = portfolio.position(self.sym)
        cur = pos.qty

        # 停損 / 停利
        if cur != 0 and (self.stop_loss > 0 or self.take_profit > 0) and pos.avg_price > 0:
            ret = (price / pos.avg_price - 1.0) if cur > 0 else (pos.avg_price / price - 1.0)
            hit_sl = self.stop_loss > 0 and ret <= -self.stop_loss
            hit_tp = self.take_profit > 0 and ret >= self.take_profit
            if hit_sl or hit_tp:
                self._lock_dir = 1 if cur > 0 else -1
                return [Order(self.instrument, -cur, "停損" if hit_sl else "停利")]

        target = self.signal(ts, data, idx, portfolio)
        if target is None:
            return []
        if self._lock_dir != 0 and target != self._lock_dir:
            self._lock_dir = 0
        if target != 0 and target == self._lock_dir:
            return []
        return self.trade_to_target(target, price, portfolio)


# ---------- 各策略（只需實作 prepare / signal） ----------

class MACrossStrategy(Strategy):
    """均線交叉：短均線在長均線之上做多。"""
    def __init__(self, instrument, fast: int = 20, slow: int = 60, **kw):
        super().__init__(instrument, **kw); self.fast, self.slow = fast, slow

    def prepare(self, data):
        df = data[self.sym]
        df["ma_fast"] = df["close"].rolling(self.fast).mean()
        df["ma_slow"] = df["close"].rolling(self.slow).mean()

    def signal(self, ts, data, idx, portfolio):
        i = idx.get(self.sym)
        if i is None or i < self.slow:
            return None
        df = data[self.sym]
        return 1 if df["ma_fast"].iloc[i] > df["ma_slow"].iloc[i] else (-1 if self.allow_short else 0)


class FuturesBreakoutStrategy(Strategy):
    """通道突破：突破近 N 根高點做多、跌破低點做空，否則抱單。"""
    def __init__(self, instrument, lookback: int = 20, **kw):
        super().__init__(instrument, **kw); self.lookback = lookback

    def prepare(self, data):
        df = data[self.sym]
        df["hh"] = df["high"].rolling(self.lookback).max().shift(1)
        df["ll"] = df["low"].rolling(self.lookback).min().shift(1)

    def signal(self, ts, data, idx, portfolio):
        i = idx.get(self.sym)
        if i is None or i < self.lookback:
            return None
        df = data[self.sym]
        c, hh, ll = df["close"].iloc[i], df["hh"].iloc[i], df["ll"].iloc[i]
        if c > hh: return 1
        if c < ll: return -1 if self.allow_short else 0
        return None


class RSIStrategy(Strategy):
    """RSI 均值回歸：超賣買進、超買出場。"""
    def __init__(self, instrument, period: int = 14, oversold: int = 30, overbought: int = 70, **kw):
        super().__init__(instrument, **kw)
        self.period, self.oversold, self.overbought = period, oversold, overbought

    def prepare(self, data):
        df = data[self.sym]; delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    def signal(self, ts, data, idx, portfolio):
        i = idx.get(self.sym)
        if i is None or i < self.period + 1:
            return None
        rsi = data[self.sym]["rsi"].iloc[i]
        if rsi < self.oversold: return 1
        if rsi > self.overbought: return -1 if self.allow_short else 0
        return None


class BollingerStrategy(Strategy):
    """布林通道均值回歸：跌破下軌買進、突破上軌出場。"""
    def __init__(self, instrument, period: int = 20, k: float = 2.0, **kw):
        super().__init__(instrument, **kw); self.period, self.k = period, k

    def prepare(self, data):
        df = data[self.sym]; ma = df["close"].rolling(self.period).mean()
        sd = df["close"].rolling(self.period).std()
        df["bb_up"], df["bb_dn"] = ma + self.k * sd, ma - self.k * sd

    def signal(self, ts, data, idx, portfolio):
        i = idx.get(self.sym)
        if i is None or i < self.period:
            return None
        df = data[self.sym]; c = df["close"].iloc[i]
        if c < df["bb_dn"].iloc[i]: return 1
        if c > df["bb_up"].iloc[i]: return -1 if self.allow_short else 0
        return None


class MACDStrategy(Strategy):
    """MACD 趨勢動能：MACD 在訊號線之上做多。"""
    def __init__(self, instrument, fast: int = 12, slow: int = 26, signal: int = 9, **kw):
        super().__init__(instrument, **kw)
        self.fast, self.slow, self.sig = fast, slow, signal

    def prepare(self, data):
        df = data[self.sym]
        macd = df["close"].ewm(span=self.fast, adjust=False).mean() - df["close"].ewm(span=self.slow, adjust=False).mean()
        df["macd"], df["macd_sig"] = macd, macd.ewm(span=self.sig, adjust=False).mean()

    def signal(self, ts, data, idx, portfolio):
        i = idx.get(self.sym)
        if i is None or i < self.slow + self.sig:
            return None
        df = data[self.sym]
        return 1 if df["macd"].iloc[i] > df["macd_sig"].iloc[i] else (-1 if self.allow_short else 0)


class KDStrategy(Strategy):
    """KD 隨機指標：K 在 D 之上做多。"""
    def __init__(self, instrument, period: int = 9, **kw):
        super().__init__(instrument, **kw); self.period = period

    def prepare(self, data):
        df = data[self.sym]
        low_n, high_n = df["low"].rolling(self.period).min(), df["high"].rolling(self.period).max()
        rsv = ((df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100).fillna(50)
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        df["kd_k"], df["kd_d"] = k, k.ewm(alpha=1/3, adjust=False).mean()

    def signal(self, ts, data, idx, portfolio):
        i = idx.get(self.sym)
        if i is None or i < self.period:
            return None
        df = data[self.sym]
        return 1 if df["kd_k"].iloc[i] > df["kd_d"].iloc[i] else (-1 if self.allow_short else 0)


class FibonacciStrategy(Strategy):
    """
    黃金分割回檔：以近 N 根的高低點為區間，計算 Fibonacci 回檔價位。
    價格回落到較深的買進回檔位(預設 0.618)時進場，反彈到較淺的出場位(預設 0.382)時出場。
    """
    FIBS = [0.236, 0.382, 0.5, 0.618, 0.786]

    def __init__(self, instrument, lookback: int = 60, buy_fib: float = 0.618,
                 exit_fib: float = 0.382, **kw):
        super().__init__(instrument, **kw)
        self.lookback, self.buy_fib, self.exit_fib = lookback, buy_fib, exit_fib

    def prepare(self, data):
        df = data[self.sym]
        df["fib_hh"] = df["high"].rolling(self.lookback).max().shift(1)
        df["fib_ll"] = df["low"].rolling(self.lookback).min().shift(1)

    def signal(self, ts, data, idx, portfolio):
        i = idx.get(self.sym)
        if i is None or i < self.lookback:
            return None
        df = data[self.sym]
        hh, ll, c = df["fib_hh"].iloc[i], df["fib_ll"].iloc[i], df["close"].iloc[i]
        rng = hh - ll
        if not (rng > 0):
            return None
        buy_price = hh - self.buy_fib * rng      # 較深回檔（接近低點）
        exit_price = hh - self.exit_fib * rng    # 較淺回檔（接近高點）
        if c <= buy_price: return 1
        if c >= exit_price: return 0
        return None


class CompositeStrategy(Strategy):
    """複合策略：結合兩個子策略的訊號。AND=兩者一致才進場；OR=任一成立即進場（衝突則空手）。"""
    def __init__(self, instrument, strat_a: Strategy, strat_b: Strategy, mode: str = "AND", **kw):
        super().__init__(instrument, **kw)
        self.a, self.b, self.mode = strat_a, strat_b, mode.upper()

    def prepare(self, data):
        self.a.prepare(data); self.b.prepare(data)

    def signal(self, ts, data, idx, portfolio):
        a = self.a.signal(ts, data, idx, portfolio) or 0
        b = self.b.signal(ts, data, idx, portfolio) or 0
        if self.mode == "AND":
            if a == 1 and b == 1: return 1
            if a == -1 and b == -1: return -1
            return 0
        # OR
        long_, short_ = (a == 1 or b == 1), (a == -1 or b == -1)
        if long_ and short_: return 0
        if long_: return 1
        if short_: return -1
        return 0


class BuyHoldStrategy(Strategy):
    """買進持有：第一根買滿抱到底，作為比較基準。"""
    def signal(self, ts, data, idx, portfolio):
        return 1 if idx.get(self.sym) is not None else None


STRATEGY_REGISTRY = {
    "均線交叉": MACrossStrategy,
    "通道突破": FuturesBreakoutStrategy,
    "RSI 超買超賣": RSIStrategy,
    "布林通道": BollingerStrategy,
    "MACD": MACDStrategy,
    "KD 隨機指標": KDStrategy,
    "黃金分割": FibonacciStrategy,
    "買進持有": BuyHoldStrategy,
}

# --- backtester/metrics.py ---
from typing import Dict
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _annualization_factor(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return TRADING_DAYS
    days = np.median(np.diff(index.values).astype("timedelta64[D]").astype(float))
    days = max(days, 1.0)
    return 365.0 / days


def compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame,
                    initial_cash: float) -> Dict[str, float]:
    if equity.empty:
        return {}
    eq = equity["equity"].astype(float)
    ruined = bool((eq <= 0).any())
    # 計算報酬時，權益觸及 0 以下會讓百分比統計失真，故以正值下限保護
    eq_safe = eq.clip(lower=1.0)
    rets = eq_safe.pct_change().dropna()
    ann = _annualization_factor(eq.index)

    total_return = eq.iloc[-1] / initial_cash - 1.0
    years = max((eq.index[-1] - eq.index[0]).days / 365.0, 1e-9)
    base = max(eq.iloc[-1], 0.0)
    cagr = (base / initial_cash) ** (1 / years) - 1.0 if base > 0 else -1.0

    vol = rets.std() * np.sqrt(ann) if len(rets) > 1 else 0.0
    sharpe = (rets.mean() * ann) / (rets.std() * np.sqrt(ann)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = (rets.mean() * ann) / (downside.std() * np.sqrt(ann)) if len(downside) > 1 and downside.std() > 0 else 0.0

    roll_max = eq.cummax()
    drawdown = (eq / roll_max - 1.0).clip(lower=-1.0)   # 回撤下限 -100%
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    # 交易層面 (以已實現損益不為 0 的平倉交易計)
    n_trades = win_rate = profit_factor = avg_win = avg_loss = 0.0
    if not trades.empty and "realized_pnl" in trades:
        closed = trades[trades["realized_pnl"] != 0]["realized_pnl"]
        n_trades = int(len(closed))
        if n_trades:
            wins, losses = closed[closed > 0], closed[closed < 0]
            win_rate = len(wins) / n_trades
            gross_win, gross_loss = wins.sum(), -losses.sum()
            profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
            avg_win = wins.mean() if len(wins) else 0.0
            avg_loss = losses.mean() if len(losses) else 0.0

    return {
        "初始資金": initial_cash,
        "期末權益": float(eq.iloc[-1]),
        "總報酬率": total_return,
        "年化報酬(CAGR)": cagr,
        "年化波動": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "最大回撤(MDD)": max_dd,
        "Calmar": calmar,
        "平倉次數": n_trades,
        "勝率": win_rate,
        "獲利因子": profit_factor,
        "平均獲利": avg_win,
        "平均虧損": avg_loss,
        "是否爆倉": ruined,
    }


def format_report(metrics: Dict[str, float]) -> str:
    if not metrics:
        return "（無資料）"
    pct = {"總報酬率", "年化報酬(CAGR)", "年化波動", "最大回撤(MDD)", "勝率"}
    money = {"初始資金", "期末權益", "平均獲利", "平均虧損"}
    lines = ["─" * 40, f"{'回測績效報告':^32}", "─" * 40]
    for k, v in metrics.items():
        if k == "是否爆倉":
            s = "是 ⚠️" if v else "否"
        elif k in pct:
            s = f"{v*100:,.2f}%"
        elif k in money:
            s = f"{v:,.0f}"
        elif k == "平倉次數":
            s = f"{int(v)}"
        else:
            s = f"{v:,.2f}"
        lines.append(f"{k:<16}{s:>22}")
    lines.append("─" * 40)
    return "\n".join(lines)

# --- backtester/engine.py ---
from typing import Dict, List, Optional
import pandas as pd



class Backtest:
    def __init__(self, strategies: List[Strategy], data: Dict[str, pd.DataFrame],
                 initial_cash: float = 1_000_000.0, broker: Optional[Broker] = None,
                 maintenance_ratio: float = 0.75):
        self.strategies = strategies
        self.data = {s: df.copy() for s, df in data.items()}
        self.broker = broker or Broker()
        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.maintenance_ratio = maintenance_ratio   # 維持保證金 / 原始保證金
        self.halted = False                           # 爆倉後停止交易
        for st in strategies:
            self.portfolio.register(st.instrument)
        self._inst = {st.instrument.symbol: st.instrument for st in strategies}

    def _margin_call(self, prices: Dict[str, float], ts) -> bool:
        """權益跌破維持保證金或 <=0 -> 以收盤強制平倉所有部位。"""
        equity = self.portfolio.equity(prices)
        maint = self.portfolio.margin_used() * self.maintenance_ratio
        if self.portfolio.margin_used() > 0 and (equity <= 0 or equity < maint):
            for sym, pos in list(self.portfolio.positions.items()):
                if pos.qty != 0 and sym in prices:
                    self.broker.execute(Order(self._inst[sym], -pos.qty, "強制平倉"),
                                        prices[sym], self.portfolio, ts, force=True)
            self.halted = True
            return True
        return False

    def run(self) -> Dict[str, float]:
        for st in self.strategies:
            st.prepare(self.data)

        # 建立共同時間軸 (所有商品時間戳的聯集)，並建立各商品 ts->iloc 對照
        timeline = sorted(set().union(*[df.index for df in self.data.values()]))
        pos_map = {s: {ts: i for i, ts in enumerate(df.index)} for s, df in self.data.items()}

        pending: List[Order] = []
        for ts in timeline:
            # 1) 撮合上一根送出的訂單，以本根開盤價成交
            still_pending: List[Order] = []
            for order in pending:
                sym = order.instrument.symbol
                i = pos_map[sym].get(ts)
                if i is None:                      # 本商品本根無資料，順延
                    still_pending.append(order)
                    continue
                open_px = float(self.data[sym]["open"].iloc[i])
                self.broker.execute(order, open_px, self.portfolio, ts)
            pending = still_pending

            # 2) 以本根收盤逐日結算
            prices = {s: float(self.data[s]["close"].iloc[pos_map[s][ts]])
                      for s in self.data if ts in pos_map[s]}

            # 2b) 保證金維持檢查：跌破則強制平倉並停止交易
            if not self.halted:
                self._margin_call(prices, ts)

            self.portfolio.record_equity(ts, prices)

            # 3) 策略產生下一批訂單 (爆倉後停止)
            if self.halted:
                pending = []
                continue
            idx = {s: pos_map[s].get(ts) for s in self.data}
            for st in self.strategies:
                pending.extend(st.on_bar(ts, self.data, idx, self.portfolio))

        return compute_metrics(self.portfolio.equity_df(),
                                 self.portfolio.trades_df(),
                                 self.portfolio.initial_cash)

    # 方便取用結果
    def equity_curve(self) -> pd.DataFrame:
        return self.portfolio.equity_df()

    def trades(self) -> pd.DataFrame:
        return self.portfolio.trades_df()

# --- Streamlit 介面 ---

import pandas as pd
import streamlit as st

COMMON = {
    "stock": [("台積電 2330", "2330"), ("鴻海 2317", "2317"), ("聯發科 2454", "2454"),
              ("台達電 2308", "2308"), ("中華電 2412", "2412"), ("長榮 2603", "2603"),
              ("元大台灣50 0050", "0050"), ("元大高股息 0056", "0056")],
    "future": [("台指期 大台 TX", "TX"), ("小型台指 MTX", "MTX")],
}
INTRADAY = {"60分": "60m", "30分": "30m", "15分": "15m"}
RESAMPLE = {"週K": "W", "月K": "ME"}
SIMPLE_STRATS = [n for n in STRATEGY_REGISTRY if n != "買進持有"]


@st.cache_data(show_spinner=False)
def load_data(source, symbol, asset, start, end, token, interval_code, resample_rule,
              syn_periods, syn_s0, syn_sigma):
    if interval_code:                                   # 分鐘K：僅 yfinance
        df = YFinanceFeed().get(symbol, start=start, end=end, interval=interval_code)
    elif source == "FinMind":
        df = FinMindFeed(token=token or None).get(symbol, start=start, end=end, asset=asset)
    elif source == "yfinance":
        df = YFinanceFeed().get(symbol, start=start, end=end)
    else:
        s0 = syn_s0 if asset == "future" else min(syn_s0, 1000)
        df = SyntheticFeed(seed=7).get("SIM", periods=syn_periods, s0=s0, sigma=syn_sigma)
    if resample_rule:
        df = resample_ohlcv(df, resample_rule)
    return df


def build_instrument(cfg):
    if cfg["asset"] == "future":
        return mtx_future(cfg["symbol"]) if cfg["symbol"] == "MTX" else tx_future(cfg["symbol"])
    return tw_stock(cfg["symbol"], fee_discount=cfg["fee_discount"])


def build_strategy(inst, cfg):
    kw = dict(lot_size=cfg["lot_size"], size_pct=cfg["size_pct"], contracts=cfg["contracts"],
              stop_loss=cfg["stop_loss"], take_profit=cfg["take_profit"])
    if cfg["strategy"] == "複合策略":
        A = STRATEGY_REGISTRY[cfg["params"]["a"]](inst)
        B = STRATEGY_REGISTRY[cfg["params"]["b"]](inst)
        return CompositeStrategy(inst, A, B, mode=cfg["params"]["mode"], **kw)
    return STRATEGY_REGISTRY[cfg["strategy"]](inst, **cfg["params"], **kw)


def run_engine(df, inst, strat, cfg):
    eng = Backtest([strat], {inst.symbol: df}, initial_cash=cfg["cash"],
                      broker=Broker(slippage=cfg["slippage"]))
    return eng.run(), eng


def drawdown_series(equity):
    return (equity / equity.cummax() - 1.0).clip(lower=-1.0)


def secret_token():
    try:
        return st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        return ""


st.set_page_config(page_title="台股/台指期 回測面板", page_icon="📈", layout="wide")
st.markdown("""<style>
[data-testid="stMetricValue"]{font-size:1.5rem;font-weight:700;}
[data-testid="stMetricLabel"]{opacity:.75;}
.block-container{padding-top:2.5rem;} h1{font-size:1.7rem;}
</style>""", unsafe_allow_html=True)
st.title("📈 台股 / 台指期 回測")
st.caption("選好條件 → 一鍵回測 → 看績效與走勢圖")

with st.sidebar:
    st.header("⚙️ 控制面板")
    source = st.selectbox("資料來源", ["FinMind", "yfinance", "模擬資料"],
                          help="FinMind 免開戶免費、含真實台指期；分鐘K需選 yfinance")
    token = ""
    if source == "FinMind":
        token = st.text_input("FinMind token（選填）", type="password",
                              help="留空 300 次/小時；填入 600 次/小時。雲端可改在 Secrets 設 FINMIND_TOKEN")
        token = token or secret_token()

    timeframe = st.selectbox("K線週期", ["日K", "週K", "月K", "60分", "30分", "15分"],
                             help="日/週/月穩定；分鐘K僅 yfinance、且只能取近期")
    asset = "future" if st.radio("商品類型", ["股票", "期貨"], horizontal=True) == "期貨" else "stock"

    opts = COMMON[asset]
    labels = [l for l, _ in opts] + ["✏️ 自訂輸入"]
    pick = st.selectbox("標的", labels)
    symbol = st.text_input("輸入代碼", value="TX" if asset == "future" else "2330") if pick == "✏️ 自訂輸入" else dict(opts)[pick]

    # 週期 → 抓取設定
    interval_code, resample_rule = None, None
    if timeframe in INTRADAY:
        if source == "yfinance":
            interval_code = INTRADAY[timeframe]
        else:
            st.warning("分鐘K僅支援 yfinance 資料源，已改用日K。")
    elif timeframe in RESAMPLE:
        resample_rule = RESAMPLE[timeframe]

    if source == "yfinance":
        if asset == "future":
            symbol = "^TWII"
            st.caption("※ yfinance 無台指期連續合約，用加權指數 ^TWII 代理")
        elif "." not in symbol and not symbol.startswith("^"):
            symbol = symbol + ".TW"

    c1, c2 = st.columns(2)
    start = c1.date_input("起始日期", value=pd.Timestamp("2019-01-01")).strftime("%Y-%m-%d")
    end = c2.date_input("結束日期", value=pd.Timestamp.today()).strftime("%Y-%m-%d")
    if interval_code:   # 分鐘K：自動把起始日拉到可抓範圍
        days = 55 if timeframe in ("15分", "30分") else 700
        min_start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        if start < min_start:
            start = min_start
            st.caption(f"※ 分鐘K僅能取近期，起始日已調整為 {start}")

    st.divider()
    names = list(STRATEGY_REGISTRY.keys()) + ["複合策略"]
    default_idx = names.index("通道突破") if asset == "future" else names.index("均線交叉")
    strategy = st.selectbox("策略", names, index=default_idx)

    p = {}
    if strategy == "均線交叉":
        p["fast"] = st.slider("短均線", 3, 60, 20); p["slow"] = st.slider("長均線", 10, 240, 60)
    elif strategy == "通道突破":
        p["lookback"] = st.slider("通道回看天數", 5, 120, 20)
    elif strategy == "RSI 超買超賣":
        p["period"] = st.slider("RSI 天數", 5, 30, 14)
        p["oversold"] = st.slider("超賣門檻(買)", 10, 40, 30)
        p["overbought"] = st.slider("超買門檻(賣)", 60, 90, 70)
    elif strategy == "布林通道":
        p["period"] = st.slider("均線天數", 5, 60, 20); p["k"] = st.slider("標準差倍數", 1.0, 3.0, 2.0, step=0.1)
    elif strategy == "MACD":
        p["fast"] = st.slider("快線 EMA", 5, 20, 12); p["slow"] = st.slider("慢線 EMA", 20, 40, 26)
        p["signal"] = st.slider("訊號線", 5, 15, 9)
    elif strategy == "KD 隨機指標":
        p["period"] = st.slider("KD 天數", 5, 30, 9)
    elif strategy == "黃金分割":
        p["lookback"] = st.slider("區間回看天數", 20, 240, 60)
        fibs = [0.236, 0.382, 0.5, 0.618, 0.786]
        p["buy_fib"] = st.select_slider("買進回檔位(較深)", fibs, value=0.618)
        p["exit_fib"] = st.select_slider("出場回檔位(較淺)", fibs, value=0.382)
    elif strategy == "買進持有":
        st.caption("買進後抱到底，作為比較基準。")
    elif strategy == "複合策略":
        p["a"] = st.selectbox("策略 A", SIMPLE_STRATS, index=SIMPLE_STRATS.index("均線交叉"))
        p["b"] = st.selectbox("策略 B", SIMPLE_STRATS, index=SIMPLE_STRATS.index("MACD"))
        p["mode"] = "AND" if "AND" in st.radio("組合方式", ["兩者都成立(AND)", "任一成立(OR)"],
                                                help="AND 較嚴格、訊號較少；OR 較寬鬆") else "OR"

    with st.expander("🛑 停損 / 停利（選用）"):
        sl = st.slider("停損 %", 0, 50, 0, help="0 = 關閉。跌破進場價這個 % 就出場")
        tp = st.slider("停利 %", 0, 100, 0, help="0 = 關閉。獲利達這個 % 就出場")

    st.divider()
    cfg = {"asset": asset, "symbol": symbol, "strategy": strategy, "params": p,
           "stop_loss": sl / 100.0, "take_profit": tp / 100.0}
    if asset == "future":
        cfg["contracts"] = st.number_input("交易口數", 1, 50, 1); cfg["lot_size"], cfg["size_pct"] = 1000, 0.95
    else:
        cfg["lot_size"] = st.number_input("每筆股數", 1, 100000, 1000, step=1000)
        cfg["size_pct"] = st.slider("資金投入比例", 0.1, 1.0, 0.95); cfg["contracts"] = 1
    cfg["cash"] = st.number_input("初始資金 (NT$)", 100000, 100000000, 1000000, step=100000)
    with st.expander("交易成本 / 進階"):
        cfg["slippage"] = st.slider("滑價 (比例)", 0.0, 0.005, 0.0005, step=0.0001, format="%.4f")
        cfg["fee_discount"] = st.slider("股票手續費折數", 0.1, 1.0, 0.6, help="期貨不適用")
    cfg["syn_periods"], cfg["syn_s0"] = 1000, (18000.0 if asset == "future" else 600.0)
    cfg["syn_sigma"] = 0.18 if asset == "future" else 0.28

    run = st.button("🚀 執行回測", type="primary", use_container_width=True)


if not run:
    st.subheader("三步驟開始")
    s1, s2, s3 = st.columns(3)
    s1.markdown("### 1️⃣\n**打開左側控制面板**\n\n手機請點左上角 **»**")
    s2.markdown("### 2️⃣\n**選條件**\n\n週期、商品、策略、停損停利")
    s3.markdown("### 3️⃣\n**按 🚀 執行回測**\n\n馬上看到結果")
    st.divider()
    st.markdown("結果分三個分頁：**績效**、**走勢圖**（價格＋買賣點、與買進持有比較）、**成交明細**。")
    st.info("💡 新手建議：資料 **FinMind**、週期 **日K**、商品 **股票**、標的 **台積電 2330**，直接按執行。")
    st.stop()


try:
    with st.spinner("抓取資料中…"):
        df = load_data(source, symbol, asset, start, end, token, interval_code, resample_rule,
                       cfg["syn_periods"], cfg["syn_s0"], cfg["syn_sigma"])
except Exception as e:
    st.error(f"資料取得失敗（{type(e).__name__}）：{e}\n\n分鐘K請確認用 yfinance 且日期在近期；無外網請改「模擬資料」。")
    st.stop()

if df is None or df.empty or len(df) < 20:
    st.warning("資料筆數不足（可能是週期太長或分鐘K範圍太短），請調整。")
    st.stop()

with st.spinner("回測運算中…"):
    inst = build_instrument(cfg)
    strat = build_strategy(inst, cfg)
    metrics, eng = run_engine(df, inst, strat, cfg)
    bench = None
    if cfg["strategy"] != "買進持有":
        b_inst = build_instrument(cfg)
        b_strat = BuyHoldStrategy(b_inst, lot_size=cfg["lot_size"], size_pct=cfg["size_pct"],
                                     contracts=cfg["contracts"])
        bench, b_eng = run_engine(df, b_inst, b_strat, cfg)

st.success(f"完成：{source}｜{symbol}｜{timeframe}｜{df.index.min().date()} ~ {df.index.max().date()}｜{len(df)} 根")
tab1, tab2, tab3 = st.tabs(["📊 績效", "📈 走勢圖", "📋 成交明細"])


def pct(x): return f"{x*100:,.2f}%"

with tab1:
    m = metrics
    delta_ret = f"{(m['總報酬率']-bench['總報酬率'])*100:+.1f}% vs 買進持有" if bench else None
    r1 = st.columns(4)
    r1[0].metric("總報酬率", pct(m["總報酬率"]), delta=delta_ret)
    r1[1].metric("年化報酬", pct(m["年化報酬(CAGR)"]))
    r1[2].metric("Sharpe", f"{m['Sharpe']:.2f}")
    r1[3].metric("最大回撤", pct(m["最大回撤(MDD)"]))
    r2 = st.columns(4)
    r2[0].metric("勝率", pct(m["勝率"])); r2[1].metric("獲利因子", f"{m['獲利因子']:.2f}")
    r2[2].metric("平倉次數", f"{int(m['平倉次數'])}"); r2[3].metric("是否爆倉", "是 ⚠️" if m["是否爆倉"] else "否")
    if bench:
        won = m["總報酬率"] > bench["總報酬率"]
        st.markdown(f"策略 **{pct(m['總報酬率'])}** ｜ 買進持有 **{pct(bench['總報酬率'])}** — "
                    + ("✅ 策略勝出" if won else "⚠️ 未贏過單純抱著"))
    with st.expander("完整指標"):
        st.dataframe(pd.DataFrame({"指標": list(m.keys()), "數值": [f"{v}" for v in m.values()]}),
                     hide_index=True, use_container_width=True)

with tab2:
    st.markdown("**價格與買賣點**")
    try:
        import altair as alt
        pdf = df.reset_index(); pdf = pdf.rename(columns={pdf.columns[0]: "datetime"})
        base = alt.Chart(pdf).mark_line(color="#7f8c8d").encode(
            x=alt.X("datetime:T", title=None), y=alt.Y("close:Q", title="價格", scale=alt.Scale(zero=False)))
        layers = [base]
        t = eng.trades()
        if not t.empty:
            t = t.copy(); t["方向"] = t["qty"].apply(lambda q: "買進" if q > 0 else "賣出")
            layers.append(alt.Chart(t).mark_point(size=80, filled=True, opacity=0.9).encode(
                x="datetime:T", y="price:Q",
                color=alt.Color("方向:N", scale=alt.Scale(domain=["買進", "賣出"], range=["#2ca02c", "#d62728"]),
                                legend=alt.Legend(title=None)),
                shape=alt.Shape("方向:N", scale=alt.Scale(domain=["買進", "賣出"], range=["triangle-up", "triangle-down"]),
                                legend=None),
                tooltip=["datetime:T", "方向:N", alt.Tooltip("price:Q", format=",.1f")]))
        st.altair_chart(alt.layer(*layers).interactive(), use_container_width=True)
    except Exception as e:
        st.line_chart(df["close"], height=280); st.caption(f"(買賣點圖略過：{type(e).__name__})")

    st.markdown("**權益曲線 vs 買進持有**")
    comp = pd.DataFrame({"策略": eng.equity_curve()["equity"]})
    if bench is not None:
        comp["買進持有"] = b_eng.equity_curve()["equity"]
    st.line_chart(comp, height=280)
    st.markdown("**回撤**")
    st.area_chart(drawdown_series(eng.equity_curve()["equity"]), height=180, color="#c0392b")

with tab3:
    trades = eng.trades()
    st.markdown(f"共 **{len(trades)}** 筆成交")
    st.dataframe(trades, use_container_width=True, height=360)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ 權益曲線 CSV", eng.equity_curve().to_csv().encode("utf-8-sig"),
                       file_name="equity.csv", mime="text/csv", use_container_width=True)
    d2.download_button("⬇️ 成交明細 CSV", trades.to_csv(index=False).encode("utf-8-sig"),
                       file_name="trades.csv", mime="text/csv", use_container_width=True)

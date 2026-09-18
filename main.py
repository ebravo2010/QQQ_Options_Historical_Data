# main.py

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Configuration
# -----------------------------

TICKERS = ["QQQ"]

MIN_DTE = 7
MAX_DTE = 90

SAVE_CALLS = True
SAVE_PUTS = True

REQUEST_DELAY_SECONDS = 0.5

# GitHub Actions saves files locally before uploading them to Google Drive
DATA_FOLDER = "output"

os.makedirs(DATA_FOLDER, exist_ok=True)

print(f"Data will be saved to: {DATA_FOLDER}")


# -----------------------------
# Helper functions
# -----------------------------

def get_new_york_timestamp():
    return datetime.now(ZoneInfo("America/New_York"))


def get_underlying_price(ticker_obj):
    """
    Try fast_info first, then fall back to recent daily data.
    """
    try:
        price = ticker_obj.fast_info.get("last_price")

        if price is not None and np.isfinite(price):
            return float(price)

    except Exception as error:
        print(f"fast_info price lookup failed: {error}")

    try:
        recent_data = ticker_obj.history(
            period="5d",
            interval="1d",
            auto_adjust=False
        )

        if not recent_data.empty:
            close_prices = recent_data["Close"].dropna()

            if not close_prices.empty:
                return float(close_prices.iloc[-1])

    except Exception as error:
        print(f"Historical price lookup failed: {error}")

    return np.nan


def clean_option_chain(
    option_df,
    ticker_symbol,
    underlying_price,
    expiration_date,
    observation_timestamp,
    option_type
):
    """
    Standardize calls or puts into a consistent format.
    """

    if option_df is None or option_df.empty:
        return pd.DataFrame()

    df = option_df.copy()

    numeric_columns = [
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "change",
        "percentChange",
        "volume",
        "openInterest",
        "impliedVolatility"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )
        else:
            df[column] = np.nan

    if "inTheMoney" not in df.columns:
        df["inTheMoney"] = np.nan

    # Add identifying information
    df["ticker"] = ticker_symbol
    df["option_type"] = option_type
    df["expiration"] = expiration_date
    df["observation_timestamp"] = observation_timestamp
    df["observation_date"] = observation_timestamp.strftime(
        "%Y-%m-%d"
    )
    df["underlying_price"] = underlying_price

    expiration_as_date = datetime.strptime(
        expiration_date,
        "%Y-%m-%d"
    ).date()

    observation_date = observation_timestamp.date()

    df["dte"] = (
        expiration_as_date - observation_date
    ).days

    # Derived quote fields
    df["mid_price"] = (
        df["bid"] + df["ask"]
    ) / 2

    df["spread"] = df["ask"] - df["bid"]

    df["spread_pct"] = np.where(
        df["mid_price"] > 0,
        df["spread"] / df["mid_price"],
        np.nan
    )

    df["moneyness"] = np.where(
        underlying_price > 0,
        df["strike"] / underlying_price,
        np.nan
    )

    df["intrinsic_value"] = np.where(
        option_type == "call",
        np.maximum(underlying_price - df["strike"], 0),
        np.maximum(df["strike"] - underlying_price, 0)
    )

    output_columns = [
        "observation_timestamp",
        "observation_date",
        "ticker",
        "option_type",
        "contractSymbol",
        "expiration",
        "dte",
        "underlying_price",
        "strike",
        "moneyness",
        "lastTradeDate",
        "lastPrice",
        "bid",
        "ask",
        "mid_price",
        "spread",
        "spread_pct",
        "volume",
        "openInterest",
        "impliedVolatility",
        "inTheMoney",
        "intrinsic_value",
        "change",
        "percentChange"
    ]

    output_columns = [
        column for column in output_columns
        if column in df.columns
    ]

    return df[output_columns]


def collect_option_snapshot(
    tickers,
    min_dte,
    max_dte
):
    """
    Download one options snapshot for each ticker.
    """

    observation_timestamp = get_new_york_timestamp()
    observation_date = observation_timestamp.date()

    all_rows = []

    for ticker_symbol in tickers:
        print(f"\nProcessing {ticker_symbol}...")

        try:
            ticker_obj = yf.Ticker(ticker_symbol)

            underlying_price = get_underlying_price(ticker_obj)

            if not np.isfinite(underlying_price):
                print(
                    f"Could not retrieve price for {ticker_symbol}"
                )
                continue

            print(
                f"Underlying price: ${underlying_price:.2f}"
            )

            expiration_dates = ticker_obj.options

            if not expiration_dates:
                print(
                    f"No option expirations found for "
                    f"{ticker_symbol}"
                )
                continue

            selected_expirations = []

            for expiration in expiration_dates:
                expiration_date = datetime.strptime(
                    expiration,
                    "%Y-%m-%d"
                ).date()

                dte = (
                    expiration_date - observation_date
                ).days

                if min_dte <= dte <= max_dte:
                    selected_expirations.append(expiration)

            print(
                f"Selected expirations: "
                f"{len(selected_expirations)}"
            )

            for expiration in selected_expirations:
                try:
                    option_chain = ticker_obj.option_chain(
                        expiration
                    )

                    if SAVE_CALLS:
                        calls = clean_option_chain(
                            option_chain.calls,
                            ticker_symbol,
                            underlying_price,
                            expiration,
                            observation_timestamp,
                            "call"
                        )

                        if not calls.empty:
                            all_rows.append(calls)

                    if SAVE_PUTS:
                        puts = clean_option_chain(
                            option_chain.puts,
                            ticker_symbol,
                            underlying_price,
                            expiration,
                            observation_timestamp,
                            "put"
                        )

                        if not puts.empty:
                            all_rows.append(puts)

                    print(
                        f"  Processed expiration: {expiration}"
                    )

                    time.sleep(REQUEST_DELAY_SECONDS)

                except Exception as error:
                    print(
                        f"  Error for {ticker_symbol} "
                        f"{expiration}: {error}"
                    )

        except Exception as error:
            print(
                f"Error processing {ticker_symbol}: {error}"
            )

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(
        all_rows,
        ignore_index=True
    )


# -----------------------------
# Main program
# -----------------------------

def main():
    print("\nStarting options collection...")

    snapshot_df = collect_option_snapshot(
        tickers=TICKERS,
        min_dte=MIN_DTE,
        max_dte=MAX_DTE
    )

    if snapshot_df.empty:
        print("\nNo data was collected.")
        return

    timestamp = get_new_york_timestamp()

    file_timestamp = timestamp.strftime(
        "%Y%m%d_%H%M%S"
    )

    snapshot_file = os.path.join(
        DATA_FOLDER,
        f"options_snapshot_{file_timestamp}.parquet"
    )

    snapshot_df.to_parquet(
        snapshot_file,
        index=False,
        compression="snappy"
    )

    print(
        f"\nSaved {len(snapshot_df):,} rows to:"
    )
    print(snapshot_file)

    print("\nPreview:")
    print(snapshot_df.head())


if __name__ == "__main__":
    main()

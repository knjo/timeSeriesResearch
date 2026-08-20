from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from statistics import stdev

import polars as pl

from src.dataloader.txfDataLoader import (
    TXF_ENTRY_TOLERANCE,
    TxfDataLoader,
    attach_txf_to_1330_return,
)
from src.features.market_beta import (
    TXF_EVENT_RESIDUAL_VOL_COLUMN,
    TXF_REMAIN_SECONDS_COLUMN,
    TXF_RESIDUAL_VOL_COLUMN,
    VOL_TIME_SCALE_COEFFICIENTS,
    MarketBetaConfig,
    add_txf_beta_features,
    add_txf_event_residual_vol,
    residual_vol_time_scale,
)


def _txf_frame(date: str, open_mid: float, close_mid: float) -> pl.DataFrame:
    day = datetime.strptime(date, "%Y%m%d")
    times = [
        day + timedelta(hours=8, minutes=50),
        day + timedelta(hours=9),
        day + timedelta(hours=9, seconds=29),
        day + timedelta(hours=13, minutes=29, seconds=59),
        day + timedelta(hours=13, minutes=31),
    ]
    mids = [open_mid, open_mid, open_mid * 1.001, close_mid, close_mid * 1.10]
    return pl.DataFrame(
        {
            "TransTime": times,
            "QuoteCode": ["TXFA"] * len(times),
            "TrialMatch": [0] * len(times),
            "FillLots": [10, 20, 10, 30, 1],
            "BidPrice1": [(value - 0.5) * 100.0 for value in mids],
            "AskPrice1": [(value + 0.5) * 100.0 for value in mids],
            "BidPrice2": [(value - 1.0) * 100.0 for value in mids],
            "AskPrice2": [(value + 1.0) * 100.0 for value in mids],
        }
    )


class TxfDataLoaderTest(unittest.TestCase):
    def test_sdk_fallback_and_event_to_close_return(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loader = TxfDataLoader(
                Path(tmp),
                sdk_loader=lambda date: _txf_frame(date, 20_000.0, 20_200.0),
            )
            daily = loader.daily_open_to_close_return("20240102")
            self.assertIsNotNone(daily)
            assert daily is not None
            self.assertEqual(daily.source, "sdk_injected")
            self.assertAlmostEqual(daily.return_value, 0.01)

            events = pl.DataFrame(
                {
                    "TransTime": [datetime(2024, 1, 2, 9, 0, 30)],
                    "QuoteCode": ["2330"],
                    "ChannelSeq": [1],
                }
            )
            result = attach_txf_to_1330_return(events, "20240102", loader)
            self.assertEqual(result["txf_contract"][0], "TXFA")
            self.assertEqual(result["txf_source"][0], "sdk_injected")
            expected = 20_200.0 / (20_000.0 * 1.001) - 1.0
            self.assertAlmostEqual(result["txf_to_1330_return"][0], expected)
            self.assertGreaterEqual(result["txf_entry_quote_age_us"][0], 0)

    def test_entry_tolerance_default_covers_stale_trade_prints(self) -> None:
        # Backfilled days carry trades, which go quiet while quotes would keep
        # refreshing. The last fixture print before this event is 09:00:29, so
        # the event is 5s stale: resolvable under the 10s default, dropped
        # under the old 2s window.
        with tempfile.TemporaryDirectory() as tmp:
            loader = TxfDataLoader(
                Path(tmp),
                sdk_loader=lambda date: _txf_frame(date, 20_000.0, 20_200.0),
            )
            events = pl.DataFrame(
                {
                    "TransTime": [datetime(2024, 1, 2, 9, 0, 34)],
                    "QuoteCode": ["2330"],
                    "ChannelSeq": [1],
                }
            )
            self.assertEqual(TXF_ENTRY_TOLERANCE, "10s")

            default = attach_txf_to_1330_return(events, "20240102", loader)
            self.assertIsNotNone(default["txf_to_1330_return"][0])
            self.assertEqual(default["txf_entry_quote_age_us"][0], 5_000_000)

            strict = attach_txf_to_1330_return(
                events, "20240102", loader, entry_tolerance="2s"
            )
            self.assertIsNone(strict["txf_to_1330_return"][0])

    def test_premarket_beta_uses_history_through_source_date(self) -> None:
        factors = [-0.02, -0.01, 0.01, 0.02]
        residuals = [-0.002, 0.002, 0.002, -0.002]
        dates = ["20240102", "20240103", "20240104", "20240105"]
        txf_by_date = {
            date: _txf_frame(date, 20_000.0, 20_000.0 * (1.0 + factor))
            for date, factor in zip(dates, factors, strict=True)
        }
        txf_by_date["20240108"] = _txf_frame("20240108", 20_000.0, 20_800.0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            market_dir = root / "marketData"
            market_dir.mkdir()
            for date, factor, residual in zip(
                dates, factors, residuals, strict=True
            ):
                pl.DataFrame(
                    {
                        "quote_code": ["2330"],
                        "open_price": [100.0],
                        "close_price": [
                            100.0 * (1.0 + 1.5 * factor + residual)
                        ],
                    }
                ).write_parquet(market_dir / f"{date}_marketData.parquet")
            pl.DataFrame(
                {
                    "quote_code": ["2330"],
                    "open_price": [100.0],
                    "close_price": [80.0],
                }
            ).write_parquet(market_dir / "20240108_marketData.parquet")

            loader = TxfDataLoader(
                root / "missing_local_ticks",
                sdk_loader=lambda date: txf_by_date[date],
            )
            result = add_txf_beta_features(
                pl.DataFrame({"QuoteCode": ["2330", "9999"]}),
                "20240105",
                market_dir,
                loader,
                MarketBetaConfig(
                    lookback_days=4,
                    min_observations=2,
                    prior_strength=0.0,
                ),
            ).sort("QuoteCode")
            known = result.filter(pl.col("QuoteCode") == "2330")
            fallback = result.filter(pl.col("QuoteCode") == "9999")
            self.assertAlmostEqual(known["txf_beta_60d"][0], 1.5)
            self.assertAlmostEqual(
                known[TXF_RESIDUAL_VOL_COLUMN][0],
                stdev(residuals),
            )
            self.assertAlmostEqual(fallback["txf_beta_60d"][0], 1.0)
            self.assertIsNone(fallback[TXF_RESIDUAL_VOL_COLUMN][0])
            self.assertNotIn("txf_beta_raw_60d", result.columns)

    def test_premarket_beta_drops_locked_stock_days(self) -> None:
        # A stock locked all session (high == low) has a mechanically zero
        # open-to-close return. Pairing that with a +3% factor day would drag
        # the estimate from 1.5 down to ~0.87, so the observation must be
        # dropped rather than the whole session blacklisted.
        factors = [-0.02, -0.01, 0.01, 0.02]
        dates = ["20240102", "20240103", "20240104", "20240105"]
        txf_by_date = {
            date: _txf_frame(date, 20_000.0, 20_000.0 * (1.0 + factor))
            for date, factor in zip(dates, factors, strict=True)
        }
        txf_by_date["20240108"] = _txf_frame("20240108", 20_000.0, 20_600.0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            market_dir = root / "marketData"
            market_dir.mkdir()
            for date, factor in zip(dates, factors, strict=True):
                close = 100.0 * (1.0 + 1.5 * factor)
                pl.DataFrame(
                    {
                        "quote_code": ["2330"],
                        "open_price": [100.0],
                        "close_price": [close],
                        "high_price": [max(100.0, close) + 1.0],
                        "low_price": [min(100.0, close) - 1.0],
                    }
                ).write_parquet(market_dir / f"{date}_marketData.parquet")
            # Locked session: open == close == high == low.
            pl.DataFrame(
                {
                    "quote_code": ["2330"],
                    "open_price": [100.0],
                    "close_price": [100.0],
                    "high_price": [100.0],
                    "low_price": [100.0],
                }
            ).write_parquet(market_dir / "20240108_marketData.parquet")

            loader = TxfDataLoader(
                root / "missing_local_ticks",
                sdk_loader=lambda date: txf_by_date[date],
            )
            result = add_txf_beta_features(
                pl.DataFrame({"QuoteCode": ["2330"]}),
                "20240108",
                market_dir,
                loader,
                MarketBetaConfig(
                    lookback_days=5,
                    min_observations=2,
                    prior_strength=0.0,
                ),
            )
            self.assertAlmostEqual(result["txf_beta_60d"][0], 1.5)

    def test_vol_time_scale_matches_calibrated_anchors(self) -> None:
        # Coefficients sum to 1.0 so the 09:00 anchor is exact by construction.
        self.assertAlmostEqual(sum(VOL_TIME_SCALE_COEFFICIENTS), 1.0, places=7)
        self.assertAlmostEqual(residual_vol_time_scale(1.0), 1.0, places=7)
        self.assertAlmostEqual(residual_vol_time_scale(0.0), 0.0, places=7)

        # Anchors from the 2024-calibrated / 2026-validated intraday vol clock.
        for clock, expected in [
            ((9, 15), 0.908),
            ((10, 0), 0.696),
            ((11, 0), 0.532),
            ((11, 30), 0.485),
            ((12, 0), 0.447),
            ((12, 30), 0.403),
            ((13, 0), 0.323),
        ]:
            remaining = (
                datetime(2024, 1, 2, 13, 30) - datetime(2024, 1, 2, *clock)
            ).total_seconds()
            self.assertAlmostEqual(
                residual_vol_time_scale(remaining / 16_200.0),
                expected,
                places=3,
                msg=f"vol clock mismatch at {clock[0]:02d}:{clock[1]:02d}",
            )

    def test_vol_time_scale_is_strictly_increasing(self) -> None:
        values = [residual_vol_time_scale(i / 500.0) for i in range(501)]
        for previous, current in zip(values, values[1:]):
            self.assertGreater(current, previous)

    def test_vol_time_scale_is_below_sqrt_over_calibrated_range(self) -> None:
        # The point of the recalibration: plain sqrt overstates remaining vol
        # across the calibrated 09:00-13:00 range. x = 1/9 is 13:00.
        for i in range(1, 900):
            x = 1.0 / 9.0 + (1.0 - 1.0 / 9.0) * i / 900.0
            self.assertLess(residual_vol_time_scale(x), x**0.5)

    def test_vol_time_scale_crosses_above_sqrt_only_near_the_close(self) -> None:
        # Past the last calibration anchor the fit turns back above sqrt at
        # 13:06:49. Events stop at 12:00 (x >= 1/3), so this is never reached
        # in production; the test pins it so the extrapolation stays visible.
        # x shrinks as the clock advances, so x below the crossover is the
        # late-session side where the fit sits above sqrt.
        crossover = 0.085853
        self.assertGreater(
            residual_vol_time_scale(crossover - 0.001),
            (crossover - 0.001) ** 0.5,
        )
        self.assertLess(
            residual_vol_time_scale(crossover + 0.001),
            (crossover + 0.001) ** 0.5,
        )
        earliest_event_x = (13.5 - 12.0) / 4.5
        self.assertGreater(earliest_event_x, crossover)

    def test_event_residual_vol_uses_ushape_time_to_1330(self) -> None:
        base_vol = 0.02
        events = pl.DataFrame(
            {
                "TransTime": [
                    datetime(2024, 1, 2, 9, 0),
                    datetime(2024, 1, 2, 10, 0),
                    datetime(2024, 1, 2, 13, 30),
                ],
                TXF_RESIDUAL_VOL_COLUMN: [base_vol] * 3,
            }
        )
        result = add_txf_event_residual_vol(events)
        self.assertAlmostEqual(result[TXF_EVENT_RESIDUAL_VOL_COLUMN][0], base_vol)
        self.assertAlmostEqual(
            result[TXF_EVENT_RESIDUAL_VOL_COLUMN][1],
            base_vol * residual_vol_time_scale(3.5 / 4.5),
        )
        self.assertAlmostEqual(result[TXF_EVENT_RESIDUAL_VOL_COLUMN][2], 0.0)

    def test_event_residual_vol_clips_outside_session(self) -> None:
        base_vol = 0.02
        events = pl.DataFrame(
            {
                "TransTime": [
                    datetime(2024, 1, 2, 8, 30),
                    datetime(2024, 1, 2, 14, 0),
                ],
                TXF_RESIDUAL_VOL_COLUMN: [base_vol] * 2,
            }
        )
        result = add_txf_event_residual_vol(events)
        self.assertAlmostEqual(result[TXF_EVENT_RESIDUAL_VOL_COLUMN][0], base_vol)
        self.assertAlmostEqual(result[TXF_EVENT_RESIDUAL_VOL_COLUMN][1], 0.0)
        self.assertAlmostEqual(result[TXF_REMAIN_SECONDS_COLUMN][0], 16_200.0)
        self.assertAlmostEqual(result[TXF_REMAIN_SECONDS_COLUMN][1], 0.0)

    def test_remain_seconds_is_emitted_for_downstream_recompute(self) -> None:
        events = pl.DataFrame(
            {
                "TransTime": [
                    datetime(2024, 1, 2, 9, 0),
                    datetime(2024, 1, 2, 10, 0),
                    datetime(2024, 1, 2, 13, 30),
                ],
                TXF_RESIDUAL_VOL_COLUMN: [0.02] * 3,
            }
        )
        result = add_txf_event_residual_vol(events)
        remain = result[TXF_REMAIN_SECONDS_COLUMN].to_list()
        self.assertEqual(remain, [16_200.0, 12_600.0, 0.0])
        # The emitted horizon must reproduce the emitted vol exactly, so a
        # downstream def can rebuild the scaling without drifting.
        for seconds, vol in zip(remain, result[TXF_EVENT_RESIDUAL_VOL_COLUMN]):
            self.assertAlmostEqual(
                vol, 0.02 * residual_vol_time_scale(seconds / 16_200.0)
            )

    def test_remain_seconds_present_even_without_premarket_vol(self) -> None:
        events = pl.DataFrame({"TransTime": [datetime(2024, 1, 2, 11, 0)]})
        result = add_txf_event_residual_vol(events)
        self.assertAlmostEqual(result[TXF_REMAIN_SECONDS_COLUMN][0], 9_000.0)
        self.assertIsNone(result[TXF_EVENT_RESIDUAL_VOL_COLUMN][0])


if __name__ == "__main__":
    unittest.main()

"""OHLC candlestick drawing utilities extracted from ML Final Project.ipynb.

This module provides the DrawOHLC class used to generate grayscale OHLC
+ volume + moving-average images for CNN training.
"""

import numpy as np
from PIL import Image, ImageDraw

import config

# Candlestick / volume drawing parameters
BAR_WIDTH: int = 3
VOLUME_HEIGHT_RATIO: float = 0.2
VOLUME_CHART_GAP: int = 2
BACKGROUND_COLOR: int = 0


class DrawOHLC:

    def __init__(
        self,
        df,
        time_frame,
        has_volume_bar=True,
        has_moving_average=True,
        has_bollinger_bands=False,
        has_vwap=False,
        has_obv=False,
        has_rsi=False,
        has_adx=False,
    ):

        self.df = df.copy()
        self.ohlc_len = len(df)
        self.time_frame = time_frame
        self.has_volume_bar = has_volume_bar
        self.has_moving_average = has_moving_average
        self.has_bollinger_bands = has_bollinger_bands
        self.has_vwap = has_vwap
        self.has_obv = has_obv
        self.has_rsi = has_rsi
        self.has_adx = has_adx

        # Image dimensions
        self.image_width = config.IMAGE_WIDTH
        self.image_height = config.IMAGE_HEIGHT

    def __value_to_yaxis(self, val: float) -> int:
        """Convert price value to y-axis pixel position, clipped to image bounds."""

        pixels_per_unit = (self.image_height - 1.0) / (self.maxp - self.minp)
        pixel = int(np.around((val - self.minp) * pixels_per_unit))
        # Clip to image bounds to prevent drawing outside
        return max(0, min(self.image_height - 1, pixel))

    def draw_image(self) -> Image.Image:
        """Draw OHLC candlestick chart with volume bars and MA line.
        
        Return: numpy array normalized to [0, 1]
        """

        first_center = (BAR_WIDTH - 1) / 2.0
        self.centers = np.arange(
            first_center,
            first_center + BAR_WIDTH * self.ohlc_len,
            BAR_WIDTH,
            dtype=int,
        )

        # Background is BLACK (0), candlesticks are WHITE (255)
        image = Image.new("L", (self.image_width, self.image_height), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(image)

        # Calculate min/max including all indicators to prevent cutoff
        price_cols = ["Open", "High", "Low", "Close"]

        # Add Bollinger Bands to min/max calculation if enabled
        if self.has_bollinger_bands and "BB_UPPER" in self.df.columns and "BB_LOWER" in self.df.columns:
            price_cols.extend(["BB_UPPER", "BB_LOWER"])

        # Add VWAP to min/max calculation if enabled
        if self.has_vwap and "VWAP" in self.df.columns:
            price_cols.append("VWAP")

        # Add MA to min/max calculation if enabled
        if self.has_moving_average and "MA" in self.df.columns:
            price_cols.append("MA")

        self.minp = self.df[price_cols].min().min()
        self.maxp = self.df[price_cols].max().max()

        # Add padding to ensure everything fits (5% margin for safety)
        price_range = self.maxp - self.minp
        padding = price_range * 0.05
        self.minp -= padding
        self.maxp += padding

        # Step 1: Draw OHLC chart (price movements)
        for day in range(self.ohlc_len):
            highp, lowp, closep, openp = self.df.iloc[day][
                ["High", "Low", "Close", "Open"]
            ]

            high_pixel = self.__value_to_yaxis(highp)
            low_pixel = self.__value_to_yaxis(lowp)
            open_pixel = self.__value_to_yaxis(openp)
            close_pixel = self.__value_to_yaxis(closep)

            draw.line(
                [self.centers[day], high_pixel, self.centers[day], low_pixel],
                fill=250,
            )
            draw.line(
                [self.centers[day] - 1, open_pixel, self.centers[day], open_pixel],
                fill=251,
            )
            draw.line(
                [self.centers[day], close_pixel, self.centers[day] + 1, close_pixel],
                fill=252,
            )

        # Step 2: Draw volume bars
        if self.has_volume_bar:
            max_volume = self.df["Volume"].max() 
            volume_height = int(self.image_height * VOLUME_HEIGHT_RATIO)

            volume_bottom = self.image_height - 1
            volume_top_limit = volume_bottom - volume_height  # kept for parity

            for day in range(self.ohlc_len):
                volume = self.df.iloc[day]["Volume"]

                volume_pixel = int((volume / max_volume) * volume_height)
                volume_top = volume_bottom - volume_pixel
                draw.line(
                    [self.centers[day], volume_bottom, self.centers[day], volume_top],
                    fill=253,
                )

        # Step 3: Draw moving average line
        if self.has_moving_average:
            for day in range(self.ohlc_len - 1):
                ma_start = self.__value_to_yaxis(self.df["MA"].iloc[day])
                ma_end = self.__value_to_yaxis(self.df["MA"].iloc[day + 1])

                if ma_start >= 0 and ma_end >= 0:
                    draw.line(
                        [
                            self.centers[day],
                            ma_start,
                            self.centers[day + 1],
                            ma_end,
                        ],
                        fill=254,
                    )

        # Step 4: Draw Bollinger Bands (upper and lower)
        if self.has_bollinger_bands and "BB_UPPER" in self.df.columns and "BB_LOWER" in self.df.columns:
            for day in range(self.ohlc_len - 1):
                # Check for NaN values before drawing
                bb_upper_start = self.df["BB_UPPER"].iloc[day]
                bb_upper_end = self.df["BB_UPPER"].iloc[day + 1]
                bb_lower_start = self.df["BB_LOWER"].iloc[day]
                bb_lower_end = self.df["BB_LOWER"].iloc[day + 1]

                # Draw upper band if values are valid
                if not (np.isnan(bb_upper_start) or np.isnan(bb_upper_end)):
                    upper_start = self.__value_to_yaxis(bb_upper_start)
                    upper_end = self.__value_to_yaxis(bb_upper_end)
                    draw.line(
                        [
                            self.centers[day],
                            upper_start,
                            self.centers[day + 1],
                            upper_end,
                        ],
                        fill=240,
                    )

                # Draw lower band if values are valid
                if not (np.isnan(bb_lower_start) or np.isnan(bb_lower_end)):
                    lower_start = self.__value_to_yaxis(bb_lower_start)
                    lower_end = self.__value_to_yaxis(bb_lower_end)
                    draw.line(
                        [
                            self.centers[day],
                            lower_start,
                            self.centers[day + 1],
                            lower_end,
                        ],
                        fill=240,
                    )

        # Step 5: Draw VWAP line
        if self.has_vwap and "VWAP" in self.df.columns:
            for day in range(self.ohlc_len - 1):
                vwap_start_val = self.df["VWAP"].iloc[day]
                vwap_end_val = self.df["VWAP"].iloc[day + 1]

                if not (np.isnan(vwap_start_val) or np.isnan(vwap_end_val)):
                    vwap_start = self.__value_to_yaxis(vwap_start_val)
                    vwap_end = self.__value_to_yaxis(vwap_end_val)
                    draw.line(
                        [
                            self.centers[day],
                            vwap_start,
                            self.centers[day + 1],
                            vwap_end,
                        ],
                        fill=245,
                    )

        # Step 6: Draw OBV line (normalized to price range)
        if self.has_obv and "OBV" in self.df.columns:
            # Normalize OBV to fit in the price range
            obv_values = self.df["OBV"].values
            obv_min = obv_values.min()
            obv_max = obv_values.max()

            if obv_max > obv_min:  # Avoid division by zero
                for day in range(self.ohlc_len - 1):
                    obv_start_val = self.df["OBV"].iloc[day]
                    obv_end_val = self.df["OBV"].iloc[day + 1]

                    if not (np.isnan(obv_start_val) or np.isnan(obv_end_val)):
                        # Normalize OBV to [minp, maxp] range
                        obv_start_norm = self.minp + (obv_start_val - obv_min) / (obv_max - obv_min) * (self.maxp - self.minp)
                        obv_end_norm = self.minp + (obv_end_val - obv_min) / (obv_max - obv_min) * (self.maxp - self.minp)

                        obv_start = self.__value_to_yaxis(obv_start_norm)
                        obv_end = self.__value_to_yaxis(obv_end_norm)
                        draw.line(
                            [
                                self.centers[day],
                                obv_start,
                                self.centers[day + 1],
                                obv_end,
                            ],
                            fill=235,
                        )

        # Step 7: Draw RSI line (normalized to price range)
        if self.has_rsi and "RSI" in self.df.columns:
            # RSI is already in [0, 100] range, normalize to [minp, maxp]
            for day in range(self.ohlc_len - 1):
                rsi_start_val = self.df["RSI"].iloc[day]
                rsi_end_val = self.df["RSI"].iloc[day + 1]

                if not (np.isnan(rsi_start_val) or np.isnan(rsi_end_val)):
                    # Normalize RSI from [0, 100] to [minp, maxp] range
                    rsi_start_norm = self.minp + (rsi_start_val / 100.0) * (self.maxp - self.minp)
                    rsi_end_norm = self.minp + (rsi_end_val / 100.0) * (self.maxp - self.minp)

                    rsi_start = self.__value_to_yaxis(rsi_start_norm)
                    rsi_end = self.__value_to_yaxis(rsi_end_norm)
                    draw.line(
                        [
                            self.centers[day],
                            rsi_start,
                            self.centers[day + 1],
                            rsi_end,
                        ],
                        fill=230,
                    )

        # Step 8: Draw ADX line (normalized to price range)
        if self.has_adx and "ADX" in self.df.columns:
            # ADX is typically in [0, 100] range (though usually 0-50)
            # Normalize to [minp, maxp] range
            for day in range(self.ohlc_len - 1):
                adx_start_val = self.df["ADX"].iloc[day]
                adx_end_val = self.df["ADX"].iloc[day + 1]

                if not (np.isnan(adx_start_val) or np.isnan(adx_end_val)):
                    # Normalize ADX from [0, 100] to [minp, maxp] range
                    adx_start_norm = self.minp + (adx_start_val / 100.0) * (self.maxp - self.minp)
                    adx_end_norm = self.minp + (adx_end_val / 100.0) * (self.maxp - self.minp)

                    adx_start = self.__value_to_yaxis(adx_start_norm)
                    adx_end = self.__value_to_yaxis(adx_end_norm)
                    draw.line(
                        [
                            self.centers[day],
                            adx_start,
                            self.centers[day + 1],
                            adx_end,
                        ],
                        fill=225,
                    )

        # Return as float32 array with channel dimension (H, W, 1)
        arr = np.array(image, dtype=np.float32)
        return arr[..., np.newaxis]

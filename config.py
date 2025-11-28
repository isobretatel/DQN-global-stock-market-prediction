# Data fetching configuration
N_DAYS: int = 32
FORECAST_HORIZON: int = 1

# Image dimensions
IMAGE_HEIGHT: int = 32
IMAGE_WIDTH: int = 32

# Feature toggles for OHLC chart images
HAS_VOLUME_BAR: bool = False
HAS_MOVING_AVERAGE: bool = True
HAS_BOLLINGER_BANDS: bool = False
HAS_VWAP: bool = False
HAS_OBV: bool = False
HAS_RSI: bool = False
HAS_ADX: bool = False

# Indicator parameters
RSI_PERIOD: int = 14
ADX_PERIOD: int = 14

# Visualization
SHOW_BUILD_SAMPLES_DF: bool = False
SHOW_SAMPLE_IMAGE: bool = False
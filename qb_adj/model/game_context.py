from utils.s_curve import s_curve
from typing import Optional
import pandas as pd

class GameContext:
    def __init__(self, game_id: str, config: dict, temp: Optional[int] = None, wind: Optional[int] = None):
        self.game_id = game_id
        self.config = config
        self.temp = temp
        self.wind = wind

    def weather_adj(self) -> float:
        """
        Calculate the negative adjustment due to wind and temperature.
        Returns:
            float: Total weather adjustment
        """
        wind_mod = max(0, min(30, self.wind - 5 if not pd.isnull(self.wind) else 0))
        temp_mod = max(0, self.temp if not pd.isnull(self.temp) else 70)

        wind_adj = s_curve(
            self.config['wind_disc_height'],
            self.config['wind_disc_mp'],
            wind_mod,
            direction='up'
        )

        temp_adj = s_curve(
            self.config['temp_disc_height'],
            self.config['temp_disc_mp'],
            temp_mod,
            direction='down'
        )

        return temp_adj + wind_adj

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import math


class WeatherDatasetGenerator:
    def __init__(self, start_date="2024-04-19", days_back=365):
        """Initialize the weather dataset generator.

        Args:
            start_date: The end date for the dataset (formatted as YYYY-MM-DD)
            days_back: Number of days to generate data for (backward from start_date)
        """
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.days_back = days_back

        # North India weather patterns
        self.seasonal_temp_patterns = {
            # Month: (min_temp, max_temp)
            1: (5, 20),  # January
            2: (7, 26),  # February
            3: (15, 36),  # March
            4: (22, 42),  # April
            5: (24, 44),  # May
            6: (25, 44),  # June
            7: (25, 38),  # July (monsoon)
            8: (24, 35),  # August (monsoon)
            9: (22, 34),  # September
            10: (16, 32),  # October
            11: (10, 27),  # November
            12: (6, 22),  # December
        }

        # Season-based rainfall probability
        self.monsoon_months = [6, 7, 8, 9]  # Jun-Sep: monsoon season
        self.winter_rain_months = [12, 1, 2]  # Winter rainfall (fewer but some)

        # AQI patterns (North India has severe air quality issues especially in winter)
        self.seasonal_aqi_patterns = {
            # Month: (min_aqi, max_aqi)
            1: (150, 350),  # January (poor AQI)
            2: (120, 300),  # February
            3: (100, 250),  # March
            4: (80, 200),  # April
            5: (70, 170),  # May
            6: (60, 150),  # June
            7: (50, 130),  # July (better in monsoon)
            8: (50, 130),  # August
            9: (70, 160),  # September
            10: (100, 250),  # October (post-harvest stubble burning)
            11: (150, 400),  # November (worst - Diwali + stubble burning)
            12: (150, 350),  # December
        }

    def generate_dataset(self, readings_per_day=10):
        """Generate a synthetic weather dataset for North India.

        Args:
            readings_per_day: Number of readings to generate per day

        Returns:
            DataFrame: Generated weather dataset
        """
        data = []

        # Generate data for each day
        for day in range(self.days_back):
            current_date = self.start_date - timedelta(days=self.days_back - day)
            month = current_date.month

            # Get temperature range for this month
            min_temp, max_temp = self.seasonal_temp_patterns[month]

            # Get base AQI range for this month
            min_aqi, max_aqi = self.seasonal_aqi_patterns[month]

            # Calculate rain probability based on season
            if month in self.monsoon_months:
                rain_prob = 0.7  # 70% chance during monsoon
            elif month in self.winter_rain_months:
                rain_prob = 0.2  # 20% chance during winter
            else:
                rain_prob = 0.1  # 10% otherwise

            # Daily randomization factors
            daily_temp_offset = random.uniform(-3, 3)  # Daily temperature variation
            is_rainy_day = random.random() < rain_prob  # Determine if it's a rainy day

            # Daily AQI variation factor
            daily_aqi_factor = random.uniform(0.8, 1.2)

            # Generate multiple readings throughout the day
            for reading_num in range(readings_per_day):
                # Calculate time of day (hour)
                hour = int((24 / readings_per_day) * reading_num)
                timestamp = current_date.replace(hour=hour, minute=random.randint(0, 59))

                # Temperature follows diurnal pattern (lowest at early morning, highest in afternoon)
                hour_factor = self._diurnal_temperature_factor(hour)
                temp_range = max_temp - min_temp
                temperature = min_temp + temp_range * hour_factor + daily_temp_offset

                # Calculate humidity (inverse relationship with temperature)
                base_humidity = self._calculate_base_humidity(month)
                humidity = base_humidity - (hour_factor * 20) + random.uniform(-5, 5)
                humidity = max(30, min(95, humidity))  # Clamp between 30-95%

                # Pressure calculation (typical range 970-1030 hPa, seasonal variations)
                base_pressure = 1013.0  # Standard atmospheric pressure
                pressure = base_pressure + self._calculate_pressure_offset(month, hour) + random.uniform(-2, 2)

                # Rain calculation
                if is_rainy_day:
                    # Rain is more likely in certain hours (afternoon/evening in monsoon)
                    hour_rain_prob = self._calculate_hourly_rain_probability(hour, month)
                    rain_detected = random.random() < hour_rain_prob
                    rain_analog = 100 - random.randint(30, 90) if rain_detected else random.randint(90, 100)
                else:
                    rain_detected = False
                    rain_analog = random.randint(95, 100)  # High value means no rain

                # Light intensity (based on hour and weather)
                light_intensity = self._calculate_light_intensity(hour, rain_detected)

                # AQI calculation
                hour_aqi_factor = self._calculate_hourly_aqi_factor(hour)
                aqi = int((min_aqi + (max_aqi - min_aqi) * hour_aqi_factor) * daily_aqi_factor)

                # Gas concentrations (correlate with AQI)
                aqi_ratio = (aqi - 50) / 300 if aqi > 50 else 0  # Normalized AQI factor
                nh3 = max(0.1, 1 + 4 * aqi_ratio + random.uniform(-0.5, 0.5))
                co = max(0.2, 2 + 8 * aqi_ratio + random.uniform(-1.0, 1.0))
                co2 = max(0.1, 0.3 + 0.5 * aqi_ratio + random.uniform(-0.1, 0.1))
                alcohol = max(0.1, 0.5 + 1.5 * aqi_ratio + random.uniform(-0.3, 0.3))
                lpg = max(10, 200 + 150 * aqi_ratio + random.uniform(-20, 20))
                ch4 = max(10, lpg * 1.0 + random.uniform(-5, 5))  # Similar to LPG
                mq9_co = max(10, 100 + 80 * aqi_ratio + random.uniform(-10, 10))

                # Create record with randomly split temperature between sensors
                temp_split = random.uniform(-0.5, 0.5)  # Random difference between sensors
                temp_dht = temperature + temp_split
                temp_bmp = temperature - temp_split

                record = {
                    'datetime': timestamp.isoformat(),
                    'timestamp': int(timestamp.timestamp()),
                    'temp_dht': round(temp_dht, 1),
                    'humidity': round(humidity, 0),
                    'temp_bmp': round(temp_bmp, 1),
                    'pressure': round(pressure, 1),
                    'rain_analog': round(rain_analog, 0),
                    'rain_detected': rain_detected,
                    'light_intensity': round(light_intensity, 0),
                    'nh3': round(nh3, 1),
                    'co': round(co, 1),
                    'co2': round(co2, 1),
                    'alcohol': round(alcohol, 1),
                    'lpg': round(lpg, 1),
                    'ch4': round(ch4, 1),
                    'mq9_co': round(mq9_co, 0),
                    'aqi': round(aqi, 0),
                    # Derived features
                    'temperature': round((temp_dht + temp_bmp) / 2, 1)
                }

                data.append(record)

        # Create DataFrame and sort by timestamp
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp')

        return df

    def _diurnal_temperature_factor(self, hour):
        """Calculate temperature factor based on hour of day (0-1 scale).
        Lowest at early morning (5-6AM), highest in afternoon (2-3PM)
        """
        if 0 <= hour < 6:
            # Decreasing from midnight to 6am
            return 0.3 - (hour / 20)
        elif 6 <= hour < 14:
            # Increasing from 6am to 2pm
            return 0.1 + ((hour - 6) / 8) * 0.8
        else:
            # Decreasing from 2pm to midnight
            return 0.9 - ((hour - 14) / 10) * 0.6

    def _calculate_base_humidity(self, month):
        """Calculate base humidity based on month."""
        # Monsoon months have higher humidity
        if month in self.monsoon_months:
            return random.uniform(65, 85)
        # Winter months have moderate humidity
        elif month in self.winter_rain_months:
            return random.uniform(55, 75)
        # Summer months have lower humidity
        elif month in [4, 5]:
            return random.uniform(30, 50)
        else:
            return random.uniform(45, 65)

    def _calculate_pressure_offset(self, month, hour):
        """Calculate pressure offset based on month and hour."""
        # Slight diurnal variation in pressure
        hour_factor = math.sin(hour * math.pi / 12) * 2

        # Seasonal variation (higher pressure in winter, lower in summer/monsoon)
        if month in [12, 1, 2]:  # Winter
            seasonal_offset = random.uniform(3, 8)
        elif month in [3, 4, 5]:  # Summer
            seasonal_offset = random.uniform(-5, 0)
        elif month in self.monsoon_months:  # Monsoon
            seasonal_offset = random.uniform(-8, -2)
        else:  # Transition
            seasonal_offset = random.uniform(-3, 3)

        return seasonal_offset + hour_factor

    def _calculate_hourly_rain_probability(self, hour, month):
        """Calculate probability of rain based on hour and month."""
        if month in self.monsoon_months:
            # Monsoon: higher chance of afternoon/evening rain
            if 13 <= hour <= 19:
                return 0.7
            elif 9 <= hour < 13:
                return 0.5
            elif 19 < hour <= 22:
                return 0.6
            else:
                return 0.3
        else:
            # Non-monsoon: more uniform distribution with slight afternoon bias
            if 14 <= hour <= 18:
                return 0.6
            else:
                return 0.4

    def _calculate_light_intensity(self, hour, is_raining):
        """Calculate light intensity based on hour and weather."""
        # Nighttime (8pm to 6am)
        if hour < 6 or hour >= 20:
            base_intensity = 0
        # Early morning/evening (6-7am, 6-8pm)
        elif (6 <= hour < 7) or (18 <= hour < 20):
            base_intensity = 50 + (hour - 6) * 50 if hour < 7 else 150 - (hour - 18) * 75
        # Day time (7am to 6pm)
        else:
            # Peak at noon
            time_to_noon = abs(12 - hour)
            base_intensity = 400 - (time_to_noon * 30)

        # Reduce intensity if raining
        if is_raining:
            base_intensity *= random.uniform(0.3, 0.7)

        # Add some randomness
        base_intensity += random.uniform(-20, 20)

        # Ensure within bounds
        return max(0, min(500, base_intensity))

    def _calculate_hourly_aqi_factor(self, hour):
        """Calculate AQI factor based on hour of day."""
        # AQI tends to be worse in morning and evening
        if 7 <= hour <= 10:  # Morning rush hour
            return random.uniform(0.7, 1.0)
        elif 16 <= hour <= 20:  # Evening rush hour
            return random.uniform(0.8, 1.0)
        elif 0 <= hour < 5:  # Early morning (pollution settles)
            return random.uniform(0.5, 0.8)
        else:  # Midday (better dispersion)
            return random.uniform(0.4, 0.7)
from datetime import datetime, timedelta
import geocoder
from geopy.geocoders import Nominatim
from shapely.geometry import Point
import holidays
import requests
 

def get_location():
    # --- Get current location from IP --- 
    g = geocoder.ip('me') 
    latitude, longitude = g.latlng 
    # Optionally get city name from coordinates 
    geolocator = Nominatim(user_agent="city_locator") 
    location_info = geolocator.reverse((latitude, longitude), language='en') 
    city = location_info.raw.get('address', {}).get('city', 'Unknown') 
    # Create Point object if needed 
    location = Point(latitude, longitude)
    return city, latitude, longitude

def today_and_tomorrow_status():
    # Current datetime
    now = datetime.now()
    today = now.date()
    tomorrow = today + timedelta(days=1)
    weekday_name = today.strftime("%A")

    # Holidays
    try:
        country_holidays = holidays.CountryHoliday(country_code)
    except:
        country_holidays = []

    # Helper functions
    def is_weekend(date):
        return date.weekday() >= 5  # Saturday=5, Sunday=6

    def is_holiday(date):
        return date in country_holidays

    # --- Conditions ---
    today_status = "Weekday"
    if is_holiday(today):
        today_status = "Holiday"
    elif is_weekend(today):
        today_status = "Weekend"

    tomorrow_status = "Weekday"
    if is_holiday(tomorrow):
        tomorrow_status = "Holiday"
    elif is_weekend(tomorrow):
        tomorrow_status = "Weekend"

    return today_status, tomorrow_status, weekday_name

def get_weather():
    api_key = "8626216750fcb3381f91b31c71b0862e"
    city, latitude, longitude = get_location()
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        return weather_description, temperature
    else:
        print("Failed to get weather:", response.text)
        return None, None

print(get_weather())
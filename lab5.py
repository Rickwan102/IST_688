import streamlit as st
import requests

# Function: get current weather
def get_current_weather(location: str, API_key: str):
    if "," in location:
        location = location.split(",")[0].strip()

    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_key}"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200 or "main" not in data:
        return {"error": f"Could not fetch weather for {location}"}

    temp = data['main']['temp'] - 273.15
    feels_like = data['main']['feels_like'] - 273.15
    temp_min = data['main']['temp_min'] - 273.15
    temp_max = data['main']['temp_max'] - 273.15
    humidity = data['main']['humidity']

    return {
        "location": location,
        "temperature": round(temp, 2),
        "feels_like": round(feels_like, 2),
        "temp_min": round(temp_min, 2),
        "temp_max": round(temp_max, 2),
        "humidity": humidity
    }

# Main entrypoint for Lab 5
def run():
    st.title("Lab 5: Weather App")

    location = st.text_input("Enter a city:", "Syracuse")

    if st.button("Get Weather"):
        API_key = st.secrets["OPENWEATHER_API_KEY"]
        weather = get_current_weather(location, API_key)

        if "error" in weather:
            st.error(weather["error"])
        else:
            st.write(f"**Weather in {weather['location']}**")
            st.write(f"Temperature: {weather['temperature']} °C")
            st.write(f"Feels Like: {weather['feels_like']} °C")
            st.write(f"High: {weather['temp_max']} °C")
            st.write(f"Low: {weather['temp_min']} °C")
            st.write(f"Humidity: {weather['humidity']} %")


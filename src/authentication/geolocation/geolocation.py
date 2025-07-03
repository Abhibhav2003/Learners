import requests

from haversine_distance import haversine

ip_address = requests.get('http://api.ipify.org').text

response = requests.get(f'http://ip-api.com/json/{ip_address}?/fields=192511').json()

if not response.get('proxy'):
    # Don't Grant Access
    pass
else:
    # Access Granted
    # Teacher's Coordinates : lat1,lon1
    # Students's Coordinates : lat2,lon2
    
    # For Example :
    # Teacher's Coordinates : (28.6139,77.2090)
    lat1 = 28.6139
    lon1 = 77.2090

    # Student's Coordinates : (28.6141,77.2092)
    lat2 = 28.6141
    lon2 = 77.2092

    distance = haversine(lat1, lon1, lat2, lon2)

    if distance <= 15 :
        pass
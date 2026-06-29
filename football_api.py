# football_api.py
import requests
import os
import random

class FootballDataAPI:
    def __init__(self):
        self.api_key = os.getenv('RAPIDAPI_KEY')
        self.base_url = 'https://v3.football.api-sports.io'
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        }
        print(f"🔑 API Key loaded: {self.api_key[:10]}...")  # Show first 10 chars
        
    def make_request(self, endpoint, params=None):
        """Make API request to API-Football"""
        try:
            url = f"{self.base_url}/{endpoint}"
            print(f"🌐 Making API request to: {endpoint}")
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            # Show rate limit info
            if 'x-ratelimit-requests-remaining' in response.headers:
                remaining = response.headers['x-ratelimit-requests-remaining']
                print(f"📊 API Requests Remaining Today: {remaining}/100")
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('errors'):
                    error_msg = data['errors'][0] if data['errors'] else 'Unknown error'
                    print(f"❌ API Error: {error_msg}")
                    return {'success': False, 'error': error_msg}
                
                if data.get('response'):
                    print("✅ API request successful!")
                    return {'success': True, 'data': data['response']}
                else:
                    print("❌ No data in response")
                    return {'success': False, 'error': 'No data received'}
                    
            else:
                print(f"❌ HTTP Error {response.status_code}")
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_premier_league_teams(self, season="2023"):
        """Get Premier League teams from API-Football"""
        endpoint = 'teams'
        params = {
            'league': 39,  # Premier League ID
            'season': season
        }
        
        return self.make_request(endpoint, params)

def get_enhanced_static_teams():
    """Enhanced static team data for fallback"""
    teams_data = []
    teams_info = {
        'Arsenal': {'founded': 1886, 'stadium': 'Emirates Stadium', 'capacity': 60704, 'trophies': 13},
        'Aston Villa': {'founded': 1874, 'stadium': 'Villa Park', 'capacity': 42657, 'trophies': 7},
        'Bournemouth': {'founded': 1899, 'stadium': 'Vitality Stadium', 'capacity': 11307, 'trophies': 0},
        'Brentford': {'founded': 1889, 'stadium': 'Gtech Community Stadium', 'capacity': 17250, 'trophies': 0},
        'Brighton': {'founded': 1901, 'stadium': 'Amex Stadium', 'capacity': 31876, 'trophies': 0},
        'Burnley': {'founded': 1882, 'stadium': 'Turf Moor', 'capacity': 21944, 'trophies': 2},
        'Chelsea': {'founded': 1905, 'stadium': 'Stamford Bridge', 'capacity': 40341, 'trophies': 25},
        'Crystal Palace': {'founded': 1905, 'stadium': 'Selhurst Park', 'capacity': 25486, 'trophies': 0},
        'Everton': {'founded': 1878, 'stadium': 'Goodison Park', 'capacity': 39414, 'trophies': 9},
        'Fulham': {'founded': 1879, 'stadium': 'Craven Cottage', 'capacity': 22384, 'trophies': 0},
        'Liverpool': {'founded': 1892, 'stadium': 'Anfield', 'capacity': 53394, 'trophies': 48},
        'Luton': {'founded': 1885, 'stadium': 'Kenilworth Road', 'capacity': 10356, 'trophies': 0},
        'Man City': {'founded': 1880, 'stadium': 'Etihad Stadium', 'capacity': 53400, 'trophies': 28},
        'Man United': {'founded': 1878, 'stadium': 'Old Trafford', 'capacity': 74879, 'trophies': 66},
        'Newcastle': {'founded': 1892, 'stadium': "St James' Park", 'capacity': 52305, 'trophies': 11},
        'Nottingham Forest': {'founded': 1865, 'stadium': 'City Ground', 'capacity': 30445, 'trophies': 6},
        'Sheffield United': {'founded': 1889, 'stadium': 'Bramall Lane', 'capacity': 32050, 'trophies': 1},
        'Tottenham': {'founded': 1882, 'stadium': 'Tottenham Hotspur Stadium', 'capacity': 62850, 'trophies': 17},
        'West Ham': {'founded': 1895, 'stadium': 'London Stadium', 'capacity': 62500, 'trophies': 5},
        'Wolves': {'founded': 1877, 'stadium': 'Molineux Stadium', 'capacity': 31750, 'trophies': 13}
    }
    
    for team_name, info in teams_info.items():
        teams_data.append({
            'name': team_name,
            'founded': info['founded'],
            'stadium': info['stadium'],
            'capacity': info['capacity'],
            'trophies': info['trophies'],
            'logo': f'https://media.api-sports.io/football/teams/{random.randint(1, 100)}.png'
        })
    
    return teams_data
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import google.generativeai as genai
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔍 Checking environment variables...")
print("RAPIDAPI_KEY exists:", bool(os.getenv('RAPIDAPI_KEY')))
print("RAPIDAPI_HOST:", os.getenv('RAPIDAPI_HOST'))
print("GEMINI_API_KEY exists:", bool(os.getenv('GEMINI_API_KEY')))

# Import the football API
from football_api import FootballDataAPI, get_enhanced_static_teams
football_api = FootballDataAPI()
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'epl_predictor_interactive_2024')

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///epl_predictor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure Gemini AI
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    try:
        genai.configure(api_key=api_key)
        print(f"✅ Gemini AI configured successfully with key length: {len(api_key)}")
        print(f"🔑 Key starts with: {api_key[:10]}...")
    except Exception as e:
        print(f"❌ Gemini configuration failed: {e}")
else:
    print("❌ GEMINI_API_KEY not found in environment variables")

# Initialize Football API
football_api = FootballDataAPI()

# Updated DEFAULT_TEAMS - Current Premier League 2023/24 teams
DEFAULT_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Liverpool", "Luton", "Man City", "Man United", "Newcastle",
    "Nottingham Forest", "Sheffield United", "Tottenham", "West Ham", "Wolves"
]

# Load your existing FTR model and encoders
try:
    with open('models/rf_epl_model.pkl', 'rb') as f:
        detailed_model = pickle.load(f)
    
    with open('models/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    detailed_ftr_model_loaded = True
    print("✅ Detailed FTR Model and encoders loaded successfully!")
    
    # Get unique team names from encoders - try different key formats
    try:
        home_teams = sorted(encoders['HomeTeam'].classes_.tolist())
        away_teams = sorted(encoders['AwayTeam'].classes_.tolist())
        seasons = sorted(encoders['Season'].classes_.tolist())
    except:
        home_teams = sorted(encoders['home'].classes_.tolist())
        away_teams = sorted(encoders['away'].classes_.tolist())
        seasons = sorted(encoders['season'].classes_.tolist())
    
    print(f"✅ Loaded {len(home_teams)} teams for detailed FTR prediction")
    
except Exception as e:
    detailed_ftr_model_loaded = False
    home_teams = away_teams = DEFAULT_TEAMS
    seasons = ['2023-2024', '2022-2023', '2021-2022']
    print(f"❌ Error loading detailed FTR model: {e}")
    print("🔄 Using default teams list")

# Load the new SIMPLE FTR and Score prediction models
try:
    print("🔄 Loading simple prediction models...")
    from prediction_engine import MatchPredictor
    simple_predictor = MatchPredictor()
    simple_models_loaded = True
    simple_teams = simple_predictor.get_available_teams()
    print("✅ Simple FTR and Score prediction models loaded successfully!")
    print(f"✅ Loaded {len(simple_teams)} teams for simple predictions")
    
except Exception as e:
    simple_models_loaded = False
    simple_teams = home_teams if home_teams else DEFAULT_TEAMS
    print(f"❌ Error loading simple FTR/Score models: {e}")
    print("🔄 Using detailed model teams for simple predictions")

def predict_ftr_detailed(user_input):
    """
    Predict FTR using detailed match statistics (your existing model)
    """
    try:
        # Extract input values
        home_team = user_input['home_team']
        away_team = user_input['away_team']
        season = user_input['season']
        
        # Match statistics
        hthg = float(user_input['hthg'])
        htag = float(user_input['htag'])
        hs = float(user_input['hs'])
        as_ = float(user_input['as'])
        hst = float(user_input['hst'])
        ast = float(user_input['ast'])
        hc = float(user_input['hc'])
        ac = float(user_input['ac'])
        hf = float(user_input['hf'])
        af = float(user_input['af'])
        hy = float(user_input['hy'])
        ay = float(user_input['ay'])
        hr = float(user_input['hr'])
        ar = float(user_input['ar'])
        
        # --- FEATURE ENGINEERING ---
        goal_diff_HT = hthg - htag
        shots_diff = hs - as_
        shots_on_target_diff = hst - ast
        corners_diff = hc - ac
        fouls_diff = hf - af
        yellow_diff = hy - ay
        red_diff = hr - ar
        shot_accuracy_home = hst / (hs + 1)
        shot_accuracy_away = ast / (as_ + 1)
        corner_ratio = (hc + 1) / (ac + 1)
        discipline_ratio = (hf + 1) / (af + 1)
        card_diff = (hy + hr) - (ay + ar)
        
        # Encode categorical variables
        try:
            home_team_enc = encoders['home'].transform([home_team])[0]
            away_team_enc = encoders['away'].transform([away_team])[0]
            season_enc = encoders['season'].transform([season])[0]
        except:
            try:
                home_team_enc = encoders['HomeTeam'].transform([home_team])[0]
                away_team_enc = encoders['AwayTeam'].transform([away_team])[0]
                season_enc = encoders['Season'].transform([season])[0]
            except:
                home_team_enc = away_team_enc = season_enc = 0
        
        # Create feature array
        feature_array = np.array([[
            home_team_enc, away_team_enc, season_enc,
            hthg, htag, hs, as_, hst, ast, hc, ac,
            hf, af, hy, ay, hr, ar,
            goal_diff_HT, shots_diff, shots_on_target_diff,
            corners_diff, fouls_diff, yellow_diff, red_diff,
            shot_accuracy_home, shot_accuracy_away,
            corner_ratio, discipline_ratio, card_diff
        ]])
        
        # Make prediction
        prediction = detailed_model.predict(feature_array)[0]
        probabilities = detailed_model.predict_proba(feature_array)[0]
        
        # Decode prediction
        result_mapping = {0: 'A', 1: 'D', 2: 'H'}
        predicted_result = result_mapping[prediction]
        
        # Get probabilities
        prob_home = probabilities[2]
        prob_draw = probabilities[1]
        prob_away = probabilities[0]
        
        return {
            'prediction': predicted_result,
            'probabilities': {
                'home_win': round(prob_home * 100, 2),
                'draw': round(prob_draw * 100, 2),
                'away_win': round(prob_away * 100, 2)
            },
            'success': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def update_prediction_count(user_id):
    """Simple function to update user's prediction count"""
    try:
        user = User.query.get(user_id)
        user.predictions_made += 1
        db.session.commit()
        print(f"✅ Updated prediction count for user {user_id}: {user.predictions_made}")
        return True
    except Exception as e:
        print(f"❌ Error updating prediction count: {e}")
        db.session.rollback()
        return False

# Initialize extensions
from models import db, User, Prediction
db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ========== ROUTES ==========

@app.route('/')
@login_required
def index():
    """Home page with navigation"""
    return render_template('index.html')

@app.route('/detailed-ftr')
@login_required
def detailed_ftr():
    """Detailed FTR prediction interface"""
    return render_template('ftr_stats_interactive.html', 
                         home_teams=home_teams, 
                         away_teams=away_teams,
                         seasons=seasons,
                         model_loaded=detailed_ftr_model_loaded)

@app.route('/simple-ftr')
@login_required
def simple_ftr():
    """Simple FTR prediction interface"""
    return render_template('ftr_prediction.html', 
                         teams=simple_teams, 
                         models_loaded=simple_models_loaded)

@app.route('/score-prediction')
@login_required
def score_prediction():
    """Score prediction interface"""
    return render_template('score_prediction.html', 
                         teams=simple_teams, 
                         models_loaded=simple_models_loaded)

# ========== PREDICTION ENDPOINTS ==========

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Detailed FTR prediction endpoint"""
    if not detailed_ftr_model_loaded:
        return jsonify({'success': False, 'error': 'Detailed FTR Model not loaded'})
    
    try:
        user_input = {
            'home_team': request.form.get('home_team', ''),
            'away_team': request.form.get('away_team', ''),
            'season': request.form.get('season', '2023-2024'),
            'hthg': request.form.get('hthg', '0'),
            'htag': request.form.get('htag', '0'),
            'hs': request.form.get('hs', '0'),
            'as': request.form.get('as', '0'),
            'hst': request.form.get('hst', '0'),
            'ast': request.form.get('ast', '0'),
            'hc': request.form.get('hc', '0'),
            'ac': request.form.get('ac', '0'),
            'hf': request.form.get('hf', '0'),
            'af': request.form.get('af', '0'),
            'hy': request.form.get('hy', '0'),
            'ay': request.form.get('ay', '0'),
            'hr': request.form.get('hr', '0'),
            'ar': request.form.get('ar', '0')
        }
        
        if not user_input['home_team'] or not user_input['away_team']:
            return jsonify({'success': False, 'error': 'Please select both home and away teams'})
        
        prediction_result = predict_ftr_detailed(user_input)
        
        # Update prediction count
        if prediction_result['success']:
            update_prediction_count(current_user.id)
        
        return jsonify(prediction_result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict-simple-ftr', methods=['POST'])
@login_required
def predict_simple_ftr():
    """Simple FTR prediction endpoint"""
    if not simple_models_loaded:
        return jsonify({'success': False, 'error': 'Simple FTR prediction models not loaded'})
    
    try:
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        date = request.form['date']
        
        result = simple_predictor.predict_ftr(home_team, away_team, date)
        
        # Update prediction count
        if result['success']:
            update_prediction_count(current_user.id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict-score', methods=['POST'])
@login_required
def predict_score_route():
    """Score prediction endpoint"""
    if not simple_models_loaded:
        return jsonify({'success': False, 'error': 'Score prediction models not loaded'})
    
    try:
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        date = request.form['date']
        
        result = simple_predictor.predict_score(home_team, away_team, date)
        
        # Update prediction count
        if result['success']:
            update_prediction_count(current_user.id)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ========== API ENDPOINTS ==========

@app.route('/api/teams/detailed')
def get_detailed_teams():
    return jsonify({'teams': home_teams, 'success': True})

@app.route('/api/teams/simple')
def get_simple_teams():
    return jsonify({'teams': simple_teams, 'success': simple_models_loaded})

# ========== NEW ENHANCEMENT ROUTES (SAFE ADDITIONS) ==========

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        favorite_team = request.form.get('favorite_team', '')
        
        user_exists = User.query.filter_by(username=username).first()
        email_exists = User.query.filter_by(email=email).first()
        
        if user_exists:
            flash('Username already exists!', 'error')
            return render_template('register.html', teams=home_teams)
        if email_exists:
            flash('Email already registered!', 'error')
            return render_template('register.html', teams=home_teams)
        
        hashed_password = generate_password_hash(password)
        new_user = User(
            username=username,
            email=email,
            password=hashed_password,
            favorite_team=favorite_team
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', teams=home_teams)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check username and password.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_stats = {
        'total_predictions': current_user.predictions_made,
        'favorite_team': current_user.favorite_team
    }
    
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id)\
        .order_by(Prediction.created_at.desc()).limit(3).all()
    
    return render_template('dashboard.html', 
                         user_stats=user_stats, 
                         recent_predictions=recent_predictions,
                         teams=home_teams)

@app.route('/epl-info')
@login_required
def epl_info():
    """EPL Information Page"""
    epl_facts = {
        'founded': 1992,
        'teams': 20,
        'current_champion': 'Manchester City',
        'most_titles': 'Manchester United (13)',
        'top_scorer': 'Alan Shearer (260 goals)',
        'season_structure': 'August to May',
        'matches_per_season': 380,
        'most_goals_season': '34 - Andy Cole (1993/94), Alan Shearer (1994/95)',
        'biggest_win': '9-0 (Manchester United vs Ipswich, 1995)',
        'most_points': '100 - Manchester City (2017/18)'
    }
    
    return render_template('epl_info.html', facts=epl_facts)

@app.route('/teams-display')
@login_required
def teams_display():
    """Interactive Teams Display Page"""
    return render_template('teams_display.html')

# ========== API ENDPOINTS FOR LIVE DATA ==========

@app.route('/api/football/teams')
@login_required
def api_football_teams():
    """API endpoint for real football teams data"""
    try:
        # Try to get real data from API
        result = football_api.get_premier_league_teams("2023")
        
        print("🔍 API Result Success:", result['success'])
        
        if result['success']:
            teams_data = []
            for team_data in result['data']:
                team = team_data['team']
                venue = team_data.get('venue', {})
                
                teams_data.append({
                    'id': team['id'],
                    'name': team['name'],
                    'code': team.get('code', ''),
                    'country': team.get('country', 'England'),
                    'founded': team.get('founded', 1880),
                    'logo': team.get('logo', ''),
                    'stadium': venue.get('name', f"{team['name']} Stadium"),
                    'stadium_capacity': venue.get('capacity', 40000),
                    'city': venue.get('city', ''),
                    'trophies': random.randint(0, 20)
                    # Removed 'color' field
                })
            
            return jsonify({'success': True, 'teams': teams_data, 'source': 'api-football'})
        else:
            print(f"❌ API failed: {result.get('error', 'Unknown error')}")
            # Fallback to enhanced static data
            return jsonify({'success': True, 'teams': get_enhanced_static_teams(), 'source': 'static'})
            
    except Exception as e:
        print(f"❌ Teams API exception: {e}")
        return jsonify({'success': True, 'teams': get_enhanced_static_teams(), 'source': 'error'})


@app.route('/test-api')
def test_api():
    """Test if API is working"""
    try:
        result = football_api.get_premier_league_teams("2023")
        return jsonify({
            'api_working': result['success'],
            'data_received': result.get('teams', [])[:1] if result['success'] else None,
            'error': result.get('error', 'No error')
        })
    except Exception as e:
        return jsonify({'api_working': False, 'error': str(e)})

# ========== CHATBOT ROUTES ==========

@app.route('/api/chat/debug')
def chat_debug():
    """Debug endpoint to check chatbot setup"""
    api_key = os.getenv('GEMINI_API_KEY')
    
    # Check if Gemini is properly configured
    gemini_configured = False
    try:
        # Try to list models to verify configuration
        genai.list_models()
        gemini_configured = True
    except:
        gemini_configured = False
    
    return jsonify({
        'api_key_exists': bool(api_key),
        'api_key_length': len(api_key) if api_key else 0,
        'gemini_configured': gemini_configured,
        'status': 'OK' if gemini_configured else 'NOT CONFIGURED'
    })

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    try:
        # Check if API key is configured
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'Gemini API key not configured'
            }), 500
        
        # Get user message
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data received'
            }), 400
            
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        print(f"🤖 User message: {user_message}")
        
        # Use the stable Gemini 2.0 Flash model that we know exists
        model_name = 'models/gemini-2.0-flash-001'
        
        try:
            print(f"🔄 Using model: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            # Create football context
            context = """You are an expert Premier League football prediction assistant. 
            Help users with match predictions, team analysis, player stats, and betting insights.
            Keep responses concise and football-focused (1-2 paragraphs max). Be helpful and informative.
            If asked about non-EPL topics, politely redirect to Premier League topics.
            
            Current Premier League context (2024/25 season):
            - Defending champions: Manchester City
            - Top teams: Man City, Arsenal, Liverpool, Chelsea, Tottenham
            - Key players: Haaland, Salah, De Bruyne, Saka, Son
            - Format: 20 teams, 38 matches each"""
            
            full_prompt = f"{context}\n\nUser: {user_message}"
            
            # Generate response
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=500,
                )
            )
            
            print(f"✅ Chatbot response generated successfully!")
            print(f"🤖 Response: {response.text}")
            
            return jsonify({
                'success': True,
                'response': response.text,
                'model_used': model_name
            })
            
        except Exception as model_error:
            print(f"❌ Model {model_name} failed: {str(model_error)}")
            # Fallback to simple response
            return jsonify({
                'success': True,
                'response': get_fallback_response(user_message),
                'fallback': True
            })
        
    except Exception as e:
        print(f"❌ Chatbot error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to process your message. Please try again.'
        }), 500

def get_fallback_response(message):
    """Enhanced fallback responses"""
    message_lower = message.lower()
    
    football_responses = {
        'hi': "Hi! I'm your EPL assistant. Ask me about teams, predictions, or match analysis! ⚽",
        'hello': "Hello! Ready to talk Premier League football? What would you like to know?",
        'hii': "Hey there! I'm here to help with EPL predictions and team insights. What's on your mind?",
        'help': "I can help with:\n• Match predictions\n• Team analysis\n• Player stats\n• Betting insights\n• EPL information",
        'prediction': "For detailed match predictions, check out our 'Simple FTR Prediction' or 'Score Prediction' tools in the navigation!",
        'man city': "🏆 Manchester City - Current champions! Known for possession football under Pep Guardiola. Key players: Haaland, De Bruyne, Foden.",
        'liverpool': "🔴 Liverpool - 19-time champions! Famous for high-press football at Anfield. Key players: Salah, Van Dijk, Alexander-Arnold.",
        'arsenal': "🔴 Arsenal - Plays at Emirates Stadium. Rivalry with Tottenham. Known for attractive football. Key player: Saka, Odegaard.",
        'chelsea': "🔵 Chelsea - Based at Stamford Bridge. Won multiple Champions League titles. Building a young squad.",
        'man united': "🔴 Manchester United - Record 13 Premier League titles. Plays at Old Trafford. Historic club in transition.",
        'tottenham': "⚪ Tottenham Hotspur - Plays at Tottenham Hotspur Stadium. Known as Spurs. Key player: Son Heung-min, Maddison.",
        'neymar': "Neymar is a Brazilian superstar who never played in the Premier League! He's known for his time at Barcelona and PSG. Want to know about Premier League Brazilian players like Alisson, Gabriel Jesus, or Richarlison?",
        'messi': "Lionel Messi never played in the Premier League! He made his mark at Barcelona and PSG. Interested in Premier League legends like Thierry Henry, Alan Shearer, or Wayne Rooney?",
        'ronaldo': "Cristiano Ronaldo had an amazing Premier League career with Manchester United! He won 3 titles and his first Ballon d'Or there before moving to Real Madrid. Later returned for a second stint.",
        'epl': "⚽ English Premier League - Top 20 teams, 380 matches per season. Founded in 1992. Current champions: Manchester City. Most titles: Manchester United (13).",
        'premier league': "The Premier League is the most-watched football league in the world! 20 teams compete from August to May. Known for its fast pace and competitive nature.",
        'top 4': "This season's top 4 race is competitive! Manchester City, Arsenal, and Liverpool look strong for Champions League spots, with Aston Villa and Tottenham also in contention.",
        'relegation': "The relegation battle is always intense! Newly promoted teams often struggle. The bottom three teams get relegated to the Championship.",
        'goal scorer': "Current top scorers include Erling Haaland (Man City), Mohamed Salah (Liverpool), and Son Heung-min (Tottenham). The Golden Boot race is always exciting!",
        'assist': "Top assist makers are often creative players like Kevin De Bruyne (Man City), Mohamed Salah (Liverpool), and Bukayo Saka (Arsenal).",
        'table': "The Premier League table shows team positions based on points. 3 points for a win, 1 for a draw. Goal difference breaks ties.",
        'fixture': "Upcoming fixtures determine the title race and relegation battles. The schedule is packed from August to May!",
        'transfer': "Transfer windows: Summer (June-September) and Winter (January). Big money moves often happen between Premier League clubs.",
        'thanks': "You're welcome! Come back anytime for more Premier League insights! ⚽",
        'thank you': "Happy to help! Let me know if you need more football knowledge!",
        'bye': "Goodbye! Hope to see you again for more football talk! 🏆"
    }
    
    # Check for exact matches first
    if message_lower in football_responses:
        return football_responses[message_lower]
    
    # Check for partial matches
    for key, response in football_responses.items():
        if key in message_lower:
            return response
    
    return "I specialize in Premier League football! Try asking about teams, match predictions, player stats, or current EPL news. For detailed predictions, use our prediction tools in the navigation menu above! ⚽"
@app.route('/api/chat/available-models')
def available_models():
    """See what models are actually available"""
    try:
        models = genai.list_models()
        available = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available.append({
                    'name': model.name,
                    'description': model.description
                })
        return jsonify({'success': True, 'models': available})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
# Update the main block
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    print("🚀 Starting EPL Predictor Pro...")
    print("📍 Available Routes:")
    print("   - Home:              http://127.0.0.1:5000/")
    print("   - Login:             http://127.0.0.1:5000/login")
    print("   - Register:          http://127.0.0.1:5000/register")
    print("   - Dashboard:         http://127.0.0.1:5000/dashboard")
    print("   - EPL Info:          http://127.0.0.1:5000/epl-info")
    print("   - Teams Display:     http://127.0.0.1:5000/teams-display")
    print("   - Detailed FTR:      http://127.0.0.1:5000/detailed-ftr")
    print("   - Simple FTR:        http://127.0.0.1:5000/simple-ftr")
    print("   - Score Prediction:  http://127.0.0.1:5000/score-prediction")
    print("   - Test API:          http://127.0.0.1:5000/test-api")
    print("   - Chatbot Debug:     http://127.0.0.1:5000/api/chat/debug")
    
    app.run(debug=True, port=5000)
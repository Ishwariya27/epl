import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os

class MatchPredictor:
    def __init__(self, data_path='epl_cleaned_dataset.csv'):
        self.data_path = data_path
        self.load_models()
        self.eps = 1e-6
        
        # EXACT features from your notebook
        self.ftr_feature_list = [
            'Season_enc', 'Month', 'DayOfWeek', 'Late_Season',
            'Home_Last5GoalsFor', 'Home_Last5GoalsAgainst', 'Away_Last5GoalsFor', 'Away_Last5GoalsAgainst',
            'Home_Last5Points', 'Away_Last5Points', 'Home_AttackStrength', 'Away_AttackStrength',
            'Home_DefenseStrength', 'Away_DefenseStrength', 'FormDiff_Pts', 'Attack_vs_Defense_H',
            'Attack_vs_Defense_A', 'Home_Last3Points_Sum', 'Away_Last3Points_Sum', 'Home_Momentum',
            'Away_Momentum', 'H2H_HomeGoalsAvg', 'H2H_AwayGoalsAvg', 'H2H_HomeWinRate', 'HomeTeam_enc',
            'AwayTeam_enc', 'HTR_enc'
        ]
        
        # EXACT 14 features from your notebook
        self.score_feature_list = [
            "HomeTeam_enc", "AwayTeam_enc", "Season_enc", "Month", "DayOfWeek",
            "Home_Last5GoalsFor", "Home_Last5GoalsAgainst", "Away_Last5GoalsFor", 
            "Away_Last5GoalsAgainst", "Home_Last5Points", "Away_Last5Points",
            "H2H_HomeGoalsAvg", "H2H_AwayGoalsAvg", "H2H_HomeWinRate"
        ]
        
        self.load_historical_data()
        
    def load_models(self):
        """Load all trained models and encoders"""
        try:
            import joblib
            
            # Load FTR model
            self.ftr_model = joblib.load('models/ftr_model.pkl')
            print("✅ FTR model loaded")
            
            # Load score models (CatBoostRegressor from your notebook)
            self.home_goals_model = joblib.load('models/home_goals_model.pkl')
            print("✅ Home goals model loaded")
            
            self.away_goals_model = joblib.load('models/away_goals_model.pkl')
            print("✅ Away goals model loaded")
            
            # Load encoders
            self.team_encoder = joblib.load('models/team_encoder.pkl')
            print("✅ Team encoder loaded")
            
            self.season_encoder = joblib.load('models/season_encoder.pkl')
            print("✅ Season encoder loaded")
            
            self.ftr_encoder = joblib.load('models/ftr_encoder.pkl')
            print("✅ FTR encoder loaded")
            
            print("✅ All prediction models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise e
    
    def load_historical_data(self):
        """Load and preprocess historical data"""
        try:
            self.df = pd.read_csv(self.data_path)
            # Convert Date column to datetime
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
            self.df = self.df.dropna(subset=['Date'])
            # Sort by date for proper rolling calculations
            self.df = self.df.sort_values('Date')
            print(f"✅ Historical data loaded: {len(self.df)} matches")
            print(f"✅ Teams in dataset: {len(set(self.df['HomeTeam'].unique()) | set(self.df['AwayTeam'].unique()))}")
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            self.df = pd.DataFrame()
    
    def get_available_teams(self):
        """Get list of available teams from historical data"""
        if not self.df.empty:
            teams = sorted(set(self.df['HomeTeam'].unique()) | set(self.df['AwayTeam'].unique()))
            return teams
        return []
    
    def get_team_last_n_matches(self, team, date, n=5, is_home=True):
        """Get last n matches for a team before the given date"""
        if self.df.empty:
            return pd.DataFrame()
        
        # Filter matches before the prediction date
        past_matches = self.df[self.df['Date'] < date]
        
        if is_home:
            team_matches = past_matches[past_matches['HomeTeam'] == team]
        else:
            team_matches = past_matches[past_matches['AwayTeam'] == team]
        
        # Return last n matches
        return team_matches.tail(n)
    
    def _get_league_averages(self, is_home=True):
        """Calculate REAL Premier League averages from historical data"""
        if self.df.empty:
            # Fallback to reasonable defaults if no data
            return {
                'goals_for_avg': 1.5, 
                'goals_against_avg': 1.2, 
                'points_avg': 1.5
            }
        
        if is_home:
            goals_for_avg = self.df['FTHG'].mean()
            goals_against_avg = self.df['FTAG'].mean()
            home_wins = (self.df['FTR'] == 'H').mean()
            home_draws = (self.df['FTR'] == 'D').mean()
            points_avg = (home_wins * 3 + home_draws)
        else:
            goals_for_avg = self.df['FTAG'].mean()
            goals_against_avg = self.df['FTHG'].mean()
            away_wins = (self.df['FTR'] == 'A').mean()
            away_draws = (self.df['FTR'] == 'D').mean()
            points_avg = (away_wins * 3 + away_draws)
        
        print(f"📊 Using REAL league averages: GF={goals_for_avg:.2f}, GA={goals_against_avg:.2f}, PTS={points_avg:.2f}")
        
        return {
            'goals_for_avg': goals_for_avg,
            'goals_against_avg': goals_against_avg,
            'points_avg': points_avg
        }

    def _get_league_averages_h2h(self):
        """Get REAL averages for any match"""
        if self.df.empty:
            return {
                'home_goals_avg': 1.5, 
                'away_goals_avg': 1.2, 
                'home_win_rate': 0.4
            }
        
        return {
            'home_goals_avg': self.df['FTHG'].mean(),
            'away_goals_avg': self.df['FTAG'].mean(),
            'home_win_rate': (self.df['FTR'] == 'H').mean()
        }

    def calculate_team_form(self, team, date, is_home=True, n_matches=5):
        """Calculate team form using REAL historical data"""
        if self.df.empty:
            return self._get_league_averages(is_home)
        
        team_matches = self.get_team_last_n_matches(team, date, n_matches, is_home)
        
        if len(team_matches) == 0:
            # Use REAL league averages instead of hardcoded values
            print(f"📊 No recent matches for {team}, using league averages")
            return self._get_league_averages(is_home)
        
        # Calculate metrics based on home/away - EXACTLY like your notebook
        if is_home:
            goals_for = team_matches['FTHG'].sum()
            goals_against = team_matches['FTAG'].sum()
            wins = (team_matches['FTR'] == 'H').sum()
            draws = (team_matches['FTR'] == 'D').sum()
        else:
            goals_for = team_matches['FTAG'].sum()
            goals_against = team_matches['FTHG'].sum()
            wins = (team_matches['FTR'] == 'A').sum()
            draws = (team_matches['FTR'] == 'D').sum()
        
        points = wins * 3 + draws
        
        print(f"📊 {team} form: {len(team_matches)} matches, GF={goals_for/len(team_matches):.2f}, GA={goals_against/len(team_matches):.2f}")
        
        return {
            'goals_for_avg': goals_for / len(team_matches),
            'goals_against_avg': goals_against / len(team_matches),
            'points_avg': points / len(team_matches)
        }
    
    def calculate_h2h_stats(self, home_team, away_team, date, n_matches=10):
        """Calculate head-to-head statistics from REAL historical data"""
        if self.df.empty:
            return self._get_league_averages_h2h()
        
        # Filter matches before prediction date
        past_matches = self.df[self.df['Date'] < date]
        
        # Get head-to-head matches
        h2h_matches = past_matches[
            ((past_matches['HomeTeam'] == home_team) & (past_matches['AwayTeam'] == away_team)) |
            ((past_matches['HomeTeam'] == away_team) & (past_matches['AwayTeam'] == home_team))
        ].tail(n_matches)
        
        if h2h_matches.empty:
            print(f"📊 No H2H history for {home_team} vs {away_team}, using league averages")
            return self._get_league_averages_h2h()
        
        home_goals = []
        away_goals = []
        home_wins = 0
        
        for _, match in h2h_matches.iterrows():
            if match['HomeTeam'] == home_team:
                home_goals.append(match['FTHG'])
                away_goals.append(match['FTAG'])
                if match['FTR'] == 'H':
                    home_wins += 1
            else:
                home_goals.append(match['FTAG'])
                away_goals.append(match['FTHG'])
                if match['FTR'] == 'A':
                    home_wins += 1
        
        print(f"📊 H2H: {len(h2h_matches)} matches, {np.mean(home_goals):.2f}-{np.mean(away_goals):.2f}")
        
        return {
            'home_goals_avg': np.mean(home_goals) if home_goals else self.df['FTHG'].mean(),
            'away_goals_avg': np.mean(away_goals) if away_goals else self.df['FTAG'].mean(),
            'home_win_rate': home_wins / len(h2h_matches)
        }

    def show_real_averages(self):
        """Show what the REAL Premier League averages are"""
        if self.df.empty:
            print("❌ No data available")
            return
        
        print("📊 REAL PREMIER LEAGUE AVERAGES:")
        print(f"   Home Goals: {self.df['FTHG'].mean():.2f}")
        print(f"   Away Goals: {self.df['FTAG'].mean():.2f}")
        print(f"   Home Win Rate: {(self.df['FTR'] == 'H').mean():.2%}")
        print(f"   Draw Rate: {(self.df['FTR'] == 'D').mean():.2%}")
        print(f"   Away Win Rate: {(self.df['FTR'] == 'A').mean():.2%}")
    
    def create_score_features(self, home_team, away_team, date_str):
        """Create BASIC feature vector for score prediction (14 features) - EXACT match to notebook"""
        try:
            print(f"🔍 Creating score features for {home_team} vs {away_team} on {date_str}")
            
            # Convert date string to datetime
            date = pd.to_datetime(date_str)

            # 1. Check if historical data is loaded
            if self.df.empty:
                print("❌ No historical data loaded!")
                return None
            
            # 2. Check if teams exist in historical data
            all_teams = set(self.df['HomeTeam'].unique()) | set(self.df['AwayTeam'].unique())
            
            if home_team not in all_teams:
                print(f"❌ Home team '{home_team}' not found in historical data")
                return None
            if away_team not in all_teams:
                print(f"❌ Away team '{away_team}' not found in historical data")
                return None

            # 3. Team encodings
            try:
                home_team_enc = self.team_encoder.transform([home_team])[0]
                away_team_enc = self.team_encoder.transform([away_team])[0]
            except Exception as e:
                print(f"❌ Team encoding error: {e}")
                return None
            
            # 4. Season encoding
            year = date.year
            month = date.month
            if month >= 8:  # Season starts in August
                season = f"{year}-{year+1}"
            else:
                season = f"{year-1}-{year}"
            
            try:
                season_enc = self.season_encoder.transform([season])[0]
            except Exception as e:
                # Use most recent season as fallback
                try:
                    available_seasons = self.season_encoder.classes_
                    most_recent = available_seasons[-1]
                    season_enc = self.season_encoder.transform([most_recent])[0]
                except:
                    print("❌ Season encoding failed")
                    return None
            
            # 5. Date features
            month_feature = date.month
            day_of_week = date.weekday()

            # 6. Team form (from historical data) - EXACTLY like notebook
            home_form = self.calculate_team_form(home_team, date, is_home=True)
            away_form = self.calculate_team_form(away_team, date, is_home=False)
            
            # 7. Head-to-head stats (from historical data) - EXACTLY like notebook
            h2h_stats = self.calculate_h2h_stats(home_team, away_team, date)
            
            # Create BASIC features dictionary for score prediction - EXACT 14 features
            features = {
                "HomeTeam_enc": home_team_enc,
                "AwayTeam_enc": away_team_enc,
                "Season_enc": season_enc,
                "Month": month_feature,
                "DayOfWeek": day_of_week,
                "Home_Last5GoalsFor": home_form['goals_for_avg'],
                "Home_Last5GoalsAgainst": home_form['goals_against_avg'],
                "Away_Last5GoalsFor": away_form['goals_for_avg'],
                "Away_Last5GoalsAgainst": away_form['goals_against_avg'],
                "Home_Last5Points": home_form['points_avg'],
                "Away_Last5Points": away_form['points_avg'],
                "H2H_HomeGoalsAvg": h2h_stats['home_goals_avg'],
                "H2H_AwayGoalsAvg": h2h_stats['away_goals_avg'],
                "H2H_HomeWinRate": h2h_stats['home_win_rate']
            }
            
            print("✅ Score features created successfully!")
            return features
            
        except Exception as e:
            print(f"❌ Error creating score features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_ftr_features(self, home_team, away_team, date_str):
        """Create features for FTR prediction"""
        try:
            print(f"🔍 Creating FTR features for {home_team} vs {away_team} on {date_str}")
            
            # Convert date string to datetime
            date = pd.to_datetime(date_str)

            # 1. Check if historical data is loaded
            if self.df.empty:
                print("❌ No historical data loaded!")
                return None
            
            # 2. Check if teams exist in historical data
            all_teams = set(self.df['HomeTeam'].unique()) | set(self.df['AwayTeam'].unique())
            
            if home_team not in all_teams:
                print(f"❌ Home team '{home_team}' not found in historical data")
                return None
            if away_team not in all_teams:
                print(f"❌ Away team '{away_team}' not found in historical data")
                return None

            # 3. Team encodings
            try:
                home_team_enc = self.team_encoder.transform([home_team])[0]
                away_team_enc = self.team_encoder.transform([away_team])[0]
            except Exception as e:
                print(f"❌ Team encoding error: {e}")
                return None
            
            # 4. Season encoding
            year = date.year
            month = date.month
            if month >= 8:  # Season starts in August
                season = f"{year}-{year+1}"
            else:
                season = f"{year-1}-{year}"
            
            try:
                season_enc = self.season_encoder.transform([season])[0]
            except Exception as e:
                # Use most recent season as fallback
                try:
                    available_seasons = self.season_encoder.classes_
                    most_recent = available_seasons[-1]
                    season_enc = self.season_encoder.transform([most_recent])[0]
                except:
                    print("❌ Season encoding failed")
                    return None
            
            # 5. Date features
            month_feature = date.month
            day_of_week = date.weekday()
            late_season = 1 if month >= 4 else 0  # April onwards is late season

            # 6. Team form (from historical data)
            home_form = self.calculate_team_form(home_team, date, is_home=True)
            away_form = self.calculate_team_form(away_team, date, is_home=False)
            
            # 7. Head-to-head stats (from historical data)
            h2h_stats = self.calculate_h2h_stats(home_team, away_team, date)
            
            # 8. STRENGTH FEATURES (for FTR model only)
            home_attack_strength = home_form['goals_for_avg'] / (home_form['goals_against_avg'] + 1 + self.eps)
            away_attack_strength = away_form['goals_for_avg'] / (away_form['goals_against_avg'] + 1 + self.eps)
            home_defense_strength = 1.0 / (home_form['goals_against_avg'] + 1 + self.eps)
            away_defense_strength = 1.0 / (away_form['goals_against_avg'] + 1 + self.eps)
            
            form_diff_pts = home_form['points_avg'] - away_form['points_avg']
            attack_vs_defense_h = home_attack_strength / (away_defense_strength + self.eps)
            attack_vs_defense_a = away_attack_strength / (home_defense_strength + self.eps)
            
            # 9. Momentum features (for FTR model only)
            home_last3_points = home_form['points_avg'] * 3  # Simplified
            away_last3_points = away_form['points_avg'] * 3  # Simplified
            home_momentum = 0.0  # Simplified
            away_momentum = 0.0  # Simplified
            
            # Create COMPLETE features dictionary for FTR model
            features = {
                # Contextual features
                "Season_enc": season_enc,
                "Month": month_feature,
                "DayOfWeek": day_of_week,
                "Late_Season": late_season,
                
                # Team rolling form
                "Home_Last5GoalsFor": home_form['goals_for_avg'],
                "Home_Last5GoalsAgainst": home_form['goals_against_avg'],
                "Away_Last5GoalsFor": away_form['goals_for_avg'],
                "Away_Last5GoalsAgainst": away_form['goals_against_avg'],
                "Home_Last5Points": home_form['points_avg'],
                "Away_Last5Points": away_form['points_avg'],
                
                # Strength & momentum features (FTR model only)
                "Home_AttackStrength": home_attack_strength,
                "Away_AttackStrength": away_attack_strength,
                "Home_DefenseStrength": home_defense_strength,
                "Away_DefenseStrength": away_defense_strength,
                "FormDiff_Pts": form_diff_pts,
                "Attack_vs_Defense_H": attack_vs_defense_h,
                "Attack_vs_Defense_A": attack_vs_defense_a,
                "Home_Last3Points_Sum": home_last3_points,
                "Away_Last3Points_Sum": away_last3_points,
                "Home_Momentum": home_momentum,
                "Away_Momentum": away_momentum,
                
                # Head-to-head
                "H2H_HomeGoalsAvg": h2h_stats['home_goals_avg'],
                "H2H_AwayGoalsAvg": h2h_stats['away_goals_avg'],
                "H2H_HomeWinRate": h2h_stats['home_win_rate'],
                
                # Encoded teams
                "HomeTeam_enc": home_team_enc,
                "AwayTeam_enc": away_team_enc,
                
                # HTR encoding (default to draw)
                "HTR_enc": 1  # Default to draw encoding
            }
            
            print("✅ FTR features created successfully!")
            return features
            
        except Exception as e:
            print(f"❌ Error creating FTR features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_score(self, home_team, away_team, date_str):
        """Predict Final Score from 3 inputs - EXACT match to notebook logic"""
        try:
            features = self.create_score_features(home_team, away_team, date_str)
            
            if features is None:
                return {'success': False, 'error': 'Could not create features for prediction'}
            
            # Create feature array in EXACT same order as notebook
            feature_array = []
            for feature in self.score_feature_list:
                feature_array.append(features.get(feature, 0))
            
            feature_array = np.array([feature_array])
            
            # DEBUG: Show feature values to compare with notebook
            print(f"🔍 FEATURE VALUES for {home_team} vs {away_team}:")
            for i, feature in enumerate(self.score_feature_list):
                print(f"   {feature}: {feature_array[0][i]:.6f}")
            
            # Predict goals using your score models - EXACT same as notebook
            home_goals_raw = self.home_goals_model.predict(feature_array)[0]
            away_goals_raw = self.away_goals_model.predict(feature_array)[0]
            
            print(f"🎯 Raw predictions - Home: {home_goals_raw:.6f}, Away: {away_goals_raw:.6f}")
            
            # EXACT SAME LOGIC AS YOUR NOTEBOOK:
            # pred_home_rounded = np.round(pred_home).astype(int)
            # pred_home_rounded = np.clip(pred_home_rounded, 0, 6)
            home_goals = int(np.round(home_goals_raw))
            away_goals = int(np.round(away_goals_raw))
            
            # Apply clipping like notebook
            home_goals = max(0, home_goals)
            away_goals = max(0, away_goals)
            home_goals = min(home_goals, 6)
            away_goals = min(away_goals, 6)
            
            print(f"🎯 Final score: {home_goals}-{away_goals}")
            
            return {
                'success': True,
                'prediction': f"{home_goals}-{away_goals}",
                'home_goals': home_goals,
                'away_goals': away_goals,
                'features_used': len(self.score_feature_list)
            }
            
        except Exception as e:
            print(f"❌ Error in predict_score: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def predict_ftr(self, home_team, away_team, date_str):
        """Predict Full Time Result from 3 inputs"""
        try:
            features = self.create_ftr_features(home_team, away_team, date_str)
            
            if features is None:
                return {'success': False, 'error': 'Could not create features for prediction'}
            
            # Create feature array in correct order for FTR model
            feature_array = []
            for feature in self.ftr_feature_list:
                feature_array.append(features.get(feature, 0))
            
            feature_array = np.array([feature_array])
            
            # Make prediction using your FTR model
            prediction_encoded = self.ftr_model.predict(feature_array)[0]
            probabilities = self.ftr_model.predict_proba(feature_array)[0]
            
            # Decode prediction
            prediction = self.ftr_encoder.inverse_transform([prediction_encoded])[0]
            
            # Map to readable results
            result_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
            prediction_readable = result_map.get(prediction, prediction)

            # FIX: Swap probabilities if needed
            # Most likely your encoder uses ['A', 'D', 'H'] order
            home_win_prob = probabilities[2]  # Third position for Home
            draw_prob = probabilities[1]      # Second position for Draw  
            away_win_prob = probabilities[0]  # First position for Away

            return {
                'success': True,
                'prediction': prediction_readable,
                'probabilities': {
                'home_win': round(home_win_prob * 100, 2),
                'draw': round(draw_prob * 100, 2),
                'away_win': round(away_win_prob * 100, 2)
                 },
              'features_used': len(self.ftr_feature_list)
                 }
            
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def debug_prediction(self, home_team, away_team, date_str):
        """Debug prediction to see exact values"""
        print(f"=== DEBUG PREDICTION: {home_team} vs {away_team} ===")
        
        # Test score prediction
        score_result = self.predict_score(home_team, away_team, date_str)
        print(f"Score Result: {score_result}")
        
        # Test FTR prediction  
        ftr_result = self.predict_ftr(home_team, away_team, date_str)
        print(f"FTR Result: {ftr_result}")


        
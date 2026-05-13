from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions_made = db.Column(db.Integer, default=0)
    correct_predictions = db.Column(db.Integer, default=0)
    favorite_team = db.Column(db.String(50))
    
    def get_accuracy(self):
        return (self.correct_predictions / self.predictions_made * 100) if self.predictions_made > 0 else 0
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'predictions_made': self.predictions_made,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.get_accuracy(),
            'favorite_team': self.favorite_team,
            'created_at': self.created_at.isoformat()
        }

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    home_team = db.Column(db.String(50), nullable=False)
    away_team = db.Column(db.String(50), nullable=False)
    predicted_result = db.Column(db.String(10), nullable=False)
    actual_result = db.Column(db.String(10))
    match_date = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_correct = db.Column(db.Boolean)
    
    user = db.relationship('User', backref='predictions')
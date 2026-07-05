import numpy as np
import pandas as pd
import lightgbm as lgb
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, precision_recall_curve, average_precision_score

# Constants
GOAL_X = 1  # Standard soccer pitch end X coordinate
GOAL_Y = 0.5

def extract_features(data_path, has_target=True):
    """
    Extract relevant features from shots data for goal probability prediction.
    Uses pre-calculated features from the CSV with new positional features.
    
    Parameters:
    - data_path: Path to the shot_opportunities_database.csv file
    - has_target: Whether the dataset contains actual goal outcomes
    
    Returns:
    - features: Numpy array of features [dist_to_goal_line, dist_from_center, density, defenders_in_path]
    - targets: Numpy array of target outcomes (1=goal, 0=no goal)
    """
    data = pd.read_csv(data_path)


    original_count = len(data)
    data = data[data['tir_observe'] == 1]
    filtered_count = len(data)
    print(f"Filtered data: keeping {filtered_count} shots out of {original_count} opportunities ({filtered_count/original_count*100:.1f}%)")
    
    features = []
    targets = []
    
    # Print the number of shots loaded
    print(f"Loaded {len(data)} shots from the database")
    
    for idx, row in data.iterrows():
        # Skip rows with missing required features or positions
        if (pd.isna(row['def_pressure']) or pd.isna(row['nb_defenders']) or
            pd.isna(row['x']) or pd.isna(row['y'])):
            continue
            
        # Calculate new positional features
        dist_to_goal_line = 1 - row['x']  # Distance to goal line (1-x)
        dist_from_center = abs(row['y'] - 0.5)  # Distance from center of field
        
        # Get defensive features directly from CSV
        density = row['distance_def_le_plus_proche']
        defenders_in_path = row['nb_defenders']
        
        # Create feature vector with new features
        feature_vector = [dist_to_goal_line, dist_from_center, density, defenders_in_path]
        features.append(feature_vector)
        
        # Target variable (actual goal outcomes)
        if has_target and 'is_goal' in row and not pd.isna(row['is_goal']):
            target = int(row['is_goal'])
        else:
            # Simplified heuristic if no actual outcomes
            target = 1 if (dist_to_goal_line < 0.1 and dist_from_center < 0.2 and defenders_in_path < 2) else 0
        
        targets.append(target)
    
    return np.array(features), np.array(targets)

def train_model(X, y):
    """Train LightGBM model with monotonicity constraints and class balancing."""
    # Calculate class weights
    neg_samples = sum(1 for label in y if label == 0)
    pos_samples = sum(1 for label in y if label == 1)
    
    # Calculate the scale_pos_weight (ratio of negative to positive samples)
    if pos_samples > 0:  # Avoid division by zero
        scale_pos_weight = neg_samples / pos_samples
        print(f"Class imbalance ratio (neg:pos): {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = 1.0
        print("Warning: No positive samples found!")
    
    # Define monotonicity constraints:
    # All features should have negative impact (higher value = worse chance)
    monotone_constraints = [-1, -1, -1, -1]
    
    train_data = lgb.Dataset(
        X, 
        label=y, 
        feature_name=['dist_to_goal_line', 'dist_from_center', 'density', 'defenders_in_path']
    )
    
    params = {
        'objective': 'binary',  # Logistic objective
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'monotone_constraints': monotone_constraints,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,  # Prevents overfitting
        'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
        'verbose': -1
    }
    
    model = lgb.train(params, train_data, num_boost_round=100)
    return model

# Then modify your code to compare models
def train_multiple_models(X_train, y_train, X_test, y_test):
    # Define models to try
    models = {
        'LightGBM': train_model(X_train, y_train),  # Your existing function
        'XGBoost': XGBClassifier(scale_pos_weight=7.8, learning_rate=0.05, 
                               n_estimators=100),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        'LogisticRegression': LogisticRegression(class_weight='balanced')
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        if name != 'LightGBM':  # LightGBM is already trained
            model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict_proba(X_test)[:,1]
        
        # Evaluate
        results[name] = {
            'log_loss': log_loss(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred),
            'avg_precision': average_precision_score(y_test, y_pred)
        }
        
        print(f"\n{name} Results:")
        print(f"Log loss: {results[name]['log_loss']:.4f}")
        print(f"ROC AUC: {results[name]['roc_auc']:.4f}")
        print(f"Average Precision: {results[name]['avg_precision']:.4f}")
    
    return models, results

# Global model variable
_model = None

def load_model(filepath='go_alone_model.txt'):
    """Load a trained model."""
    global _model
    try:
        _model = lgb.Booster(model_file=filepath)
        print(f"Model loaded successfully from {filepath}")
    except:
        raise ValueError(f"Could not load model from {filepath}")

def go_alone(x, y, op_positions=None, def_pressure=None, nb_defenders=None):
    """
    Calculate probability of scoring when going alone.
    
    Parameters:
    - x, y: Shot position coordinates 
    - op_positions: List of opponent positions (used to calculate defensive metrics if not provided)
    - def_pressure: Defensive pressure (if provided, overrides calculation)
    - nb_defenders: Number of defenders in path (if provided, overrides calculation)
    
    Returns:
    - Probability of scoring
    """
    global _model
    
    if _model is None:
        raise ValueError("Model not loaded. Call load_model() first.")
    
    # Handle None case for op_positions
    if op_positions is None:
        op_positions = []
    
    # Calculate the new features
    dist_to_goal_line = 1 - x  # Distance to goal line (1-x)
    dist_from_center = abs(y - 0.5)  # Distance from center of field
        
    # Calculate defensive features if not provided
    if def_pressure is None:
        density = np.min([np.linalg.norm(np.array([x*120, y*80]) - np.array([pos[0]*120, pos[1]*80])) for pos in op_positions]) if op_positions else 0.0
    else:
        density = def_pressure
        
    if nb_defenders is None:
        from Pass_chances_function import nb_adv_trajectoire_coords
        try:
            defenders_in_path = nb_adv_trajectoire_coords(x, y, GOAL_X, GOAL_Y, op_positions) if op_positions else 0
        except ValueError:
            defenders_in_path = 0
    else:
        defenders_in_path = nb_defenders
    
    # Create feature array for prediction
    features = np.array([[dist_to_goal_line, dist_from_center, density, defenders_in_path]])
    
    # Make prediction
    return _model.predict(features)[0]

def save_model(model, filepath='go_alone_model.txt'):
    """Save the model to a file."""
    model.save_model(filepath)
    print(f"Model saved to {filepath}")

if __name__ == "__main__":
    import os
    from config import MODELS_DIR, PROCESSED_DIR

    # Path to the shot opportunities database
    data_path = str(PROCESSED_DIR / "shot_opportunities_database.csv")
    
    print(f"Loading data from: {data_path}")
    
    try:
        # Extract features and targets
        X, y = extract_features(data_path)
        
        print(f"Extracted {len(X)} shots with features")
        print(f"Positive examples (goals): {sum(y)}")
        print(f"Negative examples (non-goals): {len(y) - sum(y)}")
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        print("Training model...")
        model = train_model(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        print(f"Log loss: {log_loss(y_test, y_pred)}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")
        print(f"Average Precision Score: {average_precision_score(y_test, y_pred)}")
        
        # Save model
        model_path = str(MODELS_DIR / "go_alone_model.txt")
        save_model(model, model_path)
        
        # Set model for prediction
        _model = model
        
        # Test extreme cases
        print("\nTesting extreme cases:")
        
        # Penalty situation: Directly in front of goal
        penalty_prob = go_alone(0.95, 0.5, [(1,0.5),(0.8,0.6)], None, None)  # x close to 1, y at 0.5, no pressure
        print(f"Penalty probability: {penalty_prob:.4f}")
        
        # Good situation: Inside the box, center
        good_prob = go_alone(0.85, 0.5, [(1,0.5),(0.8,0.6)], None, None)  # Close to goal line, centered
        print(f"Good shot probability: {good_prob:.4f}")
        
        # Difficult situation: Further from goal line with defenders
        difficult_prob = go_alone(0.7, 0.6, None, 0.5, 2)  # Further from goal line, off center
        print(f"Difficult shot probability: {difficult_prob:.4f}")
        
        # Very difficult situation: Far from goal, far off center, many defenders
        very_difficult_prob = go_alone(0.3, 0.9, None, 0.8, 4)  # Far from goal line, far off center
        print(f"Very difficult shot probability: {very_difficult_prob:.4f}")
        
        # Print feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_names = ['dist_to_goal_line', 'dist_from_center', 'density', 'defenders_in_path']
        
        print("\nFeature importance:")
        for i, name in enumerate(feature_names):
            print(f"{name}: {importance[i]}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
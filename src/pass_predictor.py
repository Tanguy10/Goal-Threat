import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import json
import os
import joblib
from config import MODELS_DIR

class PassPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.optimal_params = None
        self.model_loaded = False

    def load_model(self):
        """Charge le meilleur modèle entraîné avec ses paramètres optimaux"""

        # Charger les paramètres optimaux d'abord
        best_params_path = MODELS_DIR / "best_params.json"
        if best_params_path.exists():
            try:
                with open(best_params_path, "r") as f:
                    self.optimal_params = json.load(f)
                print(f"Paramètres optimaux chargés: {self.optimal_params}")
            except Exception as e:
                print(f"Erreur lors du chargement de best_params.json: {e}")
                self.optimal_params = None
        else:
            print("Attention: best_params.json non trouvé")
            self.optimal_params = None

        # Détecter et charger le meilleur modèle
        model_files = [
            ("best_pass_model_rf.pkl", "RandomForest", "pkl"),
            ("best_pass_model_lr.pkl", "LogisticRegression", "pkl"),
            ("best_pass_model_xgb.pkl", "XGBoost_sklearn", "pkl"),
            ("best_pass_model_xgb.json", "XGBoost_native", "json"),
        ]

        model_loaded = False

        for filename, model_name, file_type in model_files:
            filename = str(MODELS_DIR / filename)
            if os.path.exists(filename):
                try:
                    if file_type == "pkl":
                        # Modèle scikit-learn ou XGBClassifier sauvegardé avec joblib/pickle
                        with open(filename, 'rb') as f:
                            self.model = pickle.load(f)
                        print(f"✅ Modèle chargé: {model_name} depuis {filename}")
                        model_loaded = True
                        break
                    elif file_type == "json":
                        # Modèle natif XGBoost (Booster)
                        self.model = xgb.Booster()
                        self.model.load_model(filename)
                        print(f"✅ Modèle XGBoost natif chargé depuis {filename}")
                        model_loaded = True
                        break
                except Exception as e:
                    print(f"❌ Erreur chargement {filename}: {e}")
                    continue

        if not model_loaded:
            raise FileNotFoundError("Aucun modèle best_pass_model_* trouvé (pkl ou json)")

        # Charger les métadonnées
        metadata_path = MODELS_DIR / "pass_model_metadata.pkl"
        if os.path.exists(metadata_path):
            try:
                metadata = joblib.load(metadata_path)
                self.scaler = metadata.get("scaler")
                self.features = metadata.get("features")
                print("✅ Métadonnées chargées avec succès")
                if self.features:
                    print(f"Features utilisées: {self.features}")
            except Exception as e:
                print(f"Erreur chargement métadonnées: {e}")
                self.features = [
                    'distance_passe', 'sens_passe',
                    'nb_adv_proches_depart', 'nb_adv_trajectoire',
                    'nb_adv_proches_arrivee', 'nb_coequipiers_proches_arrivee'
                ]
                print(f"Utilisation des features par défaut: {self.features}")
        else:
            print("pass_model_metadata.pkl non trouvé")
            self.features = [
                'distance_passe', 'sens_passe',
                'nb_adv_proches_depart', 'nb_adv_trajectoire',
                'nb_adv_proches_arrivee', 'nb_coequipiers_proches_arrivee'
            ]

        self.model_loaded = True
        return self
    
    def predict_proba(self, df_input):
        """
        Méthode obligatoire pour compatibilité avec scikit-learn et notebooks.
        Retourne un array de forme (n_samples, 2) = [[P(neg), P(pos)], ...]
        """
        probs = self.predict_pass_success(df_input)
        # Transformer en format [[1-p, p], [1-p, p], ...] pour chaque échantillon
        return np.vstack([1 - probs, probs]).T

    def get_optimal_params(self):
        """Retourne les paramètres optimaux trouvés par Optuna"""
        if not self.model_loaded:
            self.load_model()
        return self.optimal_params

    def predict_pass_success(self, df_input):
        """
        Prédit la probabilité de succès des passes avec le modèle optimal
        
        Args:
            df_input: DataFrame avec les colonnes correspondant aux features
            
        Returns:
            Array des probabilités de succès entre 0 et 1
        """
        if not self.model_loaded:
            self.load_model()

        # Vérifier que toutes les colonnes nécessaires sont présentes
        missing_cols = [col for col in self.features if col not in df_input.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")

        # Normaliser les features si scaler disponible
        if self.scaler is not None:
            X_scaled = self.scaler.transform(df_input[self.features])
        else:
            X_scaled = df_input[self.features].values
            print("Attention: pas de scaler disponible, utilisation des données brutes")
        
        # Prédire les probabilités
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)[:, 1]
        elif isinstance(self.model, xgb.Booster):
            dmat = xgb.DMatrix(X_scaled, feature_names=self.features)
            return self.model.predict(dmat)
        else:
            return self.model.predict(X_scaled)

    def predict_single_pass(self, x_passeur, y_passeur, x_cible, y_cible,
                          densite_adv_depart=1.0, nb_adv_trajectoire=1,
                          densite_adv_arrivee=1.0, densite_coequipiers_arrivee=1.0):
        """
        Prédit la probabilité de succès pour une passe spécifique avec les paramètres optimaux
        
        Args:
            x_passeur, y_passeur: Position du passeur (coordonnées normalisées 0-1)
            x_cible, y_cible: Position cible (coordonnées normalisées 0-1)
            densite_adv_depart: Densité d'adversaires près du passeur (valeur continue)
            nb_adv_trajectoire: Nombre d'adversaires sur la trajectoire (entier)
            densite_adv_arrivee: Densité d'adversaires près de la cible (valeur continue)
            densite_coequipiers_arrivee: Densité de coéquipiers près de la cible (valeur continue)
            
        Returns:
            Probabilité de succès (0-1)
        """
                # Calculer distance_passe et sens_passe comme dans Pass_chances_function

        # Créer un DataFrame avec les bonnes features
        pass_data = pd.DataFrame({
            'x_passeur': [x_passeur],
            'y_passeur': [y_passeur],
            'nb_adv_proches_depart': [densite_adv_depart],  # Densité, pas comptage
            'nb_adv_trajectoire': [nb_adv_trajectoire],      # Comptage
            'nb_adv_proches_arrivee': [densite_adv_arrivee], # Densité, pas comptage
            'nb_coequipiers_proches_arrivee': [densite_coequipiers_arrivee] # Densité
        })
        
        return self.predict_pass_success(pass_data)[0]

    def print_model_info(self):
        """Affiche les informations sur le modèle et ses paramètres optimaux"""
        if not self.model_loaded:
            self.load_model()
            
        print("=== INFORMATIONS DU MODÈLE OPTIMAL ===")
        print(f"Type de modèle: {type(self.model).__name__}")
        print(f"Features utilisées: {self.features}")
        
        if self.optimal_params:
            print("Paramètres optimaux trouvés par Optuna:")
            for param, value in self.optimal_params.items():
                print(f"  {param}: {value}")
        else:
            print("Paramètres optimaux non disponibles")

    def calculate_pass_features(self, x_passeur, y_passeur, x_cible, y_cible, 
                               teammates_pos, opponents_pos):
        """
        Calcule toutes les features nécessaires pour une passe donnée
        
        Args:
            x_passeur, y_passeur: Position du passeur (normalisées 0-1)
            x_cible, y_cible: Position cible (normalisées 0-1)
            teammates_pos: Array des positions des coéquipiers (normalisées)
            opponents_pos: Array des positions des adversaires (normalisées)
            
        Returns:
            dict: Dictionnaire avec toutes les features
        """
        # Paramètres par défaut (ajustables selon les paramètres optimaux)
        if self.optimal_params:
            sigma = self.optimal_params.get('sigma', 0.05)
            seuil_trajectoire = self.optimal_params.get('seuil_trajectoire', 0.02)
        else:
            print("Aucun paramètre optimal trouvé, utilisation des valeurs par défaut")
            sigma = 0.05
            seuil_trajectoire = 0.02
            
        # Distance et sens de la passe
        distance_passe = np.sqrt((x_cible - x_passeur)**2 + (y_cible - y_passeur)**2)
        sens_passe = 1 if x_cible > x_passeur else -1
        
        # Position du passeur et de la cible
        passeur_pos = np.array([x_passeur, y_passeur])
        cible_pos = np.array([x_cible, y_cible])
        
        # Calcul des adversaires proches du départ (densité gaussienne)
        if len(opponents_pos) > 0:
            distances_depart = np.linalg.norm(opponents_pos - passeur_pos, axis=1)
            densites_depart = np.exp(-(distances_depart**2) / (2 * sigma**2))
            nb_adv_proches_depart = np.sum(densites_depart)
        else:
            nb_adv_proches_depart = 0
            
        # Calcul des adversaires sur la trajectoire
        nb_adv_trajectoire = self._count_opponents_on_trajectory(
            passeur_pos, cible_pos, opponents_pos, seuil_trajectoire
        )
        
        # Calcul des adversaires proches de l'arrivée (densité gaussienne)
        if len(opponents_pos) > 0:
            distances_arrivee = np.linalg.norm(opponents_pos - cible_pos, axis=1)
            densites_arrivee = np.exp(-(distances_arrivee**2) / (2 * sigma**2))
            nb_adv_proches_arrivee = np.sum(densites_arrivee)
        else:
            nb_adv_proches_arrivee = 0
            
        # Calcul des coéquipiers proches de l'arrivée (densité gaussienne)
        if len(teammates_pos) > 0:
            distances_coequipiers = np.linalg.norm(teammates_pos - cible_pos, axis=1)
            densites_coequipiers = np.exp(-(distances_coequipiers**2) / (2 * sigma**2))
            nb_coequipiers_proches_arrivee = np.sum(densites_coequipiers)
        else:
            nb_coequipiers_proches_arrivee = 0
        
        return {
            'distance_passe': distance_passe,
            'sens_passe': sens_passe,
            'nb_adv_proches_depart': nb_adv_proches_depart,
            'nb_adv_trajectoire': nb_adv_trajectoire,
            'nb_adv_proches_arrivee': nb_adv_proches_arrivee,
            'nb_coequipiers_proches_arrivee': nb_coequipiers_proches_arrivee
        }
    
    def _count_opponents_on_trajectory(self, passeur_pos, cible_pos, opponents_pos, seuil_trajectoire):
        """Compte les adversaires sur la trajectoire de la passe"""
        if len(opponents_pos) == 0:
            return 0
            
        traj_vect = cible_pos - passeur_pos
        norm_traj = np.linalg.norm(traj_vect)
        
        if norm_traj == 0:
            return 0
            
        traj_vect_norm = traj_vect / norm_traj
        
        # Calcul des projections
        adv_vectors = opponents_pos - passeur_pos
        projections = np.dot(adv_vectors, traj_vect_norm)
        
        # Adversaires sur la trajectoire
        on_trajectory = (projections > 0) & (projections < norm_traj)
        
        if not np.any(on_trajectory):
            return 0
            
        # Distance à la ligne de trajectoire
        points_on_line = passeur_pos + np.outer(projections[on_trajectory], traj_vect_norm)
        distances = np.linalg.norm(opponents_pos[on_trajectory] - points_on_line, axis=1)
        
        return np.sum(distances < seuil_trajectoire)
    
    def predict_pass_probability(self, features_df):
        """
        Prédit la probabilité de succès d'une passe
        
        Args:
            features_df: DataFrame avec les features
            
        Returns:
            array: Probabilités de succès
        """
        if not self.model_loaded:
            self.load_model()
            
        # Vérifier que toutes les features sont présentes
        for feature in self.features:
            if feature not in features_df.columns:
                raise ValueError(f"Feature manquante: {feature}")
        
        # Sélectionner et ordonner les features
        X = features_df[self.features]
        
        # Normaliser
        X_scaled = self.scaler.transform(X)
        
        # Prédire
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return probabilities



# Exemple d'utilisation
if __name__ == "__main__":
    predictor = PassPredictor().load_model()
    predictor.print_model_info()
    
    # Test d'une prédiction avec les bons noms de paramètres
    prob = predictor.predict_single_pass(
        x_passeur=0.3, y_passeur=0.4,  # Coordonnées normalisées
        x_cible=0.7, y_cible=0.2,
        densite_adv_depart=2.0,        # Densité d'adversaires au départ
        nb_adv_trajectoire=1,          # Nombre d'adversaires sur trajectoire
        densite_adv_arrivee=1.0,       # Densité d'adversaires à l'arrivée
        densite_coequipiers_arrivee=2.0 # Densité de coéquipiers à l'arrivée
    )
    print(f"\nProbabilité de succès de la passe: {prob:.1%}")
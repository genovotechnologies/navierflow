from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class OptimizationType(Enum):
    """Optimization types"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"

@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    type: OptimizationType
    n_trials: int = 100
    n_splits: int = 5
    metric: str = "mse"
    direction: str = "minimize"
    timeout: Optional[float] = None
    n_jobs: int = -1
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Initialize configuration"""
        if self.n_trials <= 0:
            raise ValueError("Number of trials must be positive")
            
        if self.n_splits <= 0:
            raise ValueError("Number of splits must be positive")
            
        if self.metric not in ["mse", "r2"]:
            raise ValueError("Metric must be 'mse' or 'r2'")
            
        if self.direction not in ["minimize", "maximize"]:
            raise ValueError("Direction must be 'minimize' or 'maximize'")
            
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Timeout must be positive")
            
        if self.n_jobs == 0:
            raise ValueError("Number of jobs must not be zero")

class Optimizer:
    def __init__(self, config: OptimizationConfig):
        """
        Initialize optimizer
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.study = None
        self.best_params = None
        self.best_value = None
        self.history = None
        
    def _create_study(self):
        """Create study"""
        self.study = optuna.create_study(
            direction=self.config.direction,
            sampler=TPESampler(seed=self.config.seed),
            pruner=MedianPruner()
        )
        
    def _objective(self,
                  trial: Trial,
                  model_fn: Callable,
                  X: np.ndarray,
                  y: np.ndarray,
                  param_space: Dict[str, Any]) -> float:
        """
        Objective function
        
        Args:
            trial: Trial
            model_fn: Model function
            X: Features
            y: Target
            param_space: Parameter space
            
        Returns:
            Objective value
        """
        # Sample parameters
        params = {}
        for name, space in param_space.items():
            if isinstance(space, tuple):
                if isinstance(space[0], int):
                    params[name] = trial.suggest_int(name, *space)
                elif isinstance(space[0], float):
                    params[name] = trial.suggest_float(name, *space)
                elif isinstance(space[0], str):
                    params[name] = trial.suggest_categorical(name, space)
            else:
                params[name] = space
                
        # Cross-validation
        kf = KFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.seed
        )
        
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = model_fn(**params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_val)
            
            if self.config.metric == "mse":
                score = mean_squared_error(y_val, y_pred)
            else:
                score = r2_score(y_val, y_pred)
                
            scores.append(score)
            
        return np.mean(scores)
        
    def optimize(self,
                model_fn: Callable,
                X: np.ndarray,
                y: np.ndarray,
                param_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize parameters
        
        Args:
            model_fn: Model function
            X: Features
            y: Target
            param_space: Parameter space
            
        Returns:
            Best parameters
        """
        # Create study
        self._create_study()
        
        # Optimize
        self.study.optimize(
            lambda trial: self._objective(trial, model_fn, X, y, param_space),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        # Store results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        self.history = {
            "params": [t.params for t in self.study.trials],
            "values": [t.value for t in self.study.trials]
        }
        
        return self.best_params
        
    def plot_optimization_history(self):
        """Plot optimization history"""
        if self.history is None:
            raise ValueError("No optimization history available")
            
        plt.figure(figsize=(10, 6))
        
        # Plot values
        plt.plot(self.history["values"], label="Objective value")
        
        # Plot best value
        best_idx = np.argmin(self.history["values"]) if self.config.direction == "minimize" else np.argmax(self.history["values"])
        plt.scatter(best_idx, self.history["values"][best_idx], color="red", label="Best value")
        
        plt.xlabel("Trial")
        plt.ylabel("Objective value")
        plt.title("Optimization History")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_parameter_importance(self):
        """Plot parameter importance"""
        if self.study is None:
            raise ValueError("No study available")
            
        importance = optuna.importance.get_param_importances(self.study)
        
        plt.figure(figsize=(10, 6))
        
        # Plot importance
        plt.bar(importance.keys(), importance.values())
        
        plt.xlabel("Parameter")
        plt.ylabel("Importance")
        plt.title("Parameter Importance")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
        
    def plot_parameter_relationships(self):
        """Plot parameter relationships"""
        if self.history is None:
            raise ValueError("No optimization history available")
            
        # Create DataFrame
        import pandas as pd
        df = pd.DataFrame(self.history["params"])
        df["value"] = self.history["values"]
        
        # Plot relationships
        plt.figure(figsize=(12, 8))
        sns.pairplot(df)
        plt.show()
        
    def save(self, path: str):
        """
        Save optimization results
        
        Args:
            path: Path to save results
        """
        if self.study is None:
            raise ValueError("No study available")
            
        optuna.dump_study(self.study, path)
        
    @classmethod
    def load(cls, path: str) -> "Optimizer":
        """
        Load optimization results
        
        Args:
            path: Path to load results from
            
        Returns:
            Loaded optimizer
        """
        study = optuna.load_study(path)
        
        optimizer = cls(OptimizationConfig(
            type=OptimizationType.BAYESIAN,
            n_trials=len(study.trials),
            direction=study.direction
        ))
        
        optimizer.study = study
        optimizer.best_params = study.best_params
        optimizer.best_value = study.best_value
        optimizer.history = {
            "params": [t.params for t in study.trials],
            "values": [t.value for t in study.trials]
        }
        
        return optimizer 
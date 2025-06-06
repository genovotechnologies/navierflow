import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Callable, Union
from dataclasses import dataclass
from enum import Enum
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

class OptimizationMethod(Enum):
    """Optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"

@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    method: OptimizationMethod = OptimizationMethod.BAYESIAN
    n_trials: int = 100
    n_splits: int = 5
    metric: str = "mse"
    use_gpu: bool = True
    use_parallel: bool = True
    use_early_stopping: bool = True
    patience: int = 10

class HyperparameterOptimizer:
    def __init__(self,
                 model_class: type,
                 param_space: Dict[str, List[Union[float, int, str]]],
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize hyperparameter optimizer
        
        Args:
            model_class: Model class to optimize
            param_space: Parameter space to search
            config: Optimization configuration
        """
        self.model_class = model_class
        self.param_space = param_space
        self.config = config or OptimizationConfig()
        self.best_params = None
        self.best_score = float("inf")
        
    def optimize(self,
                x: torch.Tensor,
                y: torch.Tensor,
                physics_equations: List[Callable]) -> Dict:
        """
        Optimize hyperparameters
        
        Args:
            x: Input tensor
            y: Target tensor
            physics_equations: List of physics equations
            
        Returns:
            Best parameters and score
        """
        if self.config.method == OptimizationMethod.GRID_SEARCH:
            return self._grid_search(x, y, physics_equations)
        elif self.config.method == OptimizationMethod.RANDOM_SEARCH:
            return self._random_search(x, y, physics_equations)
        elif self.config.method == OptimizationMethod.BAYESIAN:
            return self._bayesian_optimization(x, y, physics_equations)
        elif self.config.method == OptimizationMethod.EVOLUTIONARY:
            return self._evolutionary_optimization(x, y, physics_equations)
        elif self.config.method == OptimizationMethod.GRADIENT_BASED:
            return self._gradient_based_optimization(x, y, physics_equations)
        else:
            raise ValueError(f"Unknown optimization method: {self.config.method}")
            
    def _grid_search(self,
                    x: torch.Tensor,
                    y: torch.Tensor,
                    physics_equations: List[Callable]) -> Dict:
        """Grid search optimization"""
        from itertools import product
        
        # Generate parameter combinations
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        param_combinations = list(product(*param_values))
        
        # Cross-validation
        kf = KFold(n_splits=self.config.n_splits, shuffle=True)
        
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            scores = []
            
            for train_idx, val_idx in kf.split(x):
                # Split data
                x_train, x_val = x[train_idx], x[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model = self.model_class(**param_dict)
                model.train(x_train, y_train, physics_equations)
                
                # Evaluate model
                y_pred = model.predict(x_val)
                score = self._compute_metric(y_val, y_pred)
                scores.append(score)
                
            # Update best parameters
            mean_score = np.mean(scores)
            if mean_score < self.best_score:
                self.best_score = mean_score
                self.best_params = param_dict
                
        return {
            "best_params": self.best_params,
            "best_score": self.best_score
        }
        
    def _random_search(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      physics_equations: List[Callable]) -> Dict:
        """Random search optimization"""
        # Cross-validation
        kf = KFold(n_splits=self.config.n_splits, shuffle=True)
        
        for _ in range(self.config.n_trials):
            # Sample parameters
            params = self._sample_parameters()
            scores = []
            
            for train_idx, val_idx in kf.split(x):
                # Split data
                x_train, x_val = x[train_idx], x[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model = self.model_class(**params)
                model.train(x_train, y_train, physics_equations)
                
                # Evaluate model
                y_pred = model.predict(x_val)
                score = self._compute_metric(y_val, y_pred)
                scores.append(score)
                
            # Update best parameters
            mean_score = np.mean(scores)
            if mean_score < self.best_score:
                self.best_score = mean_score
                self.best_params = params
                
        return {
            "best_params": self.best_params,
            "best_score": self.best_score
        }
        
    def _bayesian_optimization(self,
                             x: torch.Tensor,
                             y: torch.Tensor,
                             physics_equations: List[Callable]) -> Dict:
        """Bayesian optimization"""
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        
        # Define parameter space
        param_space = []
        for param_name, param_values in self.param_space.items():
            if isinstance(param_values[0], int):
                param_space.append(Integer(param_values[0], param_values[-1], name=param_name))
            elif isinstance(param_values[0], float):
                param_space.append(Real(param_values[0], param_values[-1], name=param_name))
            else:
                param_space.append(Categorical(param_values, name=param_name))
                
        # Define objective function
        def objective(params):
            param_dict = dict(zip(self.param_space.keys(), params))
            scores = []
            
            # Cross-validation
            kf = KFold(n_splits=self.config.n_splits, shuffle=True)
            for train_idx, val_idx in kf.split(x):
                # Split data
                x_train, x_val = x[train_idx], x[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model = self.model_class(**param_dict)
                model.train(x_train, y_train, physics_equations)
                
                # Evaluate model
                y_pred = model.predict(x_val)
                score = self._compute_metric(y_val, y_pred)
                scores.append(score)
                
            return np.mean(scores)
            
        # Run optimization
        result = gp_minimize(
            objective,
            param_space,
            n_calls=self.config.n_trials,
            random_state=42
        )
        
        # Update best parameters
        self.best_params = dict(zip(self.param_space.keys(), result.x))
        self.best_score = result.fun
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score
        }
        
    def _evolutionary_optimization(self,
                                 x: torch.Tensor,
                                 y: torch.Tensor,
                                 physics_equations: List[Callable]) -> Dict:
        """Evolutionary optimization"""
        from deap import base, creator, tools, algorithms
        
        # Create fitness and individual classes
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Initialize toolbox
        toolbox = base.Toolbox()
        
        # Register parameter sampling
        for param_name, param_values in self.param_space.items():
            if isinstance(param_values[0], int):
                toolbox.register(
                    f"attr_{param_name}",
                    np.random.randint,
                    param_values[0],
                    param_values[-1] + 1
                )
            elif isinstance(param_values[0], float):
                toolbox.register(
                    f"attr_{param_name}",
                    np.random.uniform,
                    param_values[0],
                    param_values[-1]
                )
            else:
                toolbox.register(
                    f"attr_{param_name}",
                    np.random.choice,
                    param_values
                )
                
        # Create individual and population
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            [getattr(toolbox, f"attr_{param_name}") for param_name in self.param_space.keys()],
            n=1
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Define evaluation function
        def evaluate(individual):
            param_dict = dict(zip(self.param_space.keys(), individual))
            scores = []
            
            # Cross-validation
            kf = KFold(n_splits=self.config.n_splits, shuffle=True)
            for train_idx, val_idx in kf.split(x):
                # Split data
                x_train, x_val = x[train_idx], x[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model = self.model_class(**param_dict)
                model.train(x_train, y_train, physics_equations)
                
                # Evaluate model
                y_pred = model.predict(x_val)
                score = self._compute_metric(y_val, y_pred)
                scores.append(score)
                
            return (np.mean(scores),)
            
        # Register genetic operators
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run optimization
        pop = toolbox.population(n=50)
        result, logbook = algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=0.7,
            mutpb=0.3,
            ngen=self.config.n_trials // 50,
            verbose=True
        )
        
        # Update best parameters
        best_individual = tools.selBest(result, k=1)[0]
        self.best_params = dict(zip(self.param_space.keys(), best_individual))
        self.best_score = best_individual.fitness.values[0]
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score
        }
        
    def _gradient_based_optimization(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   physics_equations: List[Callable]) -> Dict:
        """Gradient-based optimization"""
        # Convert parameters to tensors
        param_tensors = {}
        for param_name, param_values in self.param_space.items():
            if isinstance(param_values[0], (int, float)):
                param_tensors[param_name] = torch.tensor(
                    param_values[0],
                    requires_grad=True,
                    dtype=torch.float32
                )
                
        # Define optimizer
        optimizer = torch.optim.Adam(param_tensors.values(), lr=0.01)
        
        # Cross-validation
        kf = KFold(n_splits=self.config.n_splits, shuffle=True)
        
        for _ in range(self.config.n_trials):
            scores = []
            
            for train_idx, val_idx in kf.split(x):
                # Split data
                x_train, x_val = x[train_idx], x[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Convert parameters to dictionary
                params = {
                    name: tensor.item() if isinstance(tensor, torch.Tensor) else tensor
                    for name, tensor in param_tensors.items()
                }
                
                # Train model
                model = self.model_class(**params)
                model.train(x_train, y_train, physics_equations)
                
                # Evaluate model
                y_pred = model.predict(x_val)
                score = self._compute_metric(y_val, y_pred)
                scores.append(score)
                
            # Compute loss
            loss = torch.tensor(np.mean(scores), requires_grad=True)
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update best parameters
            if loss.item() < self.best_score:
                self.best_score = loss.item()
                self.best_params = {
                    name: tensor.item() if isinstance(tensor, torch.Tensor) else tensor
                    for name, tensor in param_tensors.items()
                }
                
        return {
            "best_params": self.best_params,
            "best_score": self.best_score
        }
        
    def _sample_parameters(self) -> Dict:
        """Sample random parameters"""
        params = {}
        for param_name, param_values in self.param_space.items():
            if isinstance(param_values[0], int):
                params[param_name] = np.random.randint(
                    param_values[0],
                    param_values[-1] + 1
                )
            elif isinstance(param_values[0], float):
                params[param_name] = np.random.uniform(
                    param_values[0],
                    param_values[-1]
                )
            else:
                params[param_name] = np.random.choice(param_values)
        return params
        
    def _compute_metric(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Compute evaluation metric"""
        if self.config.metric == "mse":
            return mean_squared_error(y_true.cpu(), y_pred.cpu())
        elif self.config.metric == "r2":
            return -r2_score(y_true.cpu(), y_pred.cpu())  # Negative because we minimize
        else:
            raise ValueError(f"Unknown metric: {self.config.metric}") 
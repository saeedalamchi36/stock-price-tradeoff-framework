# In benchmarks/setup_benchmarks.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Note: For Deep Learning models, you would need TensorFlow/Keras.
# To install tensorflow: pip install tensorflow
# For ARIMA, you would need statsmodels: pip install statsmodels

def get_benchmark_models():
    """
    Defines the benchmark models with the hyperparameters specified in the paper.
    This is for transparency and is not meant to be a fully executable training script.
    """
    
    # --- Statistical Model (ARIMA) ---
    # We define its configuration as a dictionary, similar to the DL models.
    arima_params = {
        'order': (5, 1, 0)  # (p, d, q) as specified in the paper's Table 2
    }
    
    # --- Scikit-learn based models ---
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,
        random_state=42 # for reproducibility
    )
    
    svr_model = SVR(
        kernel='rbf', 
        C=1.0, 
        gamma='auto',
        epsilon=0.1
    )
    
    # --- Conceptual definition for Deep Learning models ---
    # A full implementation would require building the model architecture using Keras/TensorFlow.
    # Here we just represent their hyperparameters.
    
    lstm_params = {
        'layers': 2,
        'neurons_per_layer': 50,
        'optimizer': 'Adam',
        'learning_rate': 0.001
    }
    
    transformer_params = {
        'encoder_layers': 4,
        'attention_heads': 8,
        'd_model': 256,
        'learning_rate': 0.0001
    }

    # We store everything in a dictionary for easy access
    models = {
        'ARIMA_config': arima_params,
        'Random Forest': rf_model,
        'SVR': svr_model,
        'LSTM_config': lstm_params,
        'Transformer_config': transformer_params
    }
    
    return models

# You can run this file directly to check if the definitions are correct
if __name__ == '__main__':
    benchmark_models = get_benchmark_models()
    print("Benchmark models defined successfully:")
    for name, config in benchmark_models.items():
        print(f"--- {name} ---")
        print(config)
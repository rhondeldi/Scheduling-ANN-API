"""
Neural Network Model Definitions
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from . import config


class FitnessPredictorModel:
    """
    ANN Model to predict schedule fitness scores
    """
    
    @staticmethod
    def build(input_dim: int = None) -> keras.Model:
        """
        Build fitness predictor neural network
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        if input_dim is None:
            input_dim = config.FITNESS_PREDICTOR_CONFIG['input_dim']
        
        cfg = config.FITNESS_PREDICTOR_CONFIG
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,), name='input'),
            
            # Hidden layer 1
            layers.Dense(cfg['hidden_layers'][0], name='dense_1'),
            layers.BatchNormalization(name='bn_1'),
            layers.Activation('relu', name='relu_1'),
            layers.Dropout(cfg['dropout_rate'], name='dropout_1'),
            
            # Hidden layer 2
            layers.Dense(cfg['hidden_layers'][1], name='dense_2'),
            layers.BatchNormalization(name='bn_2'),
            layers.Activation('relu', name='relu_2'),
            layers.Dropout(cfg['dropout_rate'], name='dropout_2'),
            
            # Hidden layer 3
            layers.Dense(cfg['hidden_layers'][2], name='dense_3'),
            layers.Activation('relu', name='relu_3'),
            
            # Output layer
            layers.Dense(1, activation='linear', name='output')
        ], name='FitnessPredictor')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
            loss='mse',
            metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        return model
    
    @staticmethod
    def build_summary():
        """Print model architecture"""
        model = FitnessPredictorModel.build()
        model.summary()
        return model


class ConstraintClassifierModel:
    """
    ANN Model to predict constraint violations.
    """

    @staticmethod
    def build(input_dim: int = None, num_constraints: int = None) -> keras.Model:
        if input_dim is None:
            input_dim = config.CONSTRAINT_CLASSIFIER_CONFIG['input_dim']

        if num_constraints is None:
            num_constraints = config.CONSTRAINT_CLASSIFIER_CONFIG['num_constraint_types']

        cfg = config.CONSTRAINT_CLASSIFIER_CONFIG

        inputs = layers.Input(shape=(input_dim,), name='input')

        x = layers.Dense(cfg['hidden_layers'][0], name='dense_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Activation('relu', name='relu_1')(x)
        x = layers.Dropout(cfg['dropout_rates'][0], name='dropout_1')(x)

        x = layers.Dense(cfg['hidden_layers'][1], name='dense_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Activation('relu', name='relu_2')(x)
        x = layers.Dropout(cfg['dropout_rates'][1], name='dropout_2')(x)

        x = layers.Dense(cfg['hidden_layers'][2], name='dense_3')(x)
        x = layers.Activation('relu', name='relu_3')(x)
        x = layers.Dropout(cfg['dropout_rates'][2], name='dropout_3')(x)

        outputs = layers.Dense(num_constraints, activation='sigmoid', name='output_constraints')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name='ConstraintClassifier')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
            loss='binary_crossentropy',
            metrics=[
                keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
            ],
        )
        return model

    @staticmethod
    def build_summary():
        model = ConstraintClassifierModel.build()
        model.summary()
        return model


class CrossoverRecommenderModel:
    """ANN model that predicts whether a (parent1, parent2) pair is
    compatible for crossover — i.e. whether the GA should attempt
    a crossover between them or skip the pair.

    Input  : 23 compatibility features (see scripts/train_crossover_recommender.py).
    Output : single sigmoid probability of compatibility.
    """

    @staticmethod
    def build(input_dim: int = None) -> keras.Model:
        cfg = config.CROSSOVER_RECOMMENDER_CONFIG
        if input_dim is None:
            input_dim = cfg['input_dim']

        model = models.Sequential([
            layers.Input(shape=(input_dim,), name='input'),

            layers.Dense(cfg['hidden_layers'][0], name='dense_1'),
            layers.BatchNormalization(name='bn_1'),
            layers.Activation('relu', name='relu_1'),
            layers.Dropout(cfg['dropout_rates'][0], name='dropout_1'),

            layers.Dense(cfg['hidden_layers'][1], activation='relu', name='dense_2'),
            layers.Dropout(cfg['dropout_rates'][1], name='dropout_2'),

            layers.Dense(1, activation='sigmoid', name='output'),
        ], name='CrossoverRecommender')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
            loss='binary_crossentropy',
            metrics=[
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
            ],
        )
        return model

    @staticmethod
    def build_summary():
        model = CrossoverRecommenderModel.build()
        model.summary()
        return model


class MutationPredictorModel:
    """
    ANN Model to predict mutation impact (improve/neutral/worsen).
    """

    @staticmethod
    def build(input_dim: int = None, num_classes: int = 3) -> keras.Model:
        if input_dim is None:
            input_dim = config.MUTATION_PREDICTOR_CONFIG['input_dim']

        cfg = config.MUTATION_PREDICTOR_CONFIG

        hidden = cfg['hidden_layers']
        seq = [
            layers.Input(shape=(input_dim,), name='input'),

            layers.Dense(hidden[0], name='dense_1'),
            layers.BatchNormalization(name='bn_1'),
            layers.Activation('relu', name='relu_1'),
            layers.Dropout(cfg['dropout_rate'], name='dropout_1'),

            layers.Dense(hidden[1], name='dense_2'),
            layers.BatchNormalization(name='bn_2'),
            layers.Activation('relu', name='relu_2'),
            layers.Dropout(cfg['dropout_rate'], name='dropout_2'),
        ]
        if len(hidden) >= 3:
            seq += [
                layers.Dense(hidden[2], activation='relu', name='dense_3'),
            ]
        seq.append(layers.Dense(num_classes, activation='softmax', name='output'))

        model = models.Sequential(seq, name='MutationPredictor')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
            loss='categorical_crossentropy',
            metrics=[
                keras.metrics.CategoricalAccuracy(name='accuracy'),
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
            ],
        )
        return model

    @staticmethod
    def build_summary():
        model = MutationPredictorModel.build()
        model.summary()
        return model


def load_model(model_path: str) -> keras.Model:
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Loaded Keras model
    """
    return keras.models.load_model(model_path)


def save_model(model: keras.Model, model_path: str):
    """
    Save a trained model to disk
    
    Args:
        model: Keras model to save
        model_path: Destination path
    """
    model.save(model_path)
    print(f"Model saved to: {model_path}")


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Building ANN Models for Scheduling System")
    print("=" * 70)
    
    print("\n1. Fitness Predictor Model:")
    print("-" * 70)
    fitness_model = FitnessPredictorModel.build_summary()
    print(f"Total parameters: {fitness_model.count_params():,}")

    print("\n2. Constraint Classifier Model:")
    print("-" * 70)
    constraint_model = ConstraintClassifierModel.build_summary()
    print(f"Total parameters: {constraint_model.count_params():,}")

    print("\n3. Crossover Recommender Model:")
    print("-" * 70)
    crossover_model = CrossoverRecommenderModel.build_summary()
    print(f"Total parameters: {crossover_model.count_params():,}")

    print("\n4. Mutation Predictor Model:")
    print("-" * 70)
    mutation_model = MutationPredictorModel.build_summary()
    print(f"Total parameters: {mutation_model.count_params():,}")
    
    print("\n" + "=" * 70)
    print("All models built successfully!")
    print("=" * 70)

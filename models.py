"""
Neural Network Model Definitions
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import config


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
    ANN Model to predict constraint violations
    """
    
    @staticmethod
    def build(input_dim: int = None, num_constraints: int = None) -> keras.Model:
        """
        Build constraint classifier network
        
        Args:
            input_dim: Number of input features
            num_constraints: Number of constraint types to predict
            
        Returns:
            Compiled Keras model
        """
        if input_dim is None:
            input_dim = config.CONSTRAINT_CLASSIFIER_CONFIG['input_dim']
        
        if num_constraints is None:
            num_constraints = config.CONSTRAINT_CLASSIFIER_CONFIG['num_constraint_types']
        
        cfg = config.CONSTRAINT_CLASSIFIER_CONFIG
        
        # Input
        inputs = layers.Input(shape=(input_dim,), name='input')
        
        # Hidden layer 1
        x = layers.Dense(cfg['hidden_layers'][0], name='dense_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Activation('relu', name='relu_1')(x)
        x = layers.Dropout(cfg['dropout_rates'][0], name='dropout_1')(x)
        
        # Hidden layer 2
        x = layers.Dense(cfg['hidden_layers'][1], name='dense_2')(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.Activation('relu', name='relu_2')(x)
        x = layers.Dropout(cfg['dropout_rates'][1], name='dropout_2')(x)
        
        # Hidden layer 3
        x = layers.Dense(cfg['hidden_layers'][2], name='dense_3')(x)
        x = layers.Activation('relu', name='relu_3')(x)
        x = layers.Dropout(cfg['dropout_rates'][2], name='dropout_3')(x)
        
        # Output layer - multi-label classification
        outputs = layers.Dense(
            num_constraints, 
            activation='sigmoid',
            name='output_constraints'
        )(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='ConstraintClassifier')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    @staticmethod
    def build_summary():
        """Print model architecture"""
        model = ConstraintClassifierModel.build()
        model.summary()
        return model


class CrossoverRecommenderModel:
    """
    ANN Model to recommend optimal crossover points
    """
    
    @staticmethod
    def build(sequence_length: int = 144, feature_dim: int = 3) -> keras.Model:
        """
        Build crossover point recommender using LSTM
        
        Args:
            sequence_length: Length of schedule sequence (time slots)
            feature_dim: Features per time slot
            
        Returns:
            Compiled Keras model
        """
        cfg = config.CROSSOVER_RECOMMENDER_CONFIG
        
        # Input: Two parent schedules
        parent1_input = layers.Input(
            shape=(sequence_length, feature_dim), 
            name='parent1'
        )
        parent2_input = layers.Input(
            shape=(sequence_length, feature_dim), 
            name='parent2'
        )
        
        # Concatenate parents
        combined = layers.Concatenate(axis=-1, name='concat_parents')([parent1_input, parent2_input])
        
        # LSTM to capture sequential patterns
        x = layers.LSTM(
            cfg['lstm_units'], 
            return_sequences=False,
            name='lstm'
        )(combined)
        x = layers.Dropout(cfg['dropout_rate'], name='dropout_lstm')(x)
        
        # Dense layers
        x = layers.Dense(cfg['dense_units'], activation='relu', name='dense')(x)
        x = layers.Dropout(cfg['dropout_rate'], name='dropout_dense')(x)
        
        # Output: Probability distribution over crossover points
        outputs = layers.Dense(
            sequence_length, 
            activation='softmax',
            name='crossover_probs'
        )(x)
        
        model = models.Model(
            inputs=[parent1_input, parent2_input], 
            outputs=outputs,
            name='CrossoverRecommender'
        )
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    @staticmethod
    def build_summary():
        """Print model architecture"""
        model = CrossoverRecommenderModel.build()
        model.summary()
        return model


class MutationPredictorModel:
    """
    ANN Model to predict mutation impact (improve/neutral/worsen)
    """
    
    @staticmethod
    def build(input_dim: int = None, num_classes: int = 3) -> keras.Model:
        """
        Build mutation impact predictor
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes (3: improve, neutral, worsen)
            
        Returns:
            Compiled Keras model
        """
        if input_dim is None:
            input_dim = config.MUTATION_PREDICTOR_CONFIG['input_dim']
        
        cfg = config.MUTATION_PREDICTOR_CONFIG
        
        model = models.Sequential([
            # Input
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
            
            # Output layer
            layers.Dense(num_classes, activation='softmax', name='output')
        ], name='MutationPredictor')
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    @staticmethod
    def build_summary():
        """Print model architecture"""
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

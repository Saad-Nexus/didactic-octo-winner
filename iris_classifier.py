import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

class IrisFlowerClassifier:
    """Multi-class classifier for Iris Flower Dataset"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.predictions = {}
        self.results = {}
        
    def load_data(self):
        """Load and prepare Iris dataset"""
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Standardize features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("Data loaded and preprocessed successfully!")
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
    def train_decision_tree(self):
        """Train Decision Tree Classifier"""
        self.models['Decision Tree'] = DecisionTreeClassifier(random_state=self.random_state)
        self.models['Decision Tree'].fit(self.X_train, self.y_train)
        self.predictions['Decision Tree'] = self.models['Decision Tree'].predict(self.X_test)
        print("✓ Decision Tree trained")
        
    def train_random_forest(self):
        """Train Random Forest Classifier"""
        self.models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        self.models['Random Forest'].fit(self.X_train, self.y_train)
        self.predictions['Random Forest'] = self.models['Random Forest'].predict(self.X_test)
        print("✓ Random Forest trained")
        
    def train_svm(self):
        """Train Support Vector Machine"""
        self.models['SVM'] = SVC(kernel='rbf', random_state=self.random_state)
        self.models['SVM'].fit(self.X_train, self.y_train)
        self.predictions['SVM'] = self.models['SVM'].predict(self.X_test)
        print("✓ SVM trained")
        
    def train_naive_bayes(self):
        """Train Naive Bayes Classifier"""
        self.models['Naive Bayes'] = GaussianNB()
        self.models['Naive Bayes'].fit(self.X_train, self.y_train)
        self.predictions['Naive Bayes'] = self.models['Naive Bayes'].predict(self.X_test)
        print("✓ Naive Bayes trained")
        
    def train_knn(self):
        """Train K-Nearest Neighbors"""
        self.models['KNN'] = KNeighborsClassifier(n_neighbors=5)
        self.models['KNN'].fit(self.X_train, self.y_train)
        self.predictions['KNN'] = self.models['KNN'].predict(self.X_test)
        print("✓ KNN trained")
        
    def train_neural_network(self):
        """Train Keras Sequential Neural Network"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(4,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        self.models['Neural Network'] = model
        self.history = model.fit(self.X_train, self.y_train, 
                                 validation_split=0.2, 
                                 epochs=100, 
                                 batch_size=8,
                                 verbose=0)
        self.predictions['Neural Network'] = np.argmax(model.predict(self.X_test, verbose=0), axis=1)
        print("✓ Neural Network trained")
        
    def train_all_models(self):
        """Train all classifiers"""
        print("\nTraining all models...")
        self.train_decision_tree()
        self.train_random_forest()
        self.train_svm()
        self.train_naive_bayes()
        self.train_knn()
        self.train_neural_network()
        print("\nAll models trained successfully!\n")
        
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("Model Evaluation Results:")
        print("=" * 80)
        
        for model_name, predictions in self.predictions.items():
            accuracy = accuracy_score(self.y_test, predictions)
            precision = precision_score(self.y_test, predictions, average='weighted')
            recall = recall_score(self.y_test, predictions, average='weighted')
            f1 = f1_score(self.y_test, predictions, average='weighted')
            
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
    def print_detailed_reports(self):
        """Print detailed classification reports"""
        print("\n" + "=" * 80)
        print("Detailed Classification Reports")
        print("=" * 80)
        
        for model_name, predictions in self.predictions.items():
            print(f"\n{model_name}:")
            print(classification_report(self.y_test, predictions, 
                                       target_names=self.target_names))

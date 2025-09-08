"""
Model Manager GUI - Main Application
"""

import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, 
                            QHBoxLayout, QWidget, QLabel, QPushButton, QTextEdit,
                            QFileDialog, QMessageBox, QProgressBar, QComboBox,
                            QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
                            QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QIcon

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from versioning.registry_manager import ModelRegistry
from training.train_sign import train_model as train_sign_model
from training.train_braille import train_model as train_braille_model
from prediction.predict_sign import SignLanguagePredictor
from prediction.predict_braille import BraillePredictor

class TrainingThread(QThread):
    """Thread for model training"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, model_type, data_dir, model_dir, **kwargs):
        super().__init__()
        self.model_type = model_type
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.kwargs = kwargs
    
    def run(self):
        try:
            self.status.emit(f"Training {self.model_type} model...")
            
            if self.model_type == "sign_language":
                model_path = train_sign_model(
                    data_dir=self.data_dir,
                    model_dir=self.model_dir,
                    **self.kwargs
                )
            elif self.model_type == "braille":
                model_path = train_braille_model(
                    data_dir=self.data_dir,
                    model_dir=self.model_dir,
                    **self.kwargs
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            if model_path:
                self.finished.emit(model_path)
            else:
                self.error.emit("Training failed")
                
        except Exception as e:
            self.error.emit(str(e))

class HomeTab(QWidget):
    """Home tab with overview and quick actions"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.update_overview()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("🏠 Model Manager - Home")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Overview section
        overview_group = QGroupBox("📊 Current Status")
        overview_layout = QVBoxLayout()
        
        self.overview_text = QTextEdit()
        self.overview_text.setMaximumHeight(150)
        self.overview_text.setReadOnly(True)
        overview_layout.addWidget(self.overview_text)
        
        overview_group.setLayout(overview_layout)
        layout.addWidget(overview_group)
        
        # Quick actions
        actions_group = QGroupBox("⚡ Quick Actions")
        actions_layout = QGridLayout()
        
        # Buttons
        self.collect_data_btn = QPushButton("📊 Collect Data")
        self.collect_data_btn.clicked.connect(self.collect_data)
        
        self.train_model_btn = QPushButton("🧠 Train Model")
        self.train_model_btn.clicked.connect(self.train_model)
        
        self.test_model_btn = QPushButton("🔍 Test Model")
        self.test_model_btn.clicked.connect(self.test_model)
        
        self.view_models_btn = QPushButton("📋 View Models")
        self.view_models_btn.clicked.connect(self.view_models)
        
        actions_layout.addWidget(self.collect_data_btn, 0, 0)
        actions_layout.addWidget(self.train_model_btn, 0, 1)
        actions_layout.addWidget(self.test_model_btn, 1, 0)
        actions_layout.addWidget(self.view_models_btn, 1, 1)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.update_overview)
        layout.addWidget(refresh_btn)
        
        self.setLayout(layout)
    
    def update_overview(self):
        """Update overview information"""
        try:
            registry = ModelRegistry()
            stats = registry.get_model_statistics()
            
            overview = "📊 MODEL OVERVIEW\n"
            overview += "=" * 30 + "\n\n"
            
            overview += f"Total Models: {stats['total_models']}\n"
            overview += f"Sign Language Models: {stats['by_type'].get('sign_language', 0)}\n"
            overview += f"Braille Models: {stats['by_type'].get('braille', 0)}\n\n"
            
            # Latest versions
            if stats['latest_versions']['sign_language']:
                sign_info = registry.get_model_info('sign_language', stats['latest_versions']['sign_language'])
                overview += f"Latest Sign Model: {stats['latest_versions']['sign_language']}\n"
                overview += f"  Accuracy: {sign_info['accuracy']:.2%}\n"
                overview += f"  Trained: {sign_info['trained_on']}\n\n"
            
            if stats['latest_versions']['braille']:
                braille_info = registry.get_model_info('braille', stats['latest_versions']['braille'])
                overview += f"Latest Braille Model: {stats['latest_versions']['braille']}\n"
                overview += f"  Accuracy: {braille_info['accuracy']:.2%}\n"
                overview += f"  Trained: {braille_info['trained_on']}\n"
            
            self.overview_text.setPlainText(overview)
            
        except Exception as e:
            self.overview_text.setPlainText(f"Error loading overview: {e}")
    
    def collect_data(self):
        """Open data collection dialog"""
        QMessageBox.information(self, "Data Collection", 
                               "Please use the Data Collection menu to collect data.")
    
    def train_model(self):
        """Open training dialog"""
        QMessageBox.information(self, "Model Training", 
                               "Please use the Training tab to train models.")
    
    def test_model(self):
        """Open testing dialog"""
        QMessageBox.information(self, "Model Testing", 
                               "Please use the Testing tab to test models.")
    
    def view_models(self):
        """Open model versions dialog"""
        QMessageBox.information(self, "Model Versions", 
                               "Please use the Model Versions tab to view models.")

class DatasetTab(QWidget):
    """Dataset management tab"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("📊 Dataset Management")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Dataset info
        info_group = QGroupBox("📁 Dataset Information")
        info_layout = QVBoxLayout()
        
        self.dataset_info = QTextEdit()
        self.dataset_info.setMaximumHeight(200)
        self.dataset_info.setReadOnly(True)
        info_layout.addWidget(self.dataset_info)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Actions
        actions_group = QGroupBox("🔧 Actions")
        actions_layout = QHBoxLayout()
        
        self.collect_sign_btn = QPushButton("🎬 Collect Sign Language Data")
        self.collect_sign_btn.clicked.connect(self.collect_sign_data)
        
        self.collect_braille_btn = QPushButton("🔤 Collect Braille Data")
        self.collect_braille_btn.clicked.connect(self.collect_braille_data)
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.update_dataset_info)
        
        actions_layout.addWidget(self.collect_sign_btn)
        actions_layout.addWidget(self.collect_braille_btn)
        actions_layout.addWidget(self.refresh_btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        self.setLayout(layout)
        self.update_dataset_info()
    
    def update_dataset_info(self):
        """Update dataset information"""
        try:
            info = "📁 DATASET INFORMATION\n"
            info += "=" * 30 + "\n\n"
            
            # Check sign language dataset
            sign_dir = "datasets/sign_language"
            if os.path.exists(sign_dir):
                classes = [d for d in os.listdir(sign_dir) 
                          if os.path.isdir(os.path.join(sign_dir, d))]
                info += f"Sign Language Dataset:\n"
                info += f"  Classes: {len(classes)}\n"
                for cls in classes:
                    class_dir = os.path.join(sign_dir, cls)
                    files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
                    info += f"    {cls}: {len(files)} samples\n"
            else:
                info += "Sign Language Dataset: Not found\n"
            
            info += "\n"
            
            # Check braille dataset
            braille_dir = "datasets/braille"
            if os.path.exists(braille_dir):
                classes = [d for d in os.listdir(braille_dir) 
                          if os.path.isdir(os.path.join(braille_dir, d))]
                info += f"Braille Dataset:\n"
                info += f"  Classes: {len(classes)}\n"
                for cls in classes:
                    class_dir = os.path.join(braille_dir, cls)
                    files = [f for f in os.listdir(class_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg', '.npy'))]
                    info += f"    {cls}: {len(files)} samples\n"
            else:
                info += "Braille Dataset: Not found\n"
            
            self.dataset_info.setPlainText(info)
            
        except Exception as e:
            self.dataset_info.setPlainText(f"Error loading dataset info: {e}")
    
    def collect_sign_data(self):
        """Start sign language data collection"""
        QMessageBox.information(self, "Sign Language Data Collection", 
                               "Please run: python data_collection/collect_sign_data.py")
    
    def collect_braille_data(self):
        """Start braille data collection"""
        QMessageBox.information(self, "Braille Data Collection", 
                               "Please run: python data_collection/collect_braille_data.py")

class TrainingTab(QWidget):
    """Model training tab"""
    
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("🧠 Model Training")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Model type selection
        model_group = QGroupBox("🎯 Model Configuration")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_type = QComboBox()
        self.model_type.addItems(["sign_language", "braille"])
        model_layout.addWidget(self.model_type, 0, 1)
        
        model_layout.addWidget(QLabel("Data Directory:"), 1, 0)
        self.data_dir = QLabel("datasets/sign_language")
        self.data_dir.setStyleSheet("border: 1px solid gray; padding: 5px;")
        model_layout.addWidget(self.data_dir, 1, 1)
        
        browse_btn = QPushButton("📁 Browse")
        browse_btn.clicked.connect(self.browse_data_dir)
        model_layout.addWidget(browse_btn, 1, 2)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Training parameters
        params_group = QGroupBox("⚙️ Training Parameters")
        params_layout = QGridLayout()
        
        params_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(20)
        params_layout.addWidget(self.epochs, 0, 1)
        
        params_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(8)
        params_layout.addWidget(self.batch_size, 0, 3)
        
        params_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 0.1)
        self.learning_rate.setValue(0.001)
        self.learning_rate.setDecimals(4)
        params_layout.addWidget(self.learning_rate, 1, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Training controls
        controls_group = QGroupBox("🎮 Training Controls")
        controls_layout = QVBoxLayout()
        
        self.train_btn = QPushButton("🚀 Start Training")
        self.train_btn.clicked.connect(self.start_training)
        controls_layout.addWidget(self.train_btn)
        
        self.progress_bar = QProgressBar()
        controls_layout.addWidget(self.progress_bar)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        controls_layout.addWidget(self.status_text)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        self.setLayout(layout)
        
        # Connect model type change
        self.model_type.currentTextChanged.connect(self.update_data_dir)
    
    def update_data_dir(self, model_type):
        """Update data directory based on model type"""
        if model_type == "sign_language":
            self.data_dir.setText("datasets/sign_language")
        else:
            self.data_dir.setText("datasets/braille")
    
    def browse_data_dir(self):
        """Browse for data directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.data_dir.setText(directory)
    
    def start_training(self):
        """Start model training"""
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Training in Progress", 
                               "Training is already in progress. Please wait.")
            return
        
        # Get parameters
        model_type = self.model_type.currentText()
        data_dir = self.data_dir.text()
        
        if not os.path.exists(data_dir):
            QMessageBox.warning(self, "Directory Not Found", 
                               f"Data directory not found: {data_dir}")
            return
        
        # Training parameters
        kwargs = {
            'epochs': self.epochs.value(),
            'batch_size': self.batch_size.value(),
            'learning_rate': self.learning_rate.value()
        }
        
        # Start training thread
        self.training_thread = TrainingThread(
            model_type=model_type,
            data_dir=data_dir,
            model_dir=f"models/{model_type}",
            **kwargs
        )
        
        self.training_thread.progress.connect(self.progress_bar.setValue)
        self.training_thread.status.connect(self.update_status)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.error.connect(self.training_error)
        
        self.training_thread.start()
        
        self.train_btn.setEnabled(False)
        self.train_btn.setText("🔄 Training...")
    
    def update_status(self, message):
        """Update status message"""
        self.status_text.append(message)
    
    def training_finished(self, model_path):
        """Handle training completion"""
        self.train_btn.setEnabled(True)
        self.train_btn.setText("🚀 Start Training")
        self.progress_bar.setValue(100)
        
        QMessageBox.information(self, "Training Complete", 
                               f"Model trained successfully!\nSaved to: {model_path}")
    
    def training_error(self, error_message):
        """Handle training error"""
        self.train_btn.setEnabled(True)
        self.train_btn.setText("🚀 Start Training")
        
        QMessageBox.critical(self, "Training Error", f"Training failed: {error_message}")

class TestingTab(QWidget):
    """Model testing tab"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("🔍 Model Testing")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Model selection
        model_group = QGroupBox("🎯 Model Selection")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_type = QComboBox()
        self.model_type.addItems(["sign_language", "braille"])
        model_layout.addWidget(self.model_type, 0, 1)
        
        model_layout.addWidget(QLabel("Model Version:"), 1, 0)
        self.model_version = QComboBox()
        model_layout.addWidget(self.model_version, 1, 1)
        
        refresh_models_btn = QPushButton("🔄 Refresh")
        refresh_models_btn.clicked.connect(self.refresh_models)
        model_layout.addWidget(refresh_models_btn, 1, 2)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Testing controls
        controls_group = QGroupBox("🎮 Testing Controls")
        controls_layout = QVBoxLayout()
        
        self.test_btn = QPushButton("🔍 Start Test")
        self.test_btn.clicked.connect(self.start_test)
        controls_layout.addWidget(self.test_btn)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        controls_layout.addWidget(self.results_text)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        self.setLayout(layout)
        self.refresh_models()
    
    def refresh_models(self):
        """Refresh available models"""
        try:
            registry = ModelRegistry()
            model_type = self.model_type.currentText()
            models = registry.list_models(model_type)
            
            self.model_version.clear()
            for version in models.keys():
                self.model_version.addItem(version)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh models: {e}")
    
    def start_test(self):
        """Start model testing"""
        model_type = self.model_type.currentText()
        version = self.model_version.currentText()
        
        if not version:
            QMessageBox.warning(self, "No Model Selected", 
                               "Please select a model version.")
            return
        
        try:
            registry = ModelRegistry()
            model_info = registry.get_model_info(model_type, version)
            
            if not model_info:
                QMessageBox.warning(self, "Model Not Found", 
                                   f"Model {version} not found.")
                return
            
            model_path = model_info['path']
            
            if model_type == "sign_language":
                QMessageBox.information(self, "Sign Language Testing", 
                                       f"Please run: python prediction/predict_sign.py --model_path {model_path}")
            else:
                QMessageBox.information(self, "Braille Testing", 
                                       f"Please run: python prediction/predict_braille.py --model_path {model_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start test: {e}")

class ModelVersionsTab(QWidget):
    """Model versions management tab"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.refresh_models()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("📋 Model Versions")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Models table
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(6)
        self.models_table.setHorizontalHeaderLabels([
            "Model Type", "Version", "Accuracy", "Trained On", "Path", "Actions"
        ])
        
        # Set table properties
        header = self.models_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        layout.addWidget(self.models_table)
        
        # Actions
        actions_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_models)
        actions_layout.addWidget(refresh_btn)
        
        delete_btn = QPushButton("🗑️ Delete Selected")
        delete_btn.clicked.connect(self.delete_selected_model)
        actions_layout.addWidget(delete_btn)
        
        actions_layout.addStretch()
        layout.addLayout(actions_layout)
        
        self.setLayout(layout)
    
    def refresh_models(self):
        """Refresh models table"""
        try:
            registry = ModelRegistry()
            all_models = registry.list_models()
            
            # Count total models
            total_models = sum(len(models) for models in all_models.values())
            self.models_table.setRowCount(total_models)
            
            row = 0
            for model_type, models in all_models.items():
                for version, info in models.items():
                    self.models_table.setItem(row, 0, QTableWidgetItem(model_type))
                    self.models_table.setItem(row, 1, QTableWidgetItem(version))
                    self.models_table.setItem(row, 2, QTableWidgetItem(f"{info['accuracy']:.2%}"))
                    self.models_table.setItem(row, 3, QTableWidgetItem(info['trained_on']))
                    self.models_table.setItem(row, 4, QTableWidgetItem(info['path']))
                    
                    # Action button
                    action_btn = QPushButton("Use")
                    action_btn.clicked.connect(lambda checked, mt=model_type, v=version: self.use_model(mt, v))
                    self.models_table.setCellWidget(row, 5, action_btn)
                    
                    row += 1
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh models: {e}")
    
    def use_model(self, model_type, version):
        """Use selected model"""
        QMessageBox.information(self, "Use Model", 
                               f"Using {model_type} model version {version}")
    
    def delete_selected_model(self):
        """Delete selected model"""
        current_row = self.models_table.currentRow()
        if current_row < 0:
            QMessageBox.warning(self, "No Selection", 
                               "Please select a model to delete.")
            return
        
        model_type = self.models_table.item(current_row, 0).text()
        version = self.models_table.item(current_row, 1).text()
        
        reply = QMessageBox.question(self, "Confirm Delete", 
                                   f"Are you sure you want to delete {model_type} model version {version}?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                registry = ModelRegistry()
                if registry.delete_model(model_type, version):
                    QMessageBox.information(self, "Success", "Model deleted successfully.")
                    self.refresh_models()
                else:
                    QMessageBox.warning(self, "Error", "Failed to delete model.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete model: {e}")

class ModelManagerGUI(QMainWindow):
    """Main GUI application"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Model Manager")
        self.setGeometry(100, 100, 1000, 700)
        
        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Add tabs
        self.tab_widget.addTab(HomeTab(), "🏠 Home")
        self.tab_widget.addTab(DatasetTab(), "📊 Datasets")
        self.tab_widget.addTab(TrainingTab(), "🧠 Train")
        self.tab_widget.addTab(TestingTab(), "🔍 Test")
        self.tab_widget.addTab(ModelVersionsTab(), "📋 Model Versions")
        
        layout.addWidget(self.tab_widget)
        central_widget.setLayout(layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = ModelManagerGUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

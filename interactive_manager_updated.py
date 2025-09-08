"""
Interactive Model Manager - Command Line Interface (Updated)
"""

import os
import sys
import argparse
import subprocess
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from versioning.registry_manager import ModelRegistry
from training.train_sign import train_model as train_sign_model
from training.train_braille import train_model as train_braille_model
from prediction.predict_sign import SignLanguagePredictor
from prediction.predict_braille import BraillePredictor

class InteractiveManager:
    def __init__(self):
        self.registry = ModelRegistry()
    
    def show_menu(self):
        """Show main menu"""
        print("\n" + "="*50)
        print("🧠 MODEL MANAGER - Interactive Mode")
        print("="*50)
        print("1. 📊 Collect Data")
        print("2. 🧠 Train Model")
        print("3. 🔍 Test Model")
        print("4. 📋 List Models")
        print("5. 📊 Show Statistics")
        print("6. 🗑️ Delete Model")
        print("0. 🚪 Exit")
        print("="*50)
    
    def collect_data(self):
        """Data collection menu with direct execution"""
        print("\n📊 DATA COLLECTION")
        print("-" * 30)
        print("1. 🎬 Sign Language Data (Webcam + MediaPipe)")
        print("2. �� Braille Data (Images + Manual Input)")
        print("3. 📋 Unified Data Collection Menu")
        print("0. 🔙 Back")
        
        choice = input("\nEnter choice (0-3): ").strip()
        
        if choice == '1':
            print("\n🎬 Starting Sign Language Data Collection...")
            try:
                # Import and run the collector directly
                from data_collection.collect_sign_data import SignLanguageDataCollector
                collector = SignLanguageDataCollector()
                collector.interactive_collection()
            except Exception as e:
                print(f"❌ Error starting sign language data collection: {e}")
                print("Falling back to manual command...")
                print("Run: python data_collection/collect_sign_data.py")
                
        elif choice == '2':
            print("\n🔤 Starting Braille Data Collection...")
            try:
                # Import and run the collector directly
                from data_collection.collect_braille_data import BrailleDataCollector
                collector = BrailleDataCollector()
                collector.interactive_collection()
            except Exception as e:
                print(f"❌ Error starting braille data collection: {e}")
                print("Falling back to manual command...")
                print("Run: python data_collection/collect_braille_data.py")
                
        elif choice == '3':
            print("\n📋 Starting Unified Data Collection Menu...")
            try:
                # Run the unified menu
                subprocess.run([sys.executable, "data_collection_menu.py"])
            except Exception as e:
                print(f"❌ Error starting unified menu: {e}")
                print("Falling back to manual command...")
                print("Run: python data_collection_menu.py")
                
        elif choice == '0':
            return
        else:
            print("❌ Invalid choice")
    
    def train_model(self):
        """Model training menu"""
        print("\n🧠 MODEL TRAINING")
        print("-" * 30)
        print("1. 🎬 Sign Language Model")
        print("2. 🔤 Braille Model")
        print("0. 🔙 Back")
        
        choice = input("\nEnter choice (0-2): ").strip()
        
        if choice == '1':
            self._train_sign_model()
        elif choice == '2':
            self._train_braille_model()
        elif choice == '0':
            return
        else:
            print("❌ Invalid choice")
    
    def _train_sign_model(self):
        """Train sign language model"""
        print("\n🎬 TRAINING SIGN LANGUAGE MODEL")
        print("-" * 40)
        
        # Get parameters
        data_dir = input("Data directory [datasets/sign_language]: ").strip() or "datasets/sign_language"
        epochs = int(input("Epochs [20]: ").strip() or "20")
        batch_size = int(input("Batch size [8]: ").strip() or "8")
        learning_rate = float(input("Learning rate [0.001]: ").strip() or "0.001")
        
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return
        
        print(f"\n🚀 Starting training...")
        print(f"Data directory: {data_dir}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        try:
            model_path = train_sign_model(
                data_dir=data_dir,
                model_dir="models/sign_language",
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            if model_path:
                print(f"✅ Training completed successfully!")
                print(f"Model saved to: {model_path}")
            else:
                print("❌ Training failed!")
                
        except Exception as e:
            print(f"❌ Training error: {e}")
    
    def _train_braille_model(self):
        """Train braille model"""
        print("\n🔤 TRAINING BRAILLE MODEL")
        print("-" * 30)
        
        # Get parameters
        data_dir = input("Data directory [datasets/braille]: ").strip() or "datasets/braille"
        epochs = int(input("Epochs [20]: ").strip() or "20")
        batch_size = int(input("Batch size [16]: ").strip() or "16")
        learning_rate = float(input("Learning rate [0.001]: ").strip() or "0.001")
        
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return
        
        print(f"\n🚀 Starting training...")
        print(f"Data directory: {data_dir}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        try:
            model_path = train_braille_model(
                data_dir=data_dir,
                model_dir="models/braille",
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            if model_path:
                print(f"✅ Training completed successfully!")
                print(f"Model saved to: {model_path}")
            else:
                print("❌ Training failed!")
                
        except Exception as e:
            print(f"❌ Training error: {e}")
    
    def test_model(self):
        """Model testing menu"""
        print("\n🔍 MODEL TESTING")
        print("-" * 30)
        print("1. 🎬 Sign Language Model")
        print("2. 🔤 Braille Model")
        print("0. 🔙 Back")
        
        choice = input("\nEnter choice (0-2): ").strip()
        
        if choice == '1':
            self._test_sign_model()
        elif choice == '2':
            self._test_braille_model()
        elif choice == '0':
            return
        else:
            print("❌ Invalid choice")
    
    def _test_sign_model(self):
        """Test sign language model"""
        print("\n🎬 TESTING SIGN LANGUAGE MODEL")
        print("-" * 40)
        
        # List available models
        models = self.registry.list_models('sign_language')
        if not models:
            print("❌ No sign language models found. Train a model first.")
            return
        
        print("Available models:")
        for i, (version, info) in enumerate(models.items(), 1):
            print(f"{i}. {version} (accuracy: {info['accuracy']:.2%})")
        
        try:
            choice = int(input("\nSelect model (number): ")) - 1
            version = list(models.keys())[choice]
            model_info = models[version]
            
            print(f"\nUsing model: {version}")
            print(f"Path: {model_info['path']}")
            
            # Test options
            print("\nTest options:")
            print("1. Real-time webcam")
            print("2. From file")
            
            test_choice = input("Enter choice (1-2): ").strip()
            
            if test_choice == '1':
                print("\n🎬 Starting real-time prediction...")
                try:
                    # Run prediction directly
                    subprocess.run([
                        sys.executable, "prediction/predict_sign.py",
                        "--model_path", model_info['path'],
                        "--input", "realtime"
                    ])
                except Exception as e:
                    print(f"❌ Error running prediction: {e}")
                    print(f"Manual command: python prediction/predict_sign.py --model_path {model_info['path']} --input realtime")
                    
            elif test_choice == '2':
                file_path = input("Enter file path: ").strip()
                if file_path:
                    print(f"\n🔍 Testing from file: {file_path}")
                    try:
                        subprocess.run([
                            sys.executable, "prediction/predict_sign.py",
                            "--model_path", model_info['path'],
                            "--input", "file",
                            "--file_path", file_path
                        ])
                    except Exception as e:
                        print(f"❌ Error running prediction: {e}")
                        print(f"Manual command: python prediction/predict_sign.py --model_path {model_info['path']} --input file --file_path {file_path}")
            else:
                print("❌ Invalid choice")
                
        except (ValueError, IndexError):
            print("❌ Invalid selection")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _test_braille_model(self):
        """Test braille model"""
        print("\n🔤 TESTING BRAILLE MODEL")
        print("-" * 30)
        
        # List available models
        models = self.registry.list_models('braille')
        if not models:
            print("❌ No braille models found. Train a model first.")
            return
        
        print("Available models:")
        for i, (version, info) in enumerate(models.items(), 1):
            print(f"{i}. {version} (accuracy: {info['accuracy']:.2%})")
        
        try:
            choice = int(input("\nSelect model (number): ")) - 1
            version = list(models.keys())[choice]
            model_info = models[version]
            
            print(f"\nUsing model: {version}")
            print(f"Path: {model_info['path']}")
            
            # Test options
            print("\nTest options:")
            print("1. From image")
            print("2. Real-time webcam")
            print("3. Manual input")
            
            test_choice = input("Enter choice (1-3): ").strip()
            
            if test_choice == '1':
                image_path = input("Enter image path: ").strip()
                if image_path:
                    print(f"\n🖼️ Testing from image: {image_path}")
                    try:
                        subprocess.run([
                            sys.executable, "prediction/predict_braille.py",
                            "--model_path", model_info['path'],
                            "--input", "image",
                            "--image_path", image_path
                        ])
                    except Exception as e:
                        print(f"❌ Error running prediction: {e}")
                        print(f"Manual command: python prediction/predict_braille.py --model_path {model_info['path']} --input image --image_path {image_path}")
                        
            elif test_choice == '2':
                print("\n🎬 Starting real-time prediction...")
                try:
                    subprocess.run([
                        sys.executable, "prediction/predict_braille.py",
                        "--model_path", model_info['path'],
                        "--input", "realtime"
                    ])
                except Exception as e:
                    print(f"❌ Error running prediction: {e}")
                    print(f"Manual command: python prediction/predict_braille.py --model_path {model_info['path']} --input realtime")
                    
            elif test_choice == '3':
                pattern = input("Enter 6-dot pattern (e.g., 1,0,0,0,0,0): ").strip()
                if pattern:
                    print(f"\n⌨️ Testing manual input: {pattern}")
                    try:
                        subprocess.run([
                            sys.executable, "prediction/predict_braille.py",
                            "--input", "manual",
                            "--pattern", pattern
                        ])
                    except Exception as e:
                        print(f"❌ Error running prediction: {e}")
                        print(f"Manual command: python prediction/predict_braille.py --input manual --pattern {pattern}")
            else:
                print("❌ Invalid choice")
                
        except (ValueError, IndexError):
            print("❌ Invalid selection")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def list_models(self):
        """List all models"""
        print("\n📋 MODEL VERSIONS")
        print("-" * 30)
        
        all_models = self.registry.list_models()
        
        if not any(all_models.values()):
            print("No models found. Train some models first.")
            return
        
        for model_type, models in all_models.items():
            if models:
                print(f"\n{model_type.upper()}:")
                for version, info in models.items():
                    print(f"  {version}:")
                    print(f"    Accuracy: {info['accuracy']:.2%}")
                    print(f"    Trained: {info['trained_on']}")
                    print(f"    Path: {info['path']}")
    
    def show_statistics(self):
        """Show model statistics"""
        print("\n📊 MODEL STATISTICS")
        print("-" * 30)
        
        stats = self.registry.get_model_statistics()
        
        print(f"Total Models: {stats['total_models']}")
        print(f"Sign Language Models: {stats['by_type'].get('sign_language', 0)}")
        print(f"Braille Models: {stats['by_type'].get('braille', 0)}")
        
        if stats['latest_versions']['sign_language']:
            sign_info = self.registry.get_model_info('sign_language', stats['latest_versions']['sign_language'])
            print(f"\nLatest Sign Model: {stats['latest_versions']['sign_language']}")
            print(f"  Accuracy: {sign_info['accuracy']:.2%}")
            print(f"  Trained: {sign_info['trained_on']}")
        
        if stats['latest_versions']['braille']:
            braille_info = self.registry.get_model_info('braille', stats['latest_versions']['braille'])
            print(f"\nLatest Braille Model: {stats['latest_versions']['braille']}")
            print(f"  Accuracy: {braille_info['accuracy']:.2%}")
            print(f"  Trained: {braille_info['trained_on']}")
    
    def delete_model(self):
        """Delete model menu"""
        print("\n🗑️ DELETE MODEL")
        print("-" * 30)
        
        all_models = self.registry.list_models()
        
        if not any(all_models.values()):
            print("No models found.")
            return
        
        # List all models
        model_list = []
        for model_type, models in all_models.items():
            for version in models.keys():
                model_list.append((model_type, version))
        
        print("Available models:")
        for i, (model_type, version) in enumerate(model_list, 1):
            print(f"{i}. {model_type} - {version}")
        
        try:
            choice = int(input("\nSelect model to delete (number): ")) - 1
            model_type, version = model_list[choice]
            
            confirm = input(f"Are you sure you want to delete {model_type} model {version}? (y/n): ").strip().lower()
            
            if confirm == 'y':
                if self.registry.delete_model(model_type, version):
                    print("✅ Model deleted successfully.")
                else:
                    print("❌ Failed to delete model.")
            else:
                print("❌ Deletion cancelled.")
                
        except (ValueError, IndexError):
            print("❌ Invalid selection")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def run(self):
        """Run interactive manager"""
        while True:
            self.show_menu()
            choice = input("\nEnter choice (0-6): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == '1':
                self.collect_data()
            elif choice == '2':
                self.train_model()
            elif choice == '3':
                self.test_model()
            elif choice == '4':
                self.list_models()
            elif choice == '5':
                self.show_statistics()
            elif choice == '6':
                self.delete_model()
            else:
                print("❌ Invalid choice. Please try again.")

def main():
    parser = argparse.ArgumentParser(description='Model Manager - Interactive Mode (Updated)')
    parser.add_argument('--gui', action='store_true', help='Launch GUI instead of CLI')
    
    args = parser.parse_args()
    
    if args.gui:
        print("🚀 Launching GUI...")
        try:
            from gui.main import main as gui_main
            gui_main()
        except ImportError as e:
            print(f"❌ GUI not available: {e}")
            print("Falling back to CLI mode...")
            manager = InteractiveManager()
            manager.run()
    else:
        manager = InteractiveManager()
        manager.run()

if __name__ == "__main__":
    main()

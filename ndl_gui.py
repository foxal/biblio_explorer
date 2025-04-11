import sys
import os
import threading
import logging
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                           QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
                           QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                           QGroupBox, QFormLayout, QProgressBar, QMessageBox, QFrame,
                           QInputDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer

# Import wrappers for the NDL scripts
from ndl_search_wrapper import NDLSearchWrapper
from data_processor_wrapper import BiblioDataProcessorWrapper
from cinii_search_wrapper import CiNiiSearchWrapper

class LogHandler(QObject, logging.Handler):
    """Custom logging handler that emits log messages as signals."""
    log_message = pyqtSignal(str, int)  # Message and log level
    
    def __init__(self):
        super().__init__()
        # Set initial level to DEBUG to capture everything
        self.setLevel(logging.DEBUG)
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
    def emit(self, record):
        try:
            # Format the record
            msg = self.format(record)
            
            # Pass both the message and the log level
            self.log_message.emit(msg, record.levelno)
        except Exception as e:
            # Fallback in case of formatting error
            print(f"Error in LogHandler.emit: {str(e)}")
            try:
                self.log_message.emit(f"Error formatting log message: {str(e)}", logging.ERROR)
            except:
                pass  # Last resort if even the error signal fails

class SearchTab(QWidget):
    """Tab for NDL Search functionality."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize the wrapper
        self.search_wrapper = NDLSearchWrapper()
        
        # Set up logging
        self.log_handler = LogHandler()
        # Connect to signal with 2 arguments (message, level)
        self.log_handler.log_message.connect(self.append_log)
        
        logger = logging.getLogger('NDLSearchAPI')
        logger.addHandler(self.log_handler)
        
        self.init_ui()
        
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        
        # Search parameters section
        params_group = QGroupBox("Search Parameters")
        params_layout = QFormLayout()
        
        # Search mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["title", "creator", "ndc", "custom"])
        params_layout.addRow("Search Mode:", self.mode_combo)
        
        # Query
        self.query_edit = QLineEdit()
        params_layout.addRow("Search Query:", self.query_edit)
        
        # NDC code (optional)
        self.ndc_edit = QLineEdit()
        params_layout.addRow("NDC Code (optional):", self.ndc_edit)
        
        # Year range
        year_layout = QHBoxLayout()
        self.from_year_edit = QLineEdit()
        self.until_year_edit = QLineEdit()
        year_layout.addWidget(QLabel("From:"))
        year_layout.addWidget(self.from_year_edit)
        year_layout.addWidget(QLabel("Until:"))
        year_layout.addWidget(self.until_year_edit)
        year_layout.addStretch()
        params_layout.addRow("Year Range:", year_layout)
        
        # Output directory
        dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit("ndl_data")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_output_dir)
        dir_layout.addWidget(self.output_dir_edit)
        dir_layout.addWidget(self.browse_btn)
        params_layout.addRow("Output Directory:", dir_layout)
        
        # Max records
        self.max_records_spin = QSpinBox()
        self.max_records_spin.setRange(1, 500)
        self.max_records_spin.setValue(200)
        params_layout.addRow("Max Records per Request:", self.max_records_spin)
        
        # Export to JSON checkbox
        self.export_json_check = QCheckBox()
        self.export_json_check.setChecked(True)
        params_layout.addRow("Export to JSON:", self.export_json_check)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.search_btn = QPushButton("Start Search")
        self.search_btn.clicked.connect(self.start_search)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_search)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.search_btn)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Log display
        log_group = QGroupBox("Search Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        self.setLayout(main_layout)
        
    def browse_output_dir(self):
        """Open a directory selection dialog to choose the output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir_edit.text()
        )
        if directory:
            self.output_dir_edit.setText(directory)
            
    def start_search(self):
        """Start the search with the current parameters."""
        # Validate inputs
        if not self.query_edit.text():
            QMessageBox.warning(self, "Input Error", "Please enter a search query.")
            return
            
        # Disable search button, enable cancel button
        self.search_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        # Clear log
        self.log_text.clear()
        
        # Get search parameters
        params = {
            "mode": self.mode_combo.currentText(),
            "query": self.query_edit.text(),
            "ndc": self.ndc_edit.text() if self.ndc_edit.text() else None,
            "from_year": self.from_year_edit.text() if self.from_year_edit.text() else None,
            "until_year": self.until_year_edit.text() if self.until_year_edit.text() else None,
            "output_dir": self.output_dir_edit.text(),
            "max_records": self.max_records_spin.value(),
            "export_json": self.export_json_check.isChecked(),
            "gui_mode": True  # Indicate that we're running in GUI mode
        }
        
        # Start search in a separate thread
        self.search_thread = threading.Thread(
            target=self.run_search,
            args=(params,),
            daemon=True
        )
        self.search_thread.start()
        
        # Start timer to update UI
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(500)  # Update every 500ms
        
    def run_search(self, params):
        """Run the search in a background thread."""
        try:
            self.search_wrapper.run_search(**params)
        except Exception as e:
            # Use error level for exceptions
            self.append_log(f"Error: {str(e)}", logging.ERROR)
        finally:
            # Re-enable search button, disable cancel button
            self.search_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            
    def update_progress(self):
        """Update the progress bar."""
        # If search is running, update progress based on the wrapper's state
        if hasattr(self, 'search_thread') and self.search_thread.is_alive():
            progress = self.search_wrapper.get_progress()
            self.progress_bar.setValue(int(progress * 100))
        else:
            # Search completed or not running
            self.progress_bar.setValue(100)
            self.progress_timer.stop()
            self.search_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            
    def cancel_search(self):
        """Cancel the ongoing search."""
        self.search_wrapper.cancel_search()
        # Use warning level for cancellation
        self.append_log("Search operation cancelled by user.", logging.WARNING)
        
    def append_log(self, message, level=None):
        """Append a message to the log display with appropriate styling based on log level."""
        try:
            # Apply color based on log level
            if level is None:
                # If no level specified, just append the raw message
                self.log_text.append(message)
            else:
                # Color code based on log level
                if level >= logging.ERROR:
                    # Error - red
                    styled_msg = f"<span style='color:#FF0000;'>{message}</span>"
                elif level >= logging.WARNING:
                    # Warning - orange
                    styled_msg = f"<span style='color:#FFA500;'>{message}</span>"
                elif level >= logging.INFO:
                    # Info - normal
                    styled_msg = message
                else:
                    # Debug - gray
                    styled_msg = f"<span style='color:#808080;'>{message}</span>"
                    
                self.log_text.append(styled_msg)
                
            # Scroll to the bottom
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception as e:
            # Last resort fallback
            print(f"Error in append_log: {str(e)}")
            try:
                self.log_text.append(f"<span style='color:red;'>Error displaying log message: {str(e)}</span>")
            except:
                pass

class ProcessTab(QWidget):
    """Tab for NDL data processing functionality."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize the wrapper
        self.process_wrapper = BiblioDataProcessorWrapper()
        
        # Set up logging - capture from all loggers
        self.log_handler = LogHandler()
        # Connect to signal with 2 arguments (message, level)
        self.log_handler.log_message.connect(self.append_log)
        
        # Add handler to root logger to capture ALL logging messages
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        
        # Also explicitly add to specific loggers to ensure we catch everything
        logging.getLogger('ndl_processor').addHandler(self.log_handler)
        logging.getLogger('__main__').addHandler(self.log_handler)
        
        # Make sure the log handler level is set low enough to capture all messages
        self.log_handler.setLevel(logging.DEBUG)
        
        self.init_ui()
        
        # Connect log level changes
        self.log_level_combo.currentTextChanged.connect(self.update_log_level)
        self.http_log_level_combo.currentTextChanged.connect(self.update_http_log_level)
        
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        
        # Processing parameters section
        params_group = QGroupBox("Processing Parameters")
        params_layout = QFormLayout()
        
        # Database path
        db_layout = QHBoxLayout()
        self.db_path_edit = QLineEdit("ndl_data.db")
        self.db_browse_btn = QPushButton("Browse...")
        self.db_browse_btn.clicked.connect(self.browse_db_path)
        db_layout.addWidget(self.db_path_edit)
        db_layout.addWidget(self.db_browse_btn)
        params_layout.addRow("Database Path:", db_layout)
        
        # Data directory
        dir_layout = QHBoxLayout()
        self.data_dir_edit = QLineEdit("ndl_data")
        self.dir_browse_btn = QPushButton("Browse...")
        self.dir_browse_btn.clicked.connect(self.browse_data_dir)
        dir_layout.addWidget(self.data_dir_edit)
        dir_layout.addWidget(self.dir_browse_btn)
        params_layout.addRow("Data Directory:", dir_layout)
        
        # Always use JSON mode (checkbox is now hidden/removed as it's the only option)
        
        # Use method 2 checkbox
        self.use_method2_check = QCheckBox()
        self.use_method2_check.setChecked(True)
        params_layout.addRow("Use Method 2 (GPT):", self.use_method2_check)
        
        # Auto type check level
        self.auto_type_check_spin = QSpinBox()
        self.auto_type_check_spin.setRange(0, 2)
        self.auto_type_check_spin.setValue(2)
        params_layout.addRow("Auto Type Check Level:", self.auto_type_check_spin)
        
        # Deduplication threshold
        self.dedup_threshold_spin = QDoubleSpinBox()
        self.dedup_threshold_spin.setRange(0.0, 1.0)
        self.dedup_threshold_spin.setValue(0.7)
        self.dedup_threshold_spin.setSingleStep(0.05)
        params_layout.addRow("Deduplication Threshold:", self.dedup_threshold_spin)
        
        # Manual deduplication checkbox
        self.manual_dedup_check = QCheckBox()
        self.manual_dedup_check.setChecked(False)
        params_layout.addRow("Manual Deduplication:", self.manual_dedup_check)
        
        # Year difference threshold
        self.year_diff_spin = QSpinBox()
        self.year_diff_spin.setRange(0, 10)
        self.year_diff_spin.setValue(5)
        params_layout.addRow("Year Difference Threshold:", self.year_diff_spin)
        
        # GPT credibility threshold
        self.gpt_cred_spin = QDoubleSpinBox()
        self.gpt_cred_spin.setRange(0.0, 1.0)
        self.gpt_cred_spin.setValue(0.8)
        self.gpt_cred_spin.setSingleStep(0.05)
        params_layout.addRow("GPT Credibility Threshold:", self.gpt_cred_spin)
        
        # Log level dropdown
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        params_layout.addRow("Log Level:", self.log_level_combo)
        
        # HTTP Log level dropdown
        self.http_log_level_combo = QComboBox()
        self.http_log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.http_log_level_combo.setCurrentText("WARNING")
        params_layout.addRow("HTTP Log Level:", self.http_log_level_combo)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Deduplication section
        dedup_group = QGroupBox("Deduplication")
        dedup_layout = QVBoxLayout()
        
        self.dedup_items_text = QTextEdit()
        self.dedup_items_text.setReadOnly(True)
        self.dedup_items_text.setFixedHeight(150)
        dedup_layout.addWidget(self.dedup_items_text)
        
        dedup_btn_layout = QHBoxLayout()
        self.keep_item1_btn = QPushButton("Keep Item 1")
        self.keep_item1_btn.clicked.connect(lambda: self.dedup_choice_handler("1"))
        self.keep_item2_btn = QPushButton("Keep Item 2")
        self.keep_item2_btn.clicked.connect(lambda: self.dedup_choice_handler("2"))
        self.not_duplicate_btn = QPushButton("Not Duplicate")
        self.not_duplicate_btn.clicked.connect(lambda: self.dedup_choice_handler("n"))
        
        dedup_btn_layout.addWidget(self.keep_item1_btn)
        dedup_btn_layout.addWidget(self.keep_item2_btn)
        dedup_btn_layout.addWidget(self.not_duplicate_btn)
        
        dedup_layout.addLayout(dedup_btn_layout)
        dedup_group.setLayout(dedup_layout)
        
        # Initially disable deduplication controls
        self.enable_dedup_controls(False)
        
        main_layout.addWidget(dedup_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.clicked.connect(self.start_processing)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.process_btn)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Log display
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        self.setLayout(main_layout)
        
        # Connect signals from the process wrapper
        self.process_wrapper.dedup_request.connect(self.show_dedup_request)
        self.process_wrapper.entity_type_request.connect(self.show_entity_type_dialog)
        self.process_wrapper.processing_finished.connect(self.processing_finished)
        
    def browse_db_path(self):
        """Open a file dialog to select the database file."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Select Database File", self.db_path_edit.text(),
            "SQLite Database (*.db)", options=options
        )
        if file_name:
            self.db_path_edit.setText(file_name)
            
    def browse_data_dir(self):
        """Open a directory selection dialog to choose the data directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Data Directory", self.data_dir_edit.text()
        )
        if directory:
            self.data_dir_edit.setText(directory)
            
    def update_log_level(self, level_text):
        """Update the log handler's level when the dropdown changes."""
        level = getattr(logging, level_text, logging.INFO)
        self.log_handler.setLevel(level)
        self.log_text.append(f"<span style='color:blue;'>Log level changed to {level_text}</span>")
        
        # Also update the root logger level to ensure we filter at the source
        logging.getLogger('ndl_processor').setLevel(level)
        
    def update_http_log_level(self, level_text):
        """Update HTTP loggers' level when the dropdown changes."""
        level = getattr(logging, level_text, logging.WARNING)
        logging.getLogger('httpcore').setLevel(level)
        logging.getLogger('httpx').setLevel(level)
        logging.getLogger('openai').setLevel(level)
        self.log_text.append(f"<span style='color:blue;'>HTTP log level changed to {level_text}</span>")
        
    def start_processing(self):
        """Start the data processing with the current parameters."""
        # Validate inputs
        if not os.path.exists(self.data_dir_edit.text()):
            QMessageBox.warning(self, "Input Error", "The specified data directory does not exist.")
            return
            
        # Disable process button, enable cancel button
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        # Clear log
        self.log_text.clear()
        
        # Get log levels from combo boxes
        log_level = self.log_level_combo.currentText()
        http_log_level = self.http_log_level_combo.currentText()
        
        # Make sure our log handler's level matches the selected log level
        level = getattr(logging, log_level, logging.INFO)
        self.log_handler.setLevel(level)
        
        # Print a start message to show we're capturing logs
        self.log_text.append(f"<span style='color:blue;'>=========================================</span>")
        self.log_text.append(f"<span style='color:blue;'>Starting processing with log level: {log_level}</span>")
        self.log_text.append(f"<span style='color:blue;'>HTTP log level: {http_log_level}</span>")
        self.log_text.append(f"<span style='color:blue;'>Using JSON mode for data processing</span>")
        self.log_text.append(f"<span style='color:blue;'>=========================================</span>")
        
        # Get processing parameters
        params = {
            "db_path": self.db_path_edit.text(),
            "use_method2": self.use_method2_check.isChecked(),
            "auto_type_check": self.auto_type_check_spin.value(),
            "dedup_threshold": self.dedup_threshold_spin.value(),
            "manual_dedup": self.manual_dedup_check.isChecked(),
            "year_diff_threshold": self.year_diff_spin.value(),
            "gpt_credibility_threshold": self.gpt_cred_spin.value(),
            "data_dir": self.data_dir_edit.text(),
            "gui_mode": True,  # Indicate that we're running in GUI mode
            "log_level": log_level,  # Pass the selected log level
            "http_log_level": http_log_level,  # HTTP log level
            "json_mode": True  # Always use JSON mode
        }
        
        # Start processing in a separate thread
        self.process_thread = threading.Thread(
            target=self.run_processing,
            args=(params,),
            daemon=True
        )
        self.process_thread.start()
        
        # Start timer to update UI
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(500)  # Update every 500ms
        
    def run_processing(self, params):
        """Run the processing in a background thread."""
        try:
            # Get log level from params
            log_level = params.get('log_level', 'INFO')
            http_log_level = params.get('http_log_level', 'WARNING')
            
            # Pass log levels to the wrapper
            self.process_wrapper.run_processing(
                db_path=params['db_path'],
                use_method2=params['use_method2'],
                auto_type_check=params['auto_type_check'],
                dedup_threshold=params['dedup_threshold'],
                manual_dedup=params['manual_dedup'],
                year_diff_threshold=params['year_diff_threshold'],
                gpt_credibility_threshold=params['gpt_credibility_threshold'],
                data_dir=params['data_dir'],
                gui_mode=params['gui_mode'],
                log_level=log_level,
                http_log_level=http_log_level
            )
        except Exception as e:
            # Use error level for exceptions
            self.append_log(f"Error: {str(e)}", logging.ERROR)
        
    def update_progress(self):
        """Update the progress bar."""
        # If processing is running, update progress based on the wrapper's state
        if hasattr(self, 'process_thread') and self.process_thread.is_alive():
            progress = self.process_wrapper.get_progress()
            self.progress_bar.setValue(int(progress * 100))
        else:
            # Processing completed or not running
            self.progress_timer.stop()
            
    def cancel_processing(self):
        """Cancel the ongoing processing."""
        self.process_wrapper.cancel_processing()
        # Use warning level for cancellation
        self.append_log("Processing operation cancelled by user.", logging.WARNING)
        
    def append_log(self, message, level=None):
        """Append a message to the log display with appropriate styling based on log level."""
        try:
            # Apply color based on log level
            if level is None:
                # If no level specified, just append the raw message
                self.log_text.append(message)
            else:
                # Color code based on log level
                if level >= logging.ERROR:
                    # Error - red
                    styled_msg = f"<span style='color:#FF0000;'>{message}</span>"
                elif level >= logging.WARNING:
                    # Warning - orange
                    styled_msg = f"<span style='color:#FFA500;'>{message}</span>"
                elif level >= logging.INFO:
                    # Info - normal
                    styled_msg = message
                else:
                    # Debug - gray
                    styled_msg = f"<span style='color:#808080;'>{message}</span>"
                    
                self.log_text.append(styled_msg)
                
            # Scroll to the bottom
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception as e:
            # Last resort fallback
            print(f"Error in append_log: {str(e)}")
            try:
                self.log_text.append(f"<span style='color:red;'>Error displaying log message: {str(e)}</span>")
            except:
                pass
        
    def show_dedup_request(self, item1, item2):
        """Display deduplication request in the UI."""
        self.dedup_items_text.clear()
        self.dedup_items_text.append(f"<b>Item 1:</b> {item1}")
        self.dedup_items_text.append("<hr>")
        self.dedup_items_text.append(f"<b>Item 2:</b> {item2}")
        
        # Enable deduplication controls
        self.enable_dedup_controls(True)
        
        # Log the request with direct HTML formatting
        self.log_text.append(f"<span style='color:blue;'>DEDUPLICATION REQUEST: Please select which item to keep or indicate they are not duplicates</span>")
        
        # Make the deduplication group more noticeable
        self.flash_deduplication_group()
        
    def flash_deduplication_group(self):
        """Flash the deduplication group to draw attention."""
        self.dedup_items_text.setStyleSheet("background-color: #ffe0e0;")
        QTimer.singleShot(500, lambda: self.dedup_items_text.setStyleSheet("background-color: white;"))
        QTimer.singleShot(1000, lambda: self.dedup_items_text.setStyleSheet("background-color: #ffe0e0;"))
        QTimer.singleShot(1500, lambda: self.dedup_items_text.setStyleSheet("background-color: white;"))
        
    def dedup_choice_handler(self, choice):
        """Handle a deduplication choice from the user."""
        # Call the wrapper's dedup_choice method
        self.process_wrapper.dedup_choice(choice)
        
        # Log the choice with direct HTML formatting
        if choice == '1':
            self.log_text.append("<span style='color:green;'>Keeping Item 1</span>")
        elif choice == '2':
            self.log_text.append("<span style='color:green;'>Keeping Item 2</span>")
        else:
            self.log_text.append("<span style='color:orange;'>Items marked as not duplicates</span>")
            
        # Reset the UI
        self.dedup_items_text.clear()
        self.dedup_items_text.setStyleSheet("background-color: white;")
        self.enable_dedup_controls(False)
        
    def enable_dedup_controls(self, enabled):
        """Enable or disable deduplication controls."""
        self.keep_item1_btn.setEnabled(enabled)
        self.keep_item2_btn.setEnabled(enabled)
        self.not_duplicate_btn.setEnabled(enabled)
        
    def show_entity_type_dialog(self, entity_name, suggested_type=None):
        """Show a dialog for entity type determination.
        
        Args:
            entity_name (str): Name of the entity to determine type for
            suggested_type (str, optional): Type suggested by GPT
        """
        self.append_log(f"<span style='color:purple;'>ENTITY TYPE REQUEST: Determining type for '{entity_name}'</span>")
        
        if suggested_type:
            # If there's a suggested type, ask for confirmation
            msg = f'GPT suggests "{entity_name}" is a(n) {suggested_type}. Do you agree?'
            reply = QMessageBox.question(
                self, 'Entity Type Confirmation', 
                msg,
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                self.append_log(f"<span style='color:green;'>Entity '{entity_name}' set as '{suggested_type}'</span>")
                self.process_wrapper.entity_type_choice(suggested_type)
                return
            elif reply == QMessageBox.Cancel:
                self.append_log(f"<span style='color:orange;'>Entity '{entity_name}' set as 'unclassified' (cancelled)</span>")
                self.process_wrapper.entity_type_choice("unclassified")
                return
            # If No, fall through to manual selection
            
        # Show a dialog with entity type options
        items = ["person", "organization", "publication", "unclassified", "other"]
        item, ok = QInputDialog.getItem(
            self, "Entity Type Selection", 
            f'Select type for "{entity_name}":',
            items, 0, False
        )
        
        if ok and item:
            if item == "other":
                # If "other", ask for custom type
                custom_type, ok = QInputDialog.getText(
                    self, "Custom Entity Type", 
                    "Enter custom entity type:"
                )
                if ok and custom_type:
                    self.append_log(f"<span style='color:green;'>Entity '{entity_name}' set as custom type '{custom_type}'</span>")
                    self.process_wrapper.entity_type_choice(custom_type)
                else:
                    # Default to unclassified if cancelled
                    self.append_log(f"<span style='color:orange;'>Entity '{entity_name}' set as 'unclassified' (custom type cancelled)</span>")
                    self.process_wrapper.entity_type_choice("unclassified")
            else:
                self.append_log(f"<span style='color:green;'>Entity '{entity_name}' set as '{item}'</span>")
                self.process_wrapper.entity_type_choice(item)
        else:
            # Default to unclassified if cancelled
            self.append_log(f"<span style='color:orange;'>Entity '{entity_name}' set as 'unclassified' (selection cancelled)</span>")
            self.process_wrapper.entity_type_choice("unclassified")
        
    def processing_finished(self):
        """Handle the processing finished signal."""
        self.progress_bar.setValue(100)
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.enable_dedup_controls(False)
        QMessageBox.information(self, "Processing Complete", "NDL data processing has completed successfully.")

class CiNiiSearchTab(QWidget):
    """Tab for CiNii Search functionality."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize the wrapper
        self.search_wrapper = CiNiiSearchWrapper()
        
        # Set up logging
        self.log_handler = LogHandler()
        # Connect to signal with 2 arguments (message, level)
        self.log_handler.log_message.connect(self.append_log)
        
        logger = logging.getLogger('CiNiiSearchAPI')
        logger.addHandler(self.log_handler)
        
        self.init_ui()
        
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        
        # Search parameters section
        params_group = QGroupBox("Search Parameters")
        params_layout = QFormLayout()
        
        # Search type
        self.search_type_combo = QComboBox()
        self.search_type_combo.addItems(["all", "data", "articles", "books", "dissertations", "projects"])
        params_layout.addRow("Search Type:", self.search_type_combo)
        
        # Title (optional)
        self.title_edit = QLineEdit()
        params_layout.addRow("Title (optional):", self.title_edit)
        
        # Creator (optional)
        self.creator_edit = QLineEdit()
        params_layout.addRow("Creator (optional):", self.creator_edit)
        
        # Researcher ID (optional)
        self.researcher_id_edit = QLineEdit()
        params_layout.addRow("Researcher ID (optional):", self.researcher_id_edit)
        
        # Category code (optional)
        self.category_edit = QLineEdit()
        params_layout.addRow("Category Code (optional):", self.category_edit)
        
        # Year range
        year_layout = QHBoxLayout()
        self.from_year_edit = QLineEdit()
        self.until_year_edit = QLineEdit()
        year_layout.addWidget(QLabel("From:"))
        year_layout.addWidget(self.from_year_edit)
        year_layout.addWidget(QLabel("Until:"))
        year_layout.addWidget(self.until_year_edit)
        year_layout.addStretch()
        params_layout.addRow("Year Range:", year_layout)
        
        # Output directory
        dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit("cinii_data")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_output_dir)
        dir_layout.addWidget(self.output_dir_edit)
        dir_layout.addWidget(self.browse_btn)
        params_layout.addRow("Output Directory:", dir_layout)
        
        # Records per page
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 500)
        self.count_spin.setValue(100)
        params_layout.addRow("Records per Page:", self.count_spin)
        
        # Export to JSON checkbox
        self.export_json_check = QCheckBox()
        self.export_json_check.setChecked(True)
        params_layout.addRow("Export to JSON:", self.export_json_check)
        
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.search_btn = QPushButton("Start Search")
        self.search_btn.clicked.connect(self.start_search)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_search)
        self.cancel_btn.setEnabled(False)
        btn_layout.addWidget(self.search_btn)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Log display
        log_group = QGroupBox("Search Log")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        self.setLayout(main_layout)
        
    def browse_output_dir(self):
        """Open a directory selection dialog to choose the output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir_edit.text()
        )
        if directory:
            self.output_dir_edit.setText(directory)
            
    def start_search(self):
        """Start the search with the current parameters."""
        # Validate inputs - ensure at least one search parameter is provided
        if not any([
            self.title_edit.text(), 
            self.creator_edit.text(), 
            self.researcher_id_edit.text(), 
            self.category_edit.text()
        ]):
            QMessageBox.warning(self, "Input Error", "Please provide at least one search parameter (title, creator, researcher ID, or category).")
            return
            
        # Disable search button, enable cancel button
        self.search_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        # Clear log
        self.log_text.clear()
        
        # Get search parameters
        params = {
            "search_type": self.search_type_combo.currentText(),
            "title": self.title_edit.text() if self.title_edit.text() else None,
            "creator": self.creator_edit.text() if self.creator_edit.text() else None,
            "researcher_id": self.researcher_id_edit.text() if self.researcher_id_edit.text() else None,
            "category": self.category_edit.text() if self.category_edit.text() else None,
            "from_year": self.from_year_edit.text() if self.from_year_edit.text() else None,
            "until_year": self.until_year_edit.text() if self.until_year_edit.text() else None,
            "output_dir": self.output_dir_edit.text(),
            "count": self.count_spin.value(),
            "export_json": self.export_json_check.isChecked(),
            "gui_mode": True  # Indicate that we're running in GUI mode
        }
        
        # Start search in a separate thread
        self.search_thread = threading.Thread(
            target=self.run_search,
            args=(params,),
            daemon=True
        )
        self.search_thread.start()
        
        # Start timer to update UI
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(500)  # Update every 500ms
        
    def run_search(self, params):
        """Run the search in a background thread."""
        try:
            self.search_wrapper.run_search(**params)
        except Exception as e:
            # Use error level for exceptions
            self.append_log(f"Error: {str(e)}", logging.ERROR)
        finally:
            # Re-enable search button, disable cancel button
            self.search_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            
    def update_progress(self):
        """Update the progress bar."""
        # If search is running, update progress based on the wrapper's state
        if hasattr(self, 'search_thread') and self.search_thread.is_alive():
            progress = self.search_wrapper.get_progress()
            self.progress_bar.setValue(int(progress * 100))
        else:
            # Search completed or not running
            self.progress_bar.setValue(100)
            self.progress_timer.stop()
            self.search_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            
    def cancel_search(self):
        """Cancel the ongoing search."""
        self.search_wrapper.cancel_search()
        # Use warning level for cancellation
        self.append_log("Search operation cancelled by user.", logging.WARNING)
        
    def append_log(self, message, level=None):
        """Append a message to the log display with appropriate styling based on log level."""
        try:
            # Apply color based on log level
            if level is None:
                # If no level specified, just append the raw message
                self.log_text.append(message)
            else:
                # Color code based on log level
                if level >= logging.ERROR:
                    # Error - red
                    styled_msg = f"<span style='color:#FF0000;'>{message}</span>"
                elif level >= logging.WARNING:
                    # Warning - orange
                    styled_msg = f"<span style='color:#FFA500;'>{message}</span>"
                elif level >= logging.INFO:
                    # Info - normal
                    styled_msg = message
                else:
                    # Debug - gray
                    styled_msg = f"<span style='color:#808080;'>{message}</span>"
                    
                self.log_text.append(styled_msg)
                
            # Scroll to the bottom
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception as e:
            print(f"Error in append_log: {str(e)}")

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        # Set window properties
        self.setWindowTitle("Literature Search & Process Tool")
        self.setGeometry(100, 100, 900, 700)
        
        # Create tabs
        tab_widget = QTabWidget()
        ndl_search_tab = SearchTab()
        cinii_search_tab = CiNiiSearchTab()
        process_tab = ProcessTab()
        
        tab_widget.addTab(ndl_search_tab, "NDL Search")
        tab_widget.addTab(cinii_search_tab, "CiNii Search")
        tab_widget.addTab(process_tab, "Process Data")
        
        self.setCentralWidget(tab_widget)
        
        # Show the window
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 
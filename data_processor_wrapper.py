import logging
import threading
import queue
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QInputDialog, QMessageBox
import sys

# Import the original NDL processing script
from data_processor import BiblioDataProcessor

class BiblioDataProcessorWrapper(QObject):
    """Wrapper for BiblioDataProcessor that adds GUI support."""
    
    # Define signals
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal()
    dedup_request = pyqtSignal(str, str)  # Item1, Item2
    progress_updated = pyqtSignal(float)
    entity_type_request = pyqtSignal(str, str)  # Entity name, suggested type
    
    def __init__(self):
        super().__init__()
        self.processor = None
        self.is_cancelled = False
        self.progress = 0.0
        self.lock = threading.Lock()
        self.dedup_queue = queue.Queue()
        self.entity_type_queue = queue.Queue()
        self.total_files = 0
        self.processed_files = 0
        
    def run_processing(self, db_path, data_dir, use_method2=True, auto_type_check=2,
                      dedup_threshold=0.7, manual_dedup=False, year_diff_threshold=5,
                      gpt_credibility_threshold=0.8, gui_mode=False, log_level='INFO',
                      http_log_level='WARNING', json_mode=True):
        """Run data processing with the given parameters."""
        self.is_cancelled = False
        self.progress = 0.0
        self.processed_files = 0
        
        # Emit processing started signal
        self.processing_started.emit()
        
        try:
            # Configure logging to ensure messages appear in both GUI and console
            if gui_mode:
                # Set up the root logger for GUI mode
                root_logger = logging.getLogger()
                
                # Set level to capture all needed messages
                root_level = getattr(logging, log_level.upper(), logging.INFO)
                root_level = min(root_level, logging.DEBUG)  # DEBUG is most verbose
                root_logger.setLevel(root_level)
                
                # Make sure we have at least one console handler for the root logger
                has_console_handler = False
                for handler in root_logger.handlers:
                    if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
                        has_console_handler = True
                        break
                
                if not has_console_handler:
                    # Add a console handler if none exists
                    console_handler = logging.StreamHandler(sys.stderr)
                    console_handler.setLevel(root_level)
                    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
                    root_logger.addHandler(console_handler)
                
                logging.info("Starting processing in GUI mode with log level: %s", log_level)
            
            # Initialize processor with GUI-specific modifications
            self.processor = BiblioDataProcessorWithGUI(
                db_path=db_path,
                use_method2=use_method2,
                auto_type_check=auto_type_check,
                dedup_threshold=dedup_threshold,
                manual_dedup=manual_dedup,
                year_diff_threshold=year_diff_threshold,
                gpt_credibility_threshold=gpt_credibility_threshold,
                gui_mode=gui_mode,
                log_level=log_level,
                http_log_level=http_log_level,
                json_mode=True,  # Always use JSON mode
                dedup_callback=self.handle_dedup_request,
                progress_callback=self.update_progress,
                entity_type_callback=self.handle_entity_type_request
            )
            
            # Process the directory if not cancelled
            if not self.is_cancelled:
                self.processor.process_directory(data_dir)
            
            # Clean up
            if self.processor:
                self.processor.close()
                self.processor = None
            
            # Update progress to 100% if completed successfully
            if not self.is_cancelled:
                self.update_progress(1.0)
                # Emit processing finished signal
                self.processing_finished.emit()
            else:
                logging.info("Processing was cancelled by the user.")
                self.processing_finished.emit()
            
        except Exception as e:
            logging.error(f"Error during processing: {str(e)}")
            if self.processor:
                self.processor.close()
                self.processor = None
            # Ensure the processing_finished signal is emitted even in case of error
            self.processing_finished.emit()
    
    def update_progress(self, progress):
        """Update the current progress value."""
        with self.lock:
            self.progress = progress
            self.progress_updated.emit(progress)
    
    def get_progress(self):
        """Get the current progress value."""
        with self.lock:
            return self.progress
    
    def cancel_processing(self):
        """Cancel the current processing operation."""
        logging.info("Cancel request received")
        self.is_cancelled = True
        if self.processor:
            self.processor.cancel_processing()
    
    def handle_dedup_request(self, item1, item2):
        """Handle a deduplication request from the processor.
        
        This method emits a signal to display the deduplication choice in the UI
        and waits for the user's response. It blocks the processing thread until 
        a response is received through the dedup_queue.
        """
        # Log that we're requesting deduplication
        logging.info(f"Requesting deduplication decision for items: {item1} and {item2}")
        
        # Clear any existing items in the queue to prevent using old responses
        while not self.dedup_queue.empty():
            try:
                self.dedup_queue.get_nowait()
            except queue.Empty:
                break
        
        # Emit signal to show the request in the UI
        self.dedup_request.emit(item1, item2)
        
        # Wait for user input (this will block the processing thread)
        logging.info("Waiting for deduplication decision from user...")
        result = self.dedup_queue.get()
        logging.info(f"Received deduplication decision from GUI: {result}")
        
        # Return the user's choice
        return result
    
    def dedup_choice(self, choice):
        """Handle user choice for deduplication.
        
        Args:
            choice (str): '1' to keep item 1, '2' to keep item 2, 'n' for not duplicate
        """
        logging.info(f"User selected deduplication choice in GUI: {choice}")
        
        # Make sure the choice is one of the expected values
        if choice not in ['1', '2', 'n']:
            logging.warning(f"Unexpected deduplication choice: {choice}, defaulting to 'n'")
            choice = 'n'
            
        # Put the choice in the queue for the processing thread
        self.dedup_queue.put(choice)
        
    def handle_entity_type_request(self, entity_name, suggested_type=None):
        """Handle an entity type request from the processor.
        
        Args:
            entity_name (str): Name of the entity to type
            suggested_type (str, optional): Suggested type from GPT
            
        Returns:
            str: The entity type decided by the user or accepted suggestion
        """
        logging.info(f"Requesting entity type decision for: {entity_name}")
        
        # Send a signal for the main thread to show a dialog and get user input
        self.entity_type_request.emit(entity_name, suggested_type)
        
        # Wait for the response
        logging.info("Waiting for entity type decision from user...")
        result = self.entity_type_queue.get()
        logging.info(f"Received entity type decision: {result}")
        
        return result
        
    def entity_type_choice(self, entity_type):
        """Handle user choice for entity type.
        
        Args:
            entity_type (str): The type chosen by the user
        """
        logging.info(f"User selected entity type: {entity_type}")
        self.entity_type_queue.put(entity_type)

class BiblioDataProcessorWithGUI(BiblioDataProcessor):
    """Extended BiblioDataProcessor with GUI support."""
    
    def __init__(self, db_path="ndl_data.db", use_method2=True, auto_type_check=2,
                dedup_threshold=0.7, manual_dedup=False, year_diff_threshold=5,
                gpt_credibility_threshold=0.8, gui_mode=False, log_level='INFO',
                http_log_level='WARNING', json_mode=True, dedup_callback=None, progress_callback=None, 
                entity_type_callback=None):
        """Initialize the processor with GUI support."""
        # Initialize with GUI mode
        super().__init__(
            db_path=db_path,
            use_method2=use_method2,
            auto_type_check=auto_type_check,
            dedup_threshold=dedup_threshold,
            manual_dedup=manual_dedup,
            year_diff_threshold=year_diff_threshold,
            gpt_credibility_threshold=gpt_credibility_threshold,
            gui_mode=gui_mode,
            log_level=log_level,
            http_log_level=http_log_level
        )
        
        # Set GUI-specific attributes
        self.gui_mode = gui_mode
        self.dedup_callback = dedup_callback
        self.progress_callback = progress_callback
        self.entity_type_callback = entity_type_callback
        self.is_cancelled = False
        
        # Log propagation is properly configured in setup_logging method
        if gui_mode:
            self.logger.info("Initialized BiblioDataProcessorWithGUI in GUI mode.")
        
    def setup_logging(self):
        """Override setup_logging to ensure proper configuration for GUI mode."""
        # Call the parent method first
        super().setup_logging()
        
        # In GUI mode, ensure logs are visible both in GUI and console
        if self.gui_mode:
            # Check if we have multiple console handlers (which could cause duplicates)
            console_handlers = []
            for handler in self.logger.handlers:
                if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
                    console_handlers.append(handler)
            
            # If we have more than one console handler, remove the extras
            if len(console_handlers) > 1:
                # Keep just the first console handler
                for handler in console_handlers[1:]:
                    self.logger.removeHandler(handler)
                self.logger.info("Removed duplicate console handlers to prevent duplicate messages.")
            
            # We need to add the GUI handler directly to this logger and disable propagation
            # to avoid duplicate messages while ensuring GUI receives logs
            
            # First, check if the root logger has any handlers we can "borrow"
            root_logger = logging.getLogger()
            gui_handler = None
            
            for handler in root_logger.handlers:
                # Look for the custom GUI handler from the LogHandler class
                # We can identify it by checking if it has our special log_message attribute
                if hasattr(handler, 'log_message'):
                    gui_handler = handler
                    break
            
            if gui_handler:
                # Add the GUI handler directly to our logger
                self.logger.addHandler(gui_handler)
                self.logger.info("Added GUI log handler directly to processor logger.")
                
            # Disable propagation to prevent duplicate messages
            self.logger.propagate = False
            self.logger.info("Logger configured for both GUI and console display.")
        else:
            # In non-GUI mode, enable propagation
            self.logger.propagate = True
            self.logger.info("Logger configured for console display.")
        
    def cancel_processing(self):
        """Cancel the current processing operation."""
        self.is_cancelled = True
        self.logger.info("Processing cancelled by user")
        
    def process_directory(self, directory_path):
        """Override process_directory to track progress."""
        import glob
        from pathlib import Path
        
        directory = Path(directory_path)
        if not directory.exists():
            self.logger.error(f"Directory does not exist: {directory_path}")
            return
            
        # Count total JSON files
        json_files = list(directory.glob('**/*.json'))
        self.total_files = len(json_files)
        self.logger.info(f"Found {self.total_files} JSON files to process")
        
        if self.total_files == 0:
            self.logger.warning("No JSON files found to process")
            return
            
        # Process each file with progress tracking
        self.processed_files = 0
        for json_file in json_files:
            # Check if cancelled
            if self.is_cancelled:
                self.logger.warning("Processing cancelled, stopping...")
                break
                
            # Process this file
            self.logger.info(f"Processing file {self.processed_files + 1} of {self.total_files}: {json_file}")
            try:
                self.process_json_file(str(json_file))
                self.processed_files += 1
                
                # Update progress
                if self.progress_callback:
                    progress = self.processed_files / self.total_files
                    self.progress_callback(progress)
                    
            except Exception as e:
                self.logger.error(f"Error processing file {json_file}: {str(e)}")
                
        # Print summary
        self.print_summary()
        
    def process_json_file(self, file_path):
        """Override the process_json_file method to check for cancellation."""
        import json
        
        try:
            self.logger.info(f"Starting to process JSON file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as jsonfile:
                records = json.load(jsonfile)
                
            total_records = len(records)
            self.logger.info(f"Found {total_records} records in {file_path}")

            processed_records = 0
            for record in records:
                # Check for cancellation
                if self.is_cancelled:
                    self.logger.warning("Processing cancelled during file processing, stopping...")
                    return
                
                try:
                    self.process_json_record(record)
                    processed_records += 1
                    # Log progress after every record
                    if processed_records % 1 == 0:
                        self.logger.info(f"Processed {processed_records} out of {total_records} records ({processed_records/total_records*100:.1f}%)")
                except Exception as e:
                    self.logger.error(f"Error processing record: {str(e)}")
                    # Get more details with traceback
                    import traceback
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    continue

            self.conn.commit()
            self.logger.info(f"Successfully processed {processed_records} records from {file_path}")

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return
        
    def get_user_input(self, prompt, timeout=20):
        """Override get_user_input to use the GUI in GUI mode.
        
        In GUI mode, this method directly asks for deduplication decisions using
        a single-step approach where the user chooses from:
        - "1" to keep item 1
        - "2" to keep item 2 
        - "n" for not duplicates
        """
        if not self.gui_mode or not self.dedup_callback:
            # Fall back to the original implementation if not in GUI mode
            return super().get_user_input(prompt, timeout)
            
        # Extract item information from the prompt
        import re
        item1_match = re.search(r'Item 1: (.*?)(?:\n|$)', prompt)
        item2_match = re.search(r'Item 2: (.*?)(?:\n|$)', prompt)
        
        if not item1_match or not item2_match:
            self.logger.error("Could not extract item information from prompt")
            return None
            
        item1 = item1_match.group(1)
        item2 = item2_match.group(1)
        
        self.logger.info(f"Requesting GUI deduplication for items: {item1} and {item2}")
        
        # Use the callback to get user input through the GUI (1, 2, or n)
        result = self.dedup_callback(item1, item2)
        self.logger.info(f"GUI deduplication result: {result}")
        
        return result
            
    def determine_entity_type_method2(self, name: str, existing_method: int = 0) -> tuple:
        """Override determine_entity_type_method2 to use GUI for entity type confirmation."""
        # If entity was already checked with a higher or equal method, return None to keep existing type
        if existing_method >= self.auto_type_check:
            return None, existing_method
            
        # Use the parent class's get_item_type method with GUI support
        entity_type, is_auto = super().get_item_type(
            name, 
            self.auto_type_check,
            self.gui_mode,
            self.entity_type_callback
        )
        mode = self.auto_type_check
            
        # If get_item_type returns 'publication', treat it as 'organization'
        if entity_type == 'publication':
            entity_type = 'organization'
            
        return entity_type, mode 
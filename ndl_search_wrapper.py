import logging
import threading
import time
import os
import os.path
from PyQt6.QtCore import QObject, pyqtSignal

# Import the original NDL search script
from ndl_search_retriever import NDLSearchAPI

class NDLSearchWrapper(QObject):
    """Wrapper for NDLSearchAPI that adds GUI support."""
    
    # Define signals
    search_started = pyqtSignal()
    search_finished = pyqtSignal(bool)  # Success or failure
    
    def __init__(self):
        super().__init__()
        self.api = None
        self.is_cancelled = False
        self.progress = 0.0
        self.search_thread = None
        self.lock = threading.Lock()
        
    def run_search(self, mode, query, output_dir="ndl_data", max_records=200, 
                  ndc=None, from_year=None, until_year=None, gui_mode=False, export_json=True):
        """Run a search with the given parameters."""
        self.is_cancelled = False
        self.progress = 0.0
        
        # Initialize API
        self.api = NDLSearchAPIWithProgress(
            output_dir=output_dir,
            max_records=max_records,
            gui_mode=gui_mode
        )
        
        # Set progress callback
        self.api.set_progress_callback(self.update_progress)
        
        # Emit search started signal
        self.search_started.emit()
        
        # Execute search based on mode
        success = False
        search_result = None
        try:
            if mode == "title":
                search_result = self.api.search_by_title(query, ndc, from_year, until_year)
                success = search_result is not False
            elif mode == "creator":
                search_result = self.api.search_by_creator(query, ndc, from_year, until_year)
                success = search_result is not False
            elif mode == "ndc":
                search_result = self.api.search_by_ndc(query, from_year, until_year)
                success = search_result is not False
            elif mode == "custom":
                search_result = self.api.search_by_custom_query(query)
                success = search_result is not False
            
            # Export to JSON if requested and search was successful
            if success and export_json:
                logging.info(f"Exporting search results to JSON")
                # If search_result is a path (from consolidation), use that directly
                if isinstance(search_result, str) and os.path.isdir(search_result):
                    # Normalize the path to fix any duplicate directory issues
                    search_result = os.path.normpath(search_result)
                    self.api.export_results_to_json(search_result)
                else:
                    # Otherwise, let the export function find the latest directory
                    self.api.export_results_to_json()
                
            # Set progress to 100% when done
            self.update_progress(1.0)
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            success = False
        finally:
            # Emit search finished signal
            self.search_finished.emit(success)
            return success
            
    def update_progress(self, progress):
        """Update the current progress value."""
        with self.lock:
            self.progress = progress
            
    def get_progress(self):
        """Get the current progress value."""
        with self.lock:
            return self.progress
            
    def cancel_search(self):
        """Cancel the current search operation."""
        self.is_cancelled = True
        if self.api:
            self.api.cancel_search()

class NDLSearchAPIWithProgress(NDLSearchAPI):
    """Extended NDLSearchAPI with progress tracking for GUI."""
    
    def __init__(self, output_dir="ndl_data", max_records=200, gui_mode=False):
        super().__init__(output_dir, max_records)
        self.gui_mode = gui_mode
        self.is_cancelled = False
        self.progress_callback = None
        self.total_records = 0
        self.current_position = 0
        
    def set_progress_callback(self, callback):
        """Set a callback function to receive progress updates."""
        self.progress_callback = callback
        
    def cancel_search(self):
        """Cancel the current search operation."""
        self.is_cancelled = True
        
    def _report_progress(self):
        """Report the current progress."""
        if self.progress_callback and self.total_records > 0:
            progress = min(1.0, self.current_position / self.total_records)
            self.progress_callback(progress)
            
    def _make_request(self, params):
        """Override _make_request to check for cancellation."""
        if self.is_cancelled:
            self.logger.info("Search cancelled by user")
            return None
        return super()._make_request(params)
        
    def _execute_search(self, mode, query_term, query_parts=None, ndc=None, from_year=None, until_year=None, custom_query=None):
        """Override _execute_search to track progress."""
        # Set up parameters for the API request
        params = self.DEFAULT_PARAMS.copy()
        
        if custom_query:
            # Use the provided custom query directly
            params["query"] = custom_query
            query = custom_query
        else:
            # Construct the query string from parts
            # Add NDC code constraint if provided
            if ndc and mode != "ndc":
                query_parts.append(f'ndc="{ndc}"')
            
            # Add year range constraints if provided
            if from_year:
                query_parts.append(f'from="{from_year}"')
            if until_year:
                query_parts.append(f'until="{until_year}"')
            
            # Construct the query string
            query = " AND ".join(query_parts)
            params["query"] = query
            self.logger.info(f"Query parameters: {params}")
        
        # Create a directory for this specific search
        search_dir = f"{mode}_{query_term.replace('/', '_')}_{from_year}_{until_year}"
        search_path = os.path.join(self.output_dir, search_dir)
        
        # Check if there's an existing directory for this search to resume
        existing_dir = self._find_existing_search_dir(mode, query_term, from_year, until_year)
        if existing_dir:
            search_path = existing_dir
            self.logger.info(f"Found existing search directory: {search_path}")
            current_position = self._get_next_record_position(search_path)
            if current_position == 0:
                self.logger.error("Error determining next record position. Starting from the beginning.")
                current_position = 1
        else:
            Path(search_path).mkdir(parents=True, exist_ok=True)
            current_position = 1  # Start from the beginning
        
        # Track this directory for potential consolidation later
        if from_year and until_year and from_year == until_year:
            self.yearly_search_dirs.append(search_path)
        
        # If we're starting from the beginning, make an initial request to get the total count
        if current_position == 1:
            params["startRecord"] = "1"
            self.logger.info(f"Executing search: {query}")
            results = self._make_request(params)
            
            if results is None:
                self.logger.error("Initial request failed.")
                return False
            
            # Parse number of records
            num_records = self._get_number_of_records(results)
            self.logger.info(f"Found {num_records} records.")
            
            # Save the total records for progress tracking
            self.total_records = num_records
            self.current_position = 1
            self._report_progress()
            
            if num_records == 0:
                self.logger.info("No records found.")
                return False
            
            # Check if the number of records exceeds the maximum
            if num_records > self.MAX_ALLOWED_RECORDS:
                warning_msg = f"The query returns {num_records} records, which exceeds the maximum of {self.MAX_ALLOWED_RECORDS}. Only the first {self.MAX_ALLOWED_RECORDS} records will be retrieved."
                self.logger.warning(warning_msg)
                self.total_records = self.MAX_ALLOWED_RECORDS
            
            # Save the first batch
            self._save_result(results, search_path, 1)
            
            # Move to the next batch
            next_position = self._get_next_record_position_from_xml(results)
            if next_position:
                current_position = next_position
                self.current_position = next_position
            else:
                current_position = 1 + self.max_records
                self.current_position = current_position
            
            self._report_progress()
        else:
            # We're resuming, so read the number of records from the first page file
            first_page_file = os.path.join(search_path, "results_page_1.xml")
            if os.path.exists(first_page_file):
                with open(first_page_file, 'r', encoding='utf-8') as f:
                    first_page_content = f.read()
                num_records = self._get_number_of_records(first_page_content)
                self.logger.info(f"Resuming search with {num_records} total records from position {current_position}")
                
                # Save the total records for progress tracking
                self.total_records = num_records if num_records <= self.MAX_ALLOWED_RECORDS else self.MAX_ALLOWED_RECORDS
                self.current_position = current_position
                self._report_progress()
            else:
                self.logger.error("First page file not found. Cannot determine total record count.")
                return False
        
        # Fetch remaining batches, but limit to MAX_ALLOWED_RECORDS
        while current_position <= self.total_records and current_position <= self.MAX_ALLOWED_RECORDS:
            # Check if search was cancelled
            if self.is_cancelled:
                self.logger.info("Search cancelled by user")
                return False
                
            self.logger.info(f"Fetching records {current_position}-{min(current_position+self.max_records-1, self.total_records)}")
            params["startRecord"] = str(current_position)
            batch_results = self._make_request(params)
            
            if batch_results is None:
                self.logger.error(f"Failed to fetch records starting at position {current_position}.")
                break
            
            page_num = ((current_position - 1) // self.max_records) + 1
            self._save_result(batch_results, search_path, page_num)
            
            # Get the next record position from the response
            next_position = self._get_next_record_position_from_xml(batch_results)
            if next_position:
                current_position = next_position
                self.current_position = next_position
            else:
                # If no next record position is provided, just increment by the batch size
                current_position += self.max_records
                self.current_position = current_position
            
            # Update progress
            self._report_progress()
            
            time.sleep(2)  # Be nice to the API
        
        self.logger.info(f"All data saved to {search_path}")
        return True

# Import these at the bottom to avoid circular imports
from pathlib import Path 
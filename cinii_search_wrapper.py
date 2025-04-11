import logging
import threading
import time
import os
import os.path
from pathlib import Path
from PyQt5.QtCore import QObject, pyqtSignal

# Import the original CiNii search script
from cinii_search_retriever import CiNiiSearchAPI

class CiNiiSearchWrapper(QObject):
    """Wrapper for CiNiiSearchAPI that adds GUI support."""
    
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
        
    def run_search(self, search_type, output_dir="cinii_data", count=100, 
                   title=None, creator=None, researcher_id=None, category=None,
                   from_year=None, until_year=None, gui_mode=False, export_json=True):
        """Run a search with the given parameters."""
        self.is_cancelled = False
        self.progress = 0.0
        
        # Initialize API
        self.api = CiNiiSearchAPIWithProgress(
            output_dir=output_dir,
            count=count,
            gui_mode=gui_mode
        )
        
        # Set progress callback
        self.api.set_progress_callback(self.update_progress)
        
        # Emit search started signal
        self.search_started.emit()
        
        # Build search parameters
        search_params = {}
        
        if title:
            search_params['title'] = title
        
        if creator:
            search_params['creator'] = creator
        
        if category:
            search_params['category'] = category
        
        if researcher_id:
            search_params['researcherId'] = researcher_id
            
        if from_year:
            search_params['from'] = from_year
            
        if until_year:
            search_params['until'] = until_year
        
        # Execute search
        success = False
        search_result = None
        try:
            # Perform the search
            search_result = self.api.search(search_type, **search_params)
            success = search_result is not False
            
            # Export to JSON if requested and search was successful
            if success and export_json:
                logging.info(f"Exporting search results to JSON")
                if isinstance(search_result, str) and os.path.isdir(search_result):
                    # Normalize the path to fix any duplicate directory issues
                    search_result = os.path.normpath(search_result)
                    self.api.export_results_to_json(search_result)
                else:
                    # Let the export function find the latest directory
                    self.api.export_results_to_json()
                
            # Set progress to 100% when done
            self.update_progress(1.0)
        except Exception as e:
            logging.error(f"Error during CiNii search: {str(e)}")
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

class CiNiiSearchAPIWithProgress(CiNiiSearchAPI):
    """Extended CiNiiSearchAPI with progress tracking for GUI."""
    
    def __init__(self, output_dir="cinii_data", count=100, gui_mode=False):
        super().__init__(output_dir, count, gui_mode)
        self.is_cancelled = False
        self.progress_callback = None
        self.total_pages = 0
        self.current_page = 0
        
    def set_progress_callback(self, callback):
        """Set a callback function to receive progress updates."""
        self.progress_callback = callback
        
    def cancel_search(self):
        """Cancel the current search operation."""
        self.is_cancelled = True
        
    def _report_progress(self):
        """Report the current progress."""
        if self.progress_callback and self.total_pages > 0:
            progress = min(1.0, self.current_page / self.total_pages)
            self.progress_callback(progress)
            
    def _make_request(self, url, params):
        """Override _make_request to check for cancellation."""
        if self.is_cancelled:
            self.logger.info("Search cancelled by user")
            return None
        return super()._make_request(url, params)
        
    def _execute_search(self, search_type, params):
        """Override _execute_search to track progress."""
        # Create a directory for this specific search
        search_dir_parts = [search_type]
        
        # Add key parameters to the directory name
        for key in ['title', 'creator', 'researcherId', 'category', 'from', 'until']:
            if key in params:
                search_dir_parts.append(f"{key}_{params[key]}")
        
        search_dir_name = "_".join(search_dir_parts).replace('/', '_').replace(' ', '_')
        search_path = os.path.join(self.output_dir, search_dir_name)
        
        # Check if there's an existing directory for this search to resume
        existing_dir = self._find_existing_search_dir(search_dir_name)
        if existing_dir:
            search_path = existing_dir
            self.logger.info(f"Found existing search directory: {search_path}")
            current_position = self._get_next_page_number(search_path)
            self.current_page = current_position - 1  # For progress tracking
            if current_position == 1:
                self.logger.info("Starting from the beginning")
            else:
                self.logger.info(f"Resuming from page {current_position}")
        else:
            Path(search_path).mkdir(parents=True, exist_ok=True)
            current_position = 1  # Start from the beginning
            self.current_page = 0  # For progress tracking
            self.logger.info(f"Created new search directory: {search_path}")
        
        # Track this directory for potential use later
        self.search_dirs.append(search_path)
        
        # Construct the final URL for the API call
        endpoint = f"{self.BASE_URL}/{search_type}"
        self.logger.info(f"Using API endpoint: {endpoint}")
        
        # First query to get total results count if starting fresh
        if current_position == 1:
            first_params = params.copy()
            first_params['start'] = '1'
            self.logger.info(f"Executing initial search with parameters: {first_params}")
            
            results = self._make_request(endpoint, first_params)
            if results is None:
                self.logger.error("Initial request failed")
                return False
            
            # Save the first batch
            self._save_result(results, search_path, 1)
            self.current_page = 1
            
            # Get total results count
            total_results = self._get_total_results_count(results)
            if total_results is None:
                self.logger.error("Failed to get total results count")
                return False
                
            self.logger.info(f"Found {total_results} results")
            
            if total_results == 0:
                self.logger.info("No results found")
                return False
            
            # Calculate total pages
            self.total_pages = (total_results + self.count - 1) // self.count
            self.logger.info(f"Total pages to fetch: {self.total_pages}")
            
            # Report initial progress
            self._report_progress()
            
            # Move to the next page
            current_position = 2
        else:
            # We're resuming, so find the total pages from the first page file
            first_page_file = os.path.join(search_path, "results_page_1.xml")
            if os.path.exists(first_page_file):
                with open(first_page_file, 'r', encoding='utf-8') as f:
                    first_page_content = f.read()
                total_results = self._get_total_results_count(first_page_content)
                self.total_pages = (total_results + self.count - 1) // self.count
                self.logger.info(f"Resuming with {total_results} total results, {self.total_pages} total pages")
                
                # Report initial progress for resuming
                self._report_progress()
            else:
                self.logger.error("First page file not found. Cannot determine total pages")
                return False
        
        # Fetch remaining pages
        while current_position <= self.total_pages:
            # Check if search was cancelled
            if self.is_cancelled:
                self.logger.info("Search cancelled by user")
                return False
                
            self.logger.info(f"Fetching page {current_position} of {self.total_pages}")
            page_params = params.copy()
            page_params['start'] = str((current_position - 1) * self.count + 1)
            
            batch_results = self._make_request(endpoint, page_params)
            if batch_results is None:
                self.logger.error(f"Failed to fetch page {current_position}")
                break
            
            self._save_result(batch_results, search_path, current_position)
            current_position += 1
            self.current_page = current_position - 1
            
            # Report progress after each page
            self._report_progress()
            
            time.sleep(2)  # Be nice to the API
        
        self.logger.info(f"All data saved to {search_path}")
        return search_path 
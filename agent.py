import os
import sqlite3
import json
import logging
import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set, Union, Any
from pathlib import Path
import glob
import argparse
import signal
import sys

# Import the data retrieval modules
from ndl_search_retriever import NDLSearchAPI
from cinii_search_retriever import CiNiiSearchAPI
from data_processor import BiblioDataProcessor

class BiblioAgent:
    """
    Agent module for Biblio Explorer that coordinates the bibliographic network expansion
    """
    
    def __init__(self, 
                 db_path: Optional[str] = None, 
                 database_selection: str = "both",
                 person_name: Optional[str] = None,
                 title_keywords: Optional[str] = None,
                 start_year: Optional[int] = None,
                 end_year: Optional[int] = None,
                 ndc_number: Optional[str] = None,
                 publication_type: str = "books", 
                 min_publications: int = 2,
                 depth: int = 2,
                 prioritized_depth: Optional[List[str]] = None,
                 prioritized_depth_weight: float = 0.5,
                 max_authors: int = 100):
        """
        Initialize the Biblio Agent with search and control parameters.
        
        Args:
            db_path: Optional path to an existing database. If None, a new one will be created.
            database_selection: Which database to search ("ndl", "cinii", or "both")
            person_name: Name of the initial author (depth 0) to start network expansion
            title_keywords: Optional keywords to filter publications by title
            start_year: Optional start year for publication date filtering
            end_year: Optional end year for publication date filtering
            ndc_number: Optional NDC classification number
            publication_type: Type of publications to search for ("books", "articles", "all")
            min_publications: Minimum number of publications for an author to be included
            depth: Maximum depth for network expansion
            prioritized_depth: Which author depths to prioritize for calculating overlap priority
            prioritized_depth_weight: Weight for depth-based prioritization (0-1)
            max_authors: Maximum number of authors to search
        """
        # Set up logging
        self._setup_logging()
        
        # Store control parameters
        self.database_selection = database_selection
        self.depth = depth
        self.prioritized_depth = prioritized_depth
        self.prioritized_depth_weight = prioritized_depth_weight
        self.max_authors = max_authors
        self.min_publications = min_publications
        
        # Store search parameters
        self.person_name = person_name
        self.title_keywords = title_keywords
        self.start_year = start_year
        self.end_year = end_year
        self.ndc_number = ndc_number
        self.publication_type = publication_type
        
        # Initialize counters and state
        self.authors_searched = 0
        self.total_publications = 0
        self.is_terminated = False
        
        # Set up database (this must be done before initializing data retriever classes)
        self.db_path = self._initialize_database(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Set up data processor first so it can initialize the database
        self.data_processor = BiblioDataProcessor(
            db_path=self.db_path,
            gui_mode=False,
            log_level='INFO',
            http_log_level='WARNING'
        )
        
        # Set up data retrievers with properly configured parameters
        if database_selection in ["ndl", "both"]:
            self.ndl_api = NDLSearchAPI(
                output_dir="ndl_data",
                max_records=200,
                gui_mode=False
            )
        else:
            self.ndl_api = None
            
        if database_selection in ["cinii", "both"]:
            self.cinii_api = CiNiiSearchAPI(
                output_dir="cinii_data",
                count=100,
                gui_mode=False
            )
        else:
            self.cinii_api = None
        
        # Register signal handler for graceful termination
        signal.signal(signal.SIGINT, self._handle_termination)
        signal.signal(signal.SIGTERM, self._handle_termination)
        
        self.logger.info(f"BiblioAgent initialized with database: {self.db_path}")
        self.logger.info(f"Search parameters: database={database_selection}, person={person_name}, " 
                         f"title_keywords={title_keywords}, years={start_year}-{end_year}, "
                         f"ndc={ndc_number}, type={publication_type}")
        self.logger.info(f"Control parameters: depth={depth}, max_authors={max_authors}, "
                         f"prioritized_depth={prioritized_depth}, weight={prioritized_depth_weight}")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Create a logger
        self.logger = logging.getLogger('biblio_agent')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        log_file = f'logs/biblio_agent_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatter and add it to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _handle_termination(self, signum, frame):
        """Handle termination signals to gracefully shut down"""
        self.logger.info("Received termination signal, cleaning up...")
        self.is_terminated = True
        self.generate_output()
        self.close()
        sys.exit(0)
    
    def _initialize_database(self, db_path: Optional[str]) -> str:
        """
        Initialize the database - either create a new one or use an existing one.
        
        Args:
            db_path: Path to an existing database, or None to create a new one
            
        Returns:
            Path to the database being used
        """
        # If no db_path provided, find existing databases or create a new one
        if db_path is None:
            # Find existing network databases
            existing_dbs = sorted(glob.glob("network_data_*.db"))
            
            if not existing_dbs:
                # Create a new database
                db_path = f"network_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                self.logger.info(f"Creating new database: {db_path}")
                return db_path
            
            # Default to the newest database
            newest_db = existing_dbs[-1]
            
            # Ask user for confirmation
            if len(existing_dbs) == 1:
                choice = input(f"Found existing database: {newest_db}. Use it? (Y/n) ").strip().lower()
                if choice in ['', 'y', 'yes']:
                    self.logger.info(f"Using existing database: {newest_db}")
                    return newest_db
            else:
                print("Found existing databases:")
                for i, db in enumerate(existing_dbs):
                    print(f"{i+1}. {db}")
                print(f"{len(existing_dbs)+1}. Create a new database")
                
                choice = input(f"Choose a database (1-{len(existing_dbs)+1}, default={len(existing_dbs)}): ").strip()
                
                if choice == "":
                    choice = str(len(existing_dbs))  # Default to newest
                
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(existing_dbs):
                        self.logger.info(f"Using existing database: {existing_dbs[choice_idx]}")
                        return existing_dbs[choice_idx]
                except ValueError:
                    pass
            
            # If we reach here, create a new database
            db_path = f"network_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            self.logger.info(f"Creating new database: {db_path}")
            return db_path
        
        # Use the provided db_path
        self.logger.info(f"Using provided database: {db_path}")
        return db_path
    
    def _setup_database_tables(self):
        """Set up the network expansion specific tables in the database"""
        # Create a table for storing network expansion metadata
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS network_expansion_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            search_params TEXT,
            control_params TEXT,
            timestamp TEXT
        )
        ''')
        
        # Create a table for storing network expansion status
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS network_expansion_status (
            author_id TEXT PRIMARY KEY,
            author_name_cleaned TEXT,
            depth INTEGER,
            priority REAL,
            status TEXT,
            status_timestamp TEXT,
            parent_author TEXT
        )
        ''')
        
        self.conn.commit()
    
    def _save_expansion_metadata(self):
        """Save the current search and control parameters to the database"""
        search_params = {
            'database_selection': self.database_selection,
            'person_name': self.person_name,
            'title_keywords': self.title_keywords,
            'start_year': self.start_year,
            'end_year': self.end_year,
            'ndc_number': self.ndc_number,
            'publication_type': self.publication_type,
            'min_publications': self.min_publications
        }
        
        control_params = {
            'depth': self.depth,
            'prioritized_depth': self.prioritized_depth,
            'prioritized_depth_weight': self.prioritized_depth_weight,
            'max_authors': self.max_authors
        }
        
        self.cursor.execute(
            'INSERT INTO network_expansion_metadata (search_params, control_params, timestamp) VALUES (?, ?, ?)',
            (json.dumps(search_params), json.dumps(control_params), datetime.now().isoformat())
        )
        
        self.conn.commit()
    
    def _load_expansion_metadata(self):
        """Load the most recent expansion metadata from the database"""
        self.cursor.execute('SELECT search_params, control_params FROM network_expansion_metadata ORDER BY id DESC LIMIT 1')
        row = self.cursor.fetchone()
        
        if row:
            search_params = json.loads(row[0])
            control_params = json.loads(row[1])
            
            # Only update parameters if they weren't already provided in the constructor
            if self.person_name is None and 'person_name' in search_params:
                self.person_name = search_params['person_name']
            
            if self.title_keywords is None and 'title_keywords' in search_params:
                self.title_keywords = search_params['title_keywords']
            
            if self.start_year is None and 'start_year' in search_params:
                self.start_year = search_params['start_year']
            
            if self.end_year is None and 'end_year' in search_params:
                self.end_year = search_params['end_year']
            
            if self.ndc_number is None and 'ndc_number' in search_params:
                self.ndc_number = search_params['ndc_number']
            
            # Always save the current parameter state
            self._save_expansion_metadata()
            
            return True
        
        return False
    
    def close(self):
        """Close database connections and clean up resources"""
        if self.conn:
            self.conn.close()
        if self.data_processor:
            self.data_processor.close()
    
    def start_network_expansion(self):
        """Start or resume the network expansion process"""
        # Set up database tables if this is a new database
        self._setup_database_tables()
        
        # Load previous metadata if available
        is_resuming = self._load_expansion_metadata()
        
        # If this is a new run (not resuming), initialize with the seed author
        if not is_resuming and self.person_name:
            self._initialize_network_with_seed_author()
        
        # Start the expansion loop
        self._expand_network()
    
    def _initialize_network_with_seed_author(self):
        """Initialize the network with the seed author (depth 0)"""
        self.logger.info(f"Initializing network with seed author: {self.person_name}")
        
        # First, use the search APIs to find publications for the seed author
        # The _search_for_author method will use the data_processor to process the results
        results = self._search_for_author(self.person_name)
        
        if not results:
            self.logger.error(f"Could not find any publications for seed author: {self.person_name}")
            return False
        
        # Give the database a moment to process the results
        time.sleep(1)
        
        # Find the author in the database by leveraging BiblioDataProcessor's name cleaning
        # and our database queries to ensure proper integration
        author_data = self._get_author_data_from_name(self.person_name)
        
        if not author_data:
            self.logger.error(f"Could not find author record for seed author: {self.person_name}")
            self.logger.info("Listing a sample of authors in the database to help troubleshoot:")
            
            # List some authors from the database to help troubleshoot
            self.cursor.execute('SELECT id, name, cleaned_name FROM entities WHERE is_author = 1 LIMIT 10')
            authors = self.cursor.fetchall()
            if authors:
                for author in authors:
                    self.logger.info(f"  ID: {author[0]}, Name: {author[1]}, Cleaned: {author[2]}")
            else:
                self.logger.error("No authors found in the database. The search may have failed.")
            
            return False
        
        author_id, author_cleaned_name = author_data
        
        # Add the seed author to the expansion status table with depth 0
        self.cursor.execute(
            'INSERT OR IGNORE INTO network_expansion_status (author_id, author_name_cleaned, depth, priority, status, status_timestamp, parent_author) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (author_id, author_cleaned_name, 0, 1.0, 'pending', datetime.now().isoformat(), '')
        )
        
        self.conn.commit()
        self.logger.info(f"Network initialized with seed author: {author_cleaned_name} (ID: {author_id})")
        return True
    
    def _search_for_author(self, author_name: str) -> bool:
        """
        Search for publications by an author and process the results
        
        Args:
            author_name: Name of the author to search for
            
        Returns:
            Boolean indicating if the search was successful
        """
        self.logger.info(f"Searching for publications by author: {author_name}")
        
        search_successful = False
        
        # Handle NDL Search if enabled
        if self.ndl_api:
            # Set up search parameters
            kwargs = {}
            if self.title_keywords:
                kwargs['title'] = self.title_keywords
            if self.start_year:
                kwargs['from_year'] = str(self.start_year)
            if self.end_year:
                kwargs['until_year'] = str(self.end_year)
            if self.ndc_number:
                kwargs['ndc'] = self.ndc_number
            
            # Using the NDLSearchAPI as a complete object
            try:
                # Perform the search
                search_result = self.ndl_api.search_by_creator(author_name, **kwargs)
                
                if search_result:
                    # Export to JSON and get the file path
                    json_file_path = self.ndl_api.export_results_to_json(search_result)
                    
                    if json_file_path and isinstance(json_file_path, str):
                        # Use the data processor as a complete object to process the file
                        self.data_processor.process_json_file(json_file_path)
                        self.logger.info(f"Processed NDL search results for author: {author_name}")
                        search_successful = True
            except Exception as e:
                self.logger.error(f"Error in NDL search: {str(e)}")
        
        # Handle CiNii Search if enabled
        if self.cinii_api:
            # Set up search parameters
            kwargs = {}
            if self.title_keywords:
                kwargs['title'] = self.title_keywords
            if self.start_year:
                kwargs['from'] = str(self.start_year)
            if self.end_year:
                kwargs['until'] = str(self.end_year)
            
            # Determine search type based on publication_type
            search_type = 'books' if self.publication_type == 'books' else 'all'
            
            # Using the CiNiiSearchAPI as a complete object
            try:
                # Perform the search and export in a single operation
                search_result = self.cinii_api.search(
                    search_type=search_type,
                    creator=author_name,
                    **kwargs
                )
                
                if search_result:
                    # Export to JSON
                    export_success = self.cinii_api.export_results_to_json(search_result)
                    
                    if export_success:
                        # Get the full path to the exported JSON file
                        json_file_path = os.path.join(search_result, "cinii_records.json")
                        
                        # Process the JSON file with the data processor
                        self.data_processor.process_json_file(json_file_path)
                        self.logger.info(f"Processed CiNii search results for author: {author_name}")
                        search_successful = True
            except Exception as e:
                self.logger.error(f"Error in CiNii search: {str(e)}")
        
        return search_successful
    
    def _get_author_data_from_name(self, author_name: str) -> Optional[Tuple[str, str]]:
        """
        Get author ID and cleaned name from the entities table using BiblioDataProcessor
        to handle the name cleaning and entity searching.
        
        Args:
            author_name: Name of the author to look up
            
        Returns:
            Tuple of (author_id, cleaned_name) or None if not found
        """
        self.logger.info(f"Looking for author data for: {author_name}")
        
        # Step 1: Use BiblioDataProcessor to clean the name
        try:
            # The data_processor is already initialized with our database
            cleaned_name = self.data_processor.clean_japanese_name(author_name)
            self.logger.info(f"Original name: '{author_name}', Cleaned name: '{cleaned_name}'")
        except Exception as e:
            self.logger.error(f"Error cleaning author name: {e}")
            cleaned_name = author_name
        
        # Step 2: Search for entity candidates using SQL
        # Query to find entities by name or cleaned name, filtering for authors
        query = """
        SELECT id, name, cleaned_name FROM entities 
        WHERE is_author = 1 AND (name LIKE ? OR cleaned_name LIKE ?)
        """
        
        # Try direct match first
        self.cursor.execute(
            "SELECT id, cleaned_name FROM entities WHERE is_author = 1 AND (name = ? OR cleaned_name = ?)",
            (author_name, cleaned_name)
        )
        
        result = self.cursor.fetchone()
        if result:
            self.logger.info(f"Found exact author match: {result[1]} (ID: {result[0]})")
            return (result[0], result[1])
        
        # If no direct match, try partial match using wildcards
        search_pattern = f"%{author_name}%"
        cleaned_pattern = f"%{cleaned_name}%"
        
        self.cursor.execute(query, (search_pattern, cleaned_pattern))
        results = self.cursor.fetchall()
        
        if not results:
            self.logger.error(f"No author entities found for {author_name}")
            return None
        
        # Log the candidates
        self.logger.info(f"Found {len(results)} potential author matches:")
        for result in results:
            self.logger.debug(f"  ID: {result[0]}, Name: {result[1]}, Cleaned: {result[2]}")
        
        # Step 3: Find the best match using similarity
        best_match = None
        best_similarity = 0.0
        
        for entity_id, entity_name, entity_cleaned_name in results:
            # Calculate similarity between search term and entity name/cleaned name
            name_similarity = self._calculate_name_similarity(author_name, entity_name)
            cleaned_similarity = self._calculate_name_similarity(cleaned_name, entity_cleaned_name)
            
            # Take the highest similarity
            similarity = max(name_similarity, cleaned_similarity)
            
            # Update best match if this is better
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (entity_id, entity_cleaned_name)
        
        if best_match:
            self.logger.info(f"Best author match: {best_match[1]} (ID: {best_match[0]}, similarity: {best_similarity:.2f})")
            return best_match
        
        return None
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two names using simple character overlap
        
        Args:
            name1: First name
            name2: Second name
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not name1 or not name2:
            return 0.0
        
        # Convert to sets of characters for comparison
        set1 = set(name1)
        set2 = set(name2)
        
        # Calculate Jaccard similarity: intersection / union
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _expand_network(self):
        """Main loop for expanding the network based on author priorities"""
        self.logger.info("Starting network expansion process")
        
        iteration_count = 0
        
        while not self.is_terminated:
            # Check termination conditions
            if self._check_termination_conditions():
                self.logger.info("Termination condition reached. Stopping network expansion.")
                break
            
            # Get the highest priority pending author
            author = self._get_highest_priority_author()
            
            if not author:
                self.logger.info("No more pending authors to process. Network expansion complete.")
                break
            
            author_id, author_name, depth, priority = author
            
            # Update author status to 'in_progress'
            self._update_author_status(author_id, 'in_progress')
            
            # Search for author's publications
            self.logger.info(f"Processing author: {author_name} (depth: {depth}, priority: {priority:.4f})")
            search_successful = self._search_for_author(author_name)
            
            if search_successful:
                # Get coauthors
                coauthors = self._get_coauthors(author_id)
                self.logger.info(f"Found {len(coauthors)} coauthors for {author_name}")
                
                # Process coauthors
                self._process_coauthors(author_id, depth, coauthors)
                
                # Update author status to 'completed'
                self._update_author_status(author_id, 'completed')
                
                # Increment counter
                self.authors_searched += 1
            else:
                # If search failed, mark the author as 'failed'
                self._update_author_status(author_id, 'failed')
                self.logger.warning(f"Failed to retrieve publications for author: {author_name}")
            
            # Generate periodic output
            iteration_count += 1
            if iteration_count % 10 == 0:
                self.logger.info(f"Processed {iteration_count} authors so far. Generating interim output.")
                self.generate_output()
                
            # Small delay to avoid overwhelming the APIs
            time.sleep(1)
        
        # Final output generation
        self.generate_output()
        self.logger.info(f"Network expansion completed. Processed {self.authors_searched} authors.")
    
    def _check_termination_conditions(self) -> bool:
        """Check if any termination conditions are met"""
        # 1. Check if max authors reached
        if self.authors_searched >= self.max_authors:
            self.logger.info(f"Maximum number of authors ({self.max_authors}) reached.")
            return True
        
        # 2. Check for isolated branch termination
        if self.prioritized_depth and self.prioritized_depth_weight > 0:
            highest_priority = self._get_highest_pending_priority()
            if highest_priority is not None:
                current_depth = self._get_current_depth()
                if current_depth >= 1:
                    depth_priority = (1 - self.prioritized_depth_weight) * (0.9 ** current_depth)
                    if abs(highest_priority - depth_priority) < 0.0001:  # Nearly equal
                        self.logger.info("Isolated branch termination condition met.")
                        return True
        
        # 3. Check if all authors at max depth have been processed
        max_depth_authors = self._count_authors_at_depth(self.depth)
        if max_depth_authors == 0:
            # Check if we have any pending authors
            pending_count = self._count_pending_authors()
            if pending_count == 0:
                self.logger.info(f"All authors up to maximum depth ({self.depth}) have been processed.")
                return True
        
        return False
    
    def _get_highest_priority_author(self) -> Optional[Tuple[str, str, int, float]]:
        """Get the highest priority pending author for processing"""
        self.cursor.execute(
            'SELECT author_id, author_name_cleaned, depth, priority FROM network_expansion_status '
            'WHERE status = "pending" ORDER BY priority DESC LIMIT 1'
        )
        
        result = self.cursor.fetchone()
        
        if result:
            return result
        
        return None
    
    def _update_author_status(self, author_id: str, status: str):
        """Update an author's status in the database"""
        self.cursor.execute(
            'UPDATE network_expansion_status SET status = ?, status_timestamp = ? WHERE author_id = ?',
            (status, datetime.now().isoformat(), author_id)
        )
        
        self.conn.commit()
    
    def _get_coauthors(self, author_id: str) -> List[Tuple[str, str, int]]:
        """
        Get an author's coauthors from the database
        
        Returns:
            List of tuples containing (coauthor_id, coauthor_name, publication_count)
        """
        # Find all publications by the author
        self.cursor.execute(
            'SELECT item_id FROM item_entities WHERE entity_id = ? AND relationship_type = "author"',
            (author_id,)
        )
        
        publication_ids = [row[0] for row in self.cursor.fetchall()]
        
        if not publication_ids:
            return []
        
        # Find coauthors for these publications, excluding the author themselves
        coauthors = {}
        
        for item_id in publication_ids:
            self.cursor.execute(
                'SELECT e.id, e.cleaned_name FROM entities e '
                'JOIN item_entities ie ON e.id = ie.entity_id '
                'WHERE ie.item_id = ? AND ie.relationship_type = "author" AND e.id != ?',
                (item_id, author_id)
            )
            
            for coauthor_id, coauthor_name in self.cursor.fetchall():
                if coauthor_id in coauthors:
                    coauthors[coauthor_id]['count'] += 1
                else:
                    coauthors[coauthor_id] = {
                        'name': coauthor_name,
                        'count': 1
                    }
        
        # Filter coauthors by minimum publication count
        filtered_coauthors = []
        for coauthor_id, data in coauthors.items():
            if data['count'] >= self.min_publications:
                filtered_coauthors.append((coauthor_id, data['name'], data['count']))
        
        return filtered_coauthors
    
    def _process_coauthors(self, parent_author_id: str, parent_depth: int, coauthors: List[Tuple[str, str, int]]):
        """
        Process coauthors of an author, calculating priorities and adding to the queue
        
        Uses the BiblioDataProcessor's functionality to help calculate priorities based on
        network connections and author relationships.
        
        Args:
            parent_author_id: ID of the parent author
            parent_depth: Depth of the parent author
            coauthors: List of coauthor data tuples (id, name, publication count)
        """
        # Get the parent author's name
        self.cursor.execute(
            'SELECT author_name_cleaned FROM network_expansion_status WHERE author_id = ?',
            (parent_author_id,)
        )
        parent_name = self.cursor.fetchone()[0]
        
        # Calculate the new depth
        new_depth = parent_depth + 1
        
        if new_depth > self.depth:
            self.logger.info(f"Maximum depth ({self.depth}) reached. Not expanding further.")
            return
        
        self.logger.info(f"Processing {len(coauthors)} coauthors of {parent_name} at depth {parent_depth}")
        
        # For each coauthor, check if they exist in the network expansion status
        for coauthor_id, coauthor_name, pub_count in coauthors:
            # Skip if this is the parent's parent (avoid cycles)
            self.cursor.execute(
                'SELECT parent_author FROM network_expansion_status WHERE author_id = ?',
                (parent_author_id,)
            )
            parent_parent = self.cursor.fetchone()
            if parent_parent and parent_parent[0] == coauthor_id:
                self.logger.debug(f"Skipping coauthor {coauthor_name} as it is the parent's parent")
                continue
            
            # Check if coauthor already exists in expansion status
            self.cursor.execute(
                'SELECT status, depth, priority FROM network_expansion_status WHERE author_id = ?',
                (coauthor_id,)
            )
            
            existing_coauthor = self.cursor.fetchone()
            
            if existing_coauthor:
                status, coauthor_depth, current_priority = existing_coauthor
                
                # Only update priority if author is still pending
                if status == 'pending':
                    # Calculate priorities using our helper methods that integrate with BiblioDataProcessor
                    overlap_priority = self._calculate_overlap_priority(coauthor_id)
                    depth_priority = 0.9 ** coauthor_depth
                    
                    # Calculate combined priority
                    if self.prioritized_depth and self.prioritized_depth is not None and self.prioritized_depth != ["off"]:
                        combined_priority = (1 - self.prioritized_depth_weight) * depth_priority + self.prioritized_depth_weight * overlap_priority
                    else:
                        combined_priority = depth_priority
                    
                    # Update priority if new priority is higher
                    new_priority = max(current_priority, combined_priority)
                    if new_priority != current_priority:
                        self.cursor.execute(
                            'UPDATE network_expansion_status SET priority = ? WHERE author_id = ?',
                            (new_priority, coauthor_id)
                        )
                        self.logger.debug(f"Updated priority for existing coauthor {coauthor_name} from {current_priority:.4f} to {new_priority:.4f}")
            else:
                # This is a new coauthor, calculate initial priority
                # First get the cleaned name using BiblioDataProcessor 
                try:
                    # We already have the coauthor's cleaned name from the database
                    # But log it for clarity
                    self.logger.debug(f"Processing new coauthor: {coauthor_name} (ID: {coauthor_id})")
                    
                    # Calculate priority values
                    depth_priority = 0.9 ** new_depth
                    overlap_priority = self._calculate_overlap_priority(coauthor_id)
                    
                    # Calculate combined priority
                    if self.prioritized_depth and self.prioritized_depth is not None and self.prioritized_depth != ["off"]:
                        combined_priority = (1 - self.prioritized_depth_weight) * depth_priority + self.prioritized_depth_weight * overlap_priority
                    else:
                        combined_priority = depth_priority
                    
                    # Add coauthor to expansion status
                    self.cursor.execute(
                        'INSERT INTO network_expansion_status (author_id, author_name_cleaned, depth, priority, status, status_timestamp, parent_author) '
                        'VALUES (?, ?, ?, ?, ?, ?, ?)',
                        (coauthor_id, coauthor_name, new_depth, combined_priority, 'pending', datetime.now().isoformat(), parent_author_id)
                    )
                    
                    self.logger.debug(f"Added new coauthor {coauthor_name} at depth {new_depth} with priority {combined_priority:.4f}")
                except Exception as e:
                    self.logger.error(f"Error processing coauthor {coauthor_name}: {str(e)}")
        
        # Commit all the changes at once for better performance
        self.conn.commit()
    
    def _calculate_overlap_priority(self, author_id: str) -> float:
        """
        Calculate overlap-based priority for an author based on how many of their 
        coauthors are already in the prioritized depths
        """
        if not self.prioritized_depth or self.prioritized_depth == ["off"] or self.prioritized_depth is None:
            return 0.0
        
        # Get all coauthors of this author
        self.cursor.execute(
            'SELECT DISTINCT e.id FROM entities e '
            'JOIN item_entities ie1 ON e.id = ie1.entity_id '
            'JOIN item_entities ie2 ON ie1.item_id = ie2.item_id '
            'WHERE ie2.entity_id = ? AND ie1.entity_id != ? AND ie1.relationship_type = "author" AND ie2.relationship_type = "author"',
            (author_id, author_id)
        )
        
        all_coauthors = set(row[0] for row in self.cursor.fetchall())
        
        if not all_coauthors:
            return 0.0
        
        # Get depths to prioritize
        prioritized_depths = self._interpret_prioritized_depths()
        
        # Get authors at the prioritized depths
        placeholders = ','.join(['?' for _ in range(len(prioritized_depths))])
        self.cursor.execute(
            f'SELECT author_id FROM network_expansion_status WHERE depth IN ({placeholders})',
            prioritized_depths
        )
        
        prioritized_authors = set(row[0] for row in self.cursor.fetchall())
        
        # Calculate overlap
        overlap = len(all_coauthors.intersection(prioritized_authors))
        
        # Return overlap ratio
        return overlap / len(all_coauthors) if all_coauthors else 0.0
    
    def _interpret_prioritized_depths(self) -> List[int]:
        """Interpret the prioritized_depth parameter into a list of actual depth values"""
        if not self.prioritized_depth or self.prioritized_depth == ["off"] or self.prioritized_depth is None:
            return []
        
        depths = []
        
        # Get current maximum depth
        self.cursor.execute('SELECT MAX(depth) FROM network_expansion_status')
        current_max_depth = self.cursor.fetchone()[0] or 0
        
        # Interpret each value in the prioritized_depth list
        for depth_str in self.prioritized_depth:
            if depth_str == "all":
                # All depths from 0 to current max
                depths.extend(list(range(current_max_depth + 1)))
            elif depth_str.startswith("-"):
                # Relative depths (e.g., -2 means current and previous depth)
                try:
                    relative = int(depth_str)
                    current_depth = current_max_depth
                    for i in range(abs(relative)):
                        if current_depth - i >= 0:
                            depths.append(current_depth - i)
                except ValueError:
                    continue
            else:
                # Specific depths
                try:
                    if "," in depth_str:
                        for d in depth_str.split(","):
                            depths.append(int(d.strip()))
                    else:
                        depths.append(int(depth_str))
                except ValueError:
                    continue
        
        # Remove duplicates and sort
        return sorted(list(set(depths)))
    
    def _get_highest_pending_priority(self) -> Optional[float]:
        """Get the highest priority among pending authors"""
        self.cursor.execute(
            'SELECT MAX(priority) FROM network_expansion_status WHERE status = "pending"'
        )
        result = self.cursor.fetchone()
        return result[0] if result and result[0] is not None else None
    
    def _get_current_depth(self) -> int:
        """Get the current maximum depth being processed"""
        self.cursor.execute(
            'SELECT MAX(depth) FROM network_expansion_status WHERE status IN ("completed", "in_progress")'
        )
        result = self.cursor.fetchone()
        return result[0] if result and result[0] is not None else 0
    
    def _count_authors_at_depth(self, depth: int) -> int:
        """Count the number of authors at a specific depth"""
        self.cursor.execute(
            'SELECT COUNT(*) FROM network_expansion_status WHERE depth = ?',
            (depth,)
        )
        result = self.cursor.fetchone()
        return result[0] if result else 0
    
    def _count_pending_authors(self) -> int:
        """Count the number of pending authors"""
        self.cursor.execute(
            'SELECT COUNT(*) FROM network_expansion_status WHERE status = "pending"'
        )
        result = self.cursor.fetchone()
        return result[0] if result else 0
    
    def generate_output(self):
        """Generate network output files in standard graph formats"""
        self.logger.info("Generating network output files...")
        
        # Generate GraphML format
        self._generate_graphml()
        
        # Generate JSON format
        self._generate_json()
        
        # Generate tree diagram
        self._generate_tree_diagram()
    
    def _generate_graphml(self):
        """Generate a GraphML file representing the co-authorship network"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"coauthor_network_{timestamp}.graphml"
        
        # Create GraphML header
        graphml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        graphml += '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n'
        graphml += '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        graphml += '         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n'
        graphml += '         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n'
        
        # Define node attributes
        graphml += '  <key id="d0" for="node" attr.name="name" attr.type="string"/>\n'
        graphml += '  <key id="d1" for="node" attr.name="depth" attr.type="int"/>\n'
        graphml += '  <key id="d2" for="node" attr.name="status" attr.type="string"/>\n'
        graphml += '  <key id="d3" for="node" attr.name="priority" attr.type="double"/>\n'
        graphml += '  <key id="d4" for="edge" attr.name="weight" attr.type="int"/>\n'
        
        # Start graph
        graphml += '  <graph id="G" edgedefault="undirected">\n'
        
        # Add nodes
        self.cursor.execute(
            'SELECT author_id, author_name_cleaned, depth, status, priority FROM network_expansion_status'
        )
        
        for author_id, name, depth, status, priority in self.cursor.fetchall():
            graphml += f'    <node id="{author_id}">\n'
            graphml += f'      <data key="d0">{name}</data>\n'
            graphml += f'      <data key="d1">{depth}</data>\n'
            graphml += f'      <data key="d2">{status}</data>\n'
            graphml += f'      <data key="d3">{priority}</data>\n'
            graphml += '    </node>\n'
        
        # Add edges (co-authorship relationships)
        # This query finds pairs of authors who have collaborated on at least one publication
        self.cursor.execute('''
            SELECT e1.entity_id AS author1, e2.entity_id AS author2, COUNT(*) AS weight
            FROM item_entities e1
            JOIN item_entities e2 ON e1.item_id = e2.item_id
            WHERE e1.relationship_type = 'author' AND e2.relationship_type = 'author'
            AND e1.entity_id < e2.entity_id  -- Avoid duplicate edges
            GROUP BY e1.entity_id, e2.entity_id
            HAVING COUNT(*) >= ?
        ''', (self.min_publications,))
        
        for author1, author2, weight in self.cursor.fetchall():
            graphml += f'    <edge source="{author1}" target="{author2}">\n'
            graphml += f'      <data key="d4">{weight}</data>\n'
            graphml += '    </edge>\n'
        
        # Close graph and GraphML
        graphml += '  </graph>\n'
        graphml += '</graphml>\n'
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(graphml)
        
        self.logger.info(f"Network saved as GraphML: {output_file}")
    
    def _generate_json(self):
        """Generate a JSON file representing the co-authorship network"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"coauthor_network_{timestamp}.json"
        
        # Create network data structure
        network = {
            "metadata": {
                "generated": datetime.now().isoformat(),
                "database": self.db_path,
                "parameters": {
                    "search": {
                        "database_selection": self.database_selection,
                        "person_name": self.person_name,
                        "title_keywords": self.title_keywords,
                        "start_year": self.start_year,
                        "end_year": self.end_year,
                        "ndc_number": self.ndc_number,
                        "publication_type": self.publication_type,
                        "min_publications": self.min_publications
                    },
                    "control": {
                        "depth": self.depth,
                        "prioritized_depth": self.prioritized_depth,
                        "prioritized_depth_weight": self.prioritized_depth_weight,
                        "max_authors": self.max_authors
                    }
                },
                "stats": {
                    "authors_searched": self.authors_searched,
                    "total_nodes": 0,
                    "total_edges": 0
                }
            },
            "nodes": [],
            "edges": []
        }
        
        # Add nodes
        self.cursor.execute(
            'SELECT author_id, author_name_cleaned, depth, status, priority, parent_author FROM network_expansion_status'
        )
        
        for author_id, name, depth, status, priority, parent_author in self.cursor.fetchall():
            node = {
                "id": author_id,
                "name": name,
                "depth": depth,
                "status": status,
                "priority": priority,
                "parent_author": parent_author
            }
            network["nodes"].append(node)
        
        # Add edges
        self.cursor.execute('''
            SELECT e1.entity_id AS author1, e2.entity_id AS author2, COUNT(*) AS weight
            FROM item_entities e1
            JOIN item_entities e2 ON e1.item_id = e2.item_id
            WHERE e1.relationship_type = 'author' AND e2.relationship_type = 'author'
            AND e1.entity_id < e2.entity_id  -- Avoid duplicate edges
            GROUP BY e1.entity_id, e2.entity_id
            HAVING COUNT(*) >= ?
        ''', (self.min_publications,))
        
        for author1, author2, weight in self.cursor.fetchall():
            edge = {
                "source": author1,
                "target": author2,
                "weight": weight
            }
            network["edges"].append(edge)
        
        # Update stats
        network["metadata"]["stats"]["total_nodes"] = len(network["nodes"])
        network["metadata"]["stats"]["total_edges"] = len(network["edges"])
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(network, f, indent=2)
        
        self.logger.info(f"Network saved as JSON: {output_file}")
    
    def _generate_tree_diagram(self):
        """Generate a tree diagram starting from the initial author"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"author_tree_{timestamp}.txt"
        
        # Find the root author (depth 0)
        self.cursor.execute(
            'SELECT author_id, author_name_cleaned FROM network_expansion_status WHERE depth = 0 LIMIT 1'
        )
        root = self.cursor.fetchone()
        
        if not root:
            self.logger.warning("Could not find root author (depth 0) for tree diagram")
            return
        
        root_id, root_name = root
        
        # Build the tree recursively
        tree_text = f"{root_name}\n"
        self._build_tree(root_id, "", tree_text, output_file)
        
        self.logger.info(f"Author tree saved as: {output_file}")
    
    def _build_tree(self, parent_id: str, prefix: str, tree_text: str, output_file: str, max_depth: int = 10):
        """
        Recursively build the tree diagram
        
        Args:
            parent_id: ID of the parent author
            prefix: String prefix for indentation
            tree_text: Current tree text
            output_file: Path to output file
            max_depth: Maximum recursion depth
        """
        if max_depth <= 0:
            return
        
        # Find children of this author
        self.cursor.execute(
            'SELECT author_id, author_name_cleaned FROM network_expansion_status WHERE parent_author = ? ORDER BY priority DESC',
            (parent_id,)
        )
        
        children = self.cursor.fetchall()
        
        # Write the current tree to file (to handle very large trees)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(tree_text)
        
        # Process each child
        for i, (child_id, child_name) in enumerate(children):
            is_last = (i == len(children) - 1)
            
            # Add this child to the tree
            with open(output_file, 'a', encoding='utf-8') as f:
                if is_last:
                    f.write(f"{prefix}└── {child_name}\n")
                    new_prefix = prefix + "    "
                else:
                    f.write(f"{prefix}├── {child_name}\n")
                    new_prefix = prefix + "│   "
            
            # Recursively process this child's children
            self._build_tree(child_id, new_prefix, "", output_file, max_depth - 1)


def main():
    """Main function to run the Biblio Agent"""
    parser = argparse.ArgumentParser(description="Biblio Explorer Agent - Coauthorship Network Expansion")
    
    # Required parameters
    parser.add_argument("--database", choices=["ndl", "cinii", "both"], default="both",
                        help="Database to search (NDL Search, CiNii, or both)")
    
    # Optional parameters with defaults
    parser.add_argument("--db_path", type=str, help="Path to an existing database file")
    parser.add_argument("--person", type=str, help="Initial author name to start expansion from")
    parser.add_argument("--title", type=str, help="Keywords to filter publications by title")
    parser.add_argument("--start_year", type=int, help="Start year for publication filtering")
    parser.add_argument("--end_year", type=int, help="End year for publication filtering")
    parser.add_argument("--ndc", type=str, help="NDC classification number")
    parser.add_argument("--type", choices=["books", "articles", "all"], default="books",
                        help="Type of publications to search for")
    parser.add_argument("--min_pubs", type=int, default=2,
                        help="Minimum number of publications for including authors")
    parser.add_argument("--depth", type=int, default=2,
                        help="Maximum depth for network expansion")
    parser.add_argument("--priority_depth", nargs="+", default=None,
                        help="Author depths to consider for calculating priority (e.g., '0', '1,2', '-2', or 'all')")
    parser.add_argument("--priority_weight", type=float, default=0.5,
                        help="Weight for depth-based prioritization (0-1)")
    parser.add_argument("--max_authors", type=int, default=100,
                        help="Maximum number of authors to search")
    
    args = parser.parse_args()
    
    # Create and run the agent
    agent = BiblioAgent(
        db_path=args.db_path,
        database_selection=args.database,
        person_name=args.person,
        title_keywords=args.title,
        start_year=args.start_year,
        end_year=args.end_year,
        ndc_number=args.ndc,
        publication_type=args.type,
        min_publications=args.min_pubs,
        depth=args.depth,
        prioritized_depth=args.priority_depth,
        prioritized_depth_weight=args.priority_weight,
        max_authors=args.max_authors
    )
    
    try:
        agent.start_network_expansion()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Saving current state and exiting...")
        agent.generate_output()
    finally:
        agent.close()
        print("Biblio Agent finished.")


if __name__ == "__main__":
    main()
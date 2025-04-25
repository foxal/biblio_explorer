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
                 author_from_rdf: bool = True,
                 title_keywords: Optional[str] = None,
                 start_year: Optional[int] = None,
                 end_year: Optional[int] = None,
                 ndc_number: Optional[str] = None,
                 publication_type: List[str] = ["books"],
                 min_publications: int = 1,
                 entity_types_in_network: Optional[List[str]] = None,
                 depth: int = 3,
                 prioritized_depth: Optional[List[str]] = None,
                 prioritized_depth_weight: float = 0.8,
                 max_authors: int = 200):
        """
        Initialize the Biblio Agent with search and control parameters.

        Args:
            db_path: Optional path to an existing database. If None, a new one will be created
            database_selection: Which database to search ("ndl", "cinii", or "both")
            person_name: Name of the initial author (depth 0) to start network expansion
            author_from_rdf: Whether to extract detailed author information from RDF files and use the author ID there as the author's primary identifier
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
        self.entity_types_in_network = entity_types_in_network or ["person"]

        # Store search parameters
        self.person_name = person_name
        self.author_from_rdf = author_from_rdf
        self.title_keywords = title_keywords
        self.start_year = start_year
        self.end_year = end_year
        self.ndc_number = ndc_number
        self.publication_type = publication_type


        # Initialize counters and state
        self.authors_searched = 0
        self.total_publications = 0
        self.specific_prioritized_authors = []
        self.is_terminated = False

        # Set up database (this must be done before initializing data retriever classes)
        self.db_path = self._initialize_database(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Set up data processor first so it can initialize the database
        self.data_processor = BiblioDataProcessor(
            db_path=self.db_path,
            use_method2=True,
            auto_type_check=2,
            dedup_threshold=1.1, # When larger than 1, no deduplication is performed
            gui_mode=False,
            log_level='WARNING',
            http_log_level='WARNING'
        )

        # Set up data retrievers with properly configured parameters
        if database_selection in ["ndl", "both"]:
            self.ndl_api = NDLSearchAPI(
                output_dir="ndl_data",
                max_records=200,
                gui_mode=False,
                log_level='WARNING'
            )
        else:
            self.ndl_api = None

        if database_selection in ["cinii", "both"]:
            self.cinii_api = CiNiiSearchAPI(
                output_dir="cinii_data",
                author_from_rdf=self.author_from_rdf,
                count=100,
                gui_mode=False,
                log_level='WARNING'
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
            'max_authors': self.max_authors,
            'entity_types_in_network': self.entity_types_in_network
        }

        self.cursor.execute(
            'INSERT INTO network_expansion_metadata (search_params, control_params, timestamp) VALUES (?, ?, ?)',
            (json.dumps(search_params, ensure_ascii=False), json.dumps(control_params, ensure_ascii=False), datetime.now().isoformat())
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
            # check search parameters
            if self.database_selection is None and 'database_selection' in search_params:
                self.database_selection = search_params['database_selection']

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

            if self.publication_type is None and 'publication_type' in search_params:
                self.publication_type = search_params['publication_type']

            if self.min_publications is None and 'min_publications' in search_params:
                self.min_publications = search_params['min_publications']

            # check control parameters
            if self.depth is None and 'depth' in control_params:
                self.depth = control_params['depth']

            if self.prioritized_depth is None and 'prioritized_depth' in control_params:
                self.prioritized_depth = control_params['prioritized_depth']

            if self.prioritized_depth_weight is None and 'prioritized_depth_weight' in control_params:
                self.prioritized_depth_weight = control_params['prioritized_depth_weight']

            if self.max_authors is None and 'max_authors' in control_params:
                self.max_authors = control_params['max_authors']

            if self.entity_types_in_network is None and 'entity_types_in_network' in control_params:
                self.entity_types_in_network = control_params['entity_types_in_network']

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
            self._save_expansion_metadata()
            self._initialize_network_with_seed_author()

        # Start the expansion loop
        self._expand_network()

    def _initialize_network_with_seed_author(self):
        """Initialize the network with the seed author (depth 0)"""
        self.logger.info(f"Initializing network with seed author: {self.person_name}")

        # First, use the search APIs to find publications for the seed author
        # The _search_for_author method will use the data_processor to process the results
        results = self._search_for_author(author_name=self.person_name)

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

    def _search_for_author(self, author_name: Optional[str] = None, author_id: Optional[str] = None) -> bool:
        """
        Search for publications by an author or author ID, process the results, and save them to the database.

        Args:
            author_name: Name of the author to search for
            author_id: ID of the author to search for. If this is provided, it will be used for CiNii search instead of author_name.

        Returns:
            Boolean indicating if the search was successful
        """
        self.logger.info(f"Searching for publications by author: {author_name} ({author_id})")

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

            # Using the NDLSearchAPI
            try:
                # Perform the search
                search_result = self.ndl_api.search_by_creator(author_name, **kwargs)

                if search_result:
                    # Export to JSON and get the file path
                    json_file_path = self.ndl_api.export_results_to_json(search_result)

                    if json_file_path and isinstance(json_file_path, str):
                        self.data_processor.process_json_file(json_file_path)
                        self.logger.info(f"Processed NDL search results for author: {author_name}")
                        search_successful = True
            except Exception as e:
                self.logger.error(f"Error in NDL search: {str(e)}")

        # Handle CiNii Search if enabled
        if self.cinii_api:
            # Set up search parameters
            kwargs = {}
            if author_id and (len(author_id) == 10):
                kwargs['researcherId'] = author_id
            elif author_name:
                kwargs['creator'] = author_name
            if self.title_keywords:
                kwargs['title'] = self.title_keywords
            if self.start_year:
                kwargs['from'] = str(self.start_year)
            if self.end_year:
                kwargs['until'] = str(self.end_year)

            try:
                # Perform the search and export in a single operation
                search_result_dirs = self.cinii_api.search(
                    search_types=self.publication_type,
                    **kwargs
                )

                if search_result_dirs is not None and len(search_result_dirs) > 0:
                    for search_result_dir in search_result_dirs:
                        # Export to JSON
                        export_success = self.cinii_api.export_results_to_json(search_result_dir)

                        if export_success:
                            # Get the full path to the exported JSON file
                            json_file_path = os.path.join(search_result_dir, "cinii_records.json")

                            # Process the JSON file with the data processor
                            self.data_processor.process_json_file(json_file_path)
                            self.logger.info(f"Processed CiNii search results for {author_name} ({author_id})'s {self.publication_type} in {search_result_dir}.")
                            search_successful = True
                else:
                    self.logger.error(f"No CiNii search results returned for {author_name}'s {self.publication_type}.")

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

            # Search for author's publications and save them to the database.
            self.logger.info(f"Processing author: {author_name} (depth: {depth}, priority: {priority:.4f})")
            search_successful = self._search_for_author(author_name=author_name, author_id=author_id)

            if search_successful:
                # Get coauthors
                coauthors = self._get_coauthors(author_id)
                self.logger.info(f"Found {len(coauthors)} coauthors for {author_name}")

                # Save newly discovered coauthors to the queue (network expansion status table).
                new_coauthors_depth = depth + 1
                self._save_coauthors_to_queue(coauthors, new_coauthors_depth, author_id)

                # Read all coauthors' depth from the network expansion status table
                coauthor_ids = [coauthor_id for coauthor_id, _, _, _ in coauthors]

                if coauthor_ids:
                    # Create the correct number of placeholders for the IN clause
                    placeholders = ','.join(['?' for _ in coauthor_ids])
                    query = f'SELECT author_id, depth FROM network_expansion_status WHERE author_id IN ({placeholders})'

                    self.cursor.execute(query, coauthor_ids)
                    coauthor_depths_map = {row[0]: row[1] for row in self.cursor.fetchall()}

                    # Add depths to coauthors list for priority calculation
                    coauthors = [
                        (coauthor_id, coauthor_name, coauthor_type, pub_count,
                         coauthor_depths_map.get(coauthor_id, new_coauthors_depth))
                        for coauthor_id, coauthor_name, coauthor_type, pub_count in coauthors
                    ]
                else:
                    coauthors = []

                # Here the user or AI can designate prioritized authors directly, change the prioritized depths, or change the depth values of specific authors.

                # get the current prioritized zone
                prioritized_authors = self._get_prioritized_authors()

                # Process coauthors: calculate the priority of each coauthor, and recalculate the priority of those affected by this iteration.
                self._process_coauthors(coauthors, prioritized_authors)

                # Update author status to 'completed'
                self._update_author_status(author_id, 'completed')

                # Increment counter
                self.authors_searched += 1
            else:
                # If search failed, mark the author as 'failed'
                self._update_author_status(author_id, 'failed')
                self.logger.warning(f"Failed to retrieve publications for author: {author_name}")

            # Update iteration count
            iteration_count += 1
            if iteration_count % 100 == 0:
                self.logger.info(f"Processed {iteration_count} authors so far.")

            # Small delay to avoid overwhelming the APIs
            time.sleep(1)

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
        if max_depth_authors > 0:
            # Check if we have any uncompleted authors at the maximum depth
            uncompleted_count = self._count_uncompleted_authors_at_depth(self.depth)
            if uncompleted_count == 0:
                self.logger.info(f"All authors up to maximum depth ({self.depth}) have been processed.")
                return True

        return False

    def _get_highest_priority_author(self) -> Optional[Tuple[str, str, int, float]]:
        """Get the in-progress or the highest priority pending author for processing. The in-progress author is processed first."""
        # First try to find an in-progress author (higher priority than pending authors) that is within the maximum depth
        self.cursor.execute(
            'SELECT author_id, author_name_cleaned, depth, priority FROM network_expansion_status '
            'WHERE status = "in_progress" AND depth <= ? ORDER BY priority DESC LIMIT 1',
            (self.depth,)
        )
        result = self.cursor.fetchone()

        if result:
            return result

        # If no in-progress author, find the highest priority pending author that is within the maximum depth
        self.cursor.execute(
            'SELECT author_id, author_name_cleaned, depth, priority FROM network_expansion_status '
            'WHERE status = "pending" AND depth <= ? ORDER BY priority DESC LIMIT 1',
            (self.depth,)
        )
        result = self.cursor.fetchone()

        return result

    def _update_author_status(self, author_id: str, status: str):
        """Update an author's status in the database"""
        self.cursor.execute(
            'UPDATE network_expansion_status SET status = ?, status_timestamp = ? WHERE author_id = ?',
            (status, datetime.now().isoformat(), author_id)
        )

        self.conn.commit()

    def _get_coauthors(self, author_id: str) -> List[Tuple[str, str, str,int]]:
        """
        Get an author's coauthors from the database.
        Exclude those with less than self.min_publications publications and not in the allowed entity types.

        Returns:
            List of tuples containing (coauthor_id, coauthor_cleaned_name, coauthor_type, publication_count)
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
                'SELECT e.id, e.cleaned_name, e.entity_type FROM entities e '
                'JOIN item_entities ie ON e.id = ie.entity_id '
                'WHERE ie.item_id = ? AND ie.relationship_type = "author" AND e.id != ?',
                (item_id, author_id)
            )

            for coauthor_id, coauthor_cleaned_name, coauthor_type in self.cursor.fetchall():
                if coauthor_id in coauthors:
                    coauthors[coauthor_id]['count'] += 1
                else:
                    coauthors[coauthor_id] = {
                        'cleaned_name': coauthor_cleaned_name,
                        'type': coauthor_type,
                        'count': 1
                    }

        # Remove the parent author of the current author from the coauthors list.
        parent_author_id = self.cursor.execute(
            'SELECT parent_author FROM network_expansion_status WHERE author_id = ?',
            (author_id,)
        ).fetchone()[0]
        if parent_author_id in coauthors:
            del coauthors[parent_author_id]

        # Remove coauthors that are not of the allowed entity types
        coauthors = {k: v for k, v in coauthors.items() if v['type'] in self.entity_types_in_network}

        # Filter coauthors by minimum publication count
        filtered_coauthors = []
        for coauthor_id, data in coauthors.items():
            if data['count'] >= self.min_publications:
                filtered_coauthors.append((coauthor_id, data['cleaned_name'], data['type'], data['count']))

        return filtered_coauthors

    def _save_coauthors_to_queue(self, coauthors: List[Tuple[str, str, str, int]], new_coauthors_depth: int, parent_author_id: str):
        """Save coauthors to the queue (network expansion status table)
            If they are not previously in the queue, add them but leave the priority blank.
            The parent_id and new_coauthors_depth fields are the author_id and depth + 1 of their parent author.
        Args:
            coauthors: List of coauthor data tuples (id, name, type, publication count)
            new_coauthors_depth: Depth of the new coauthors
            parent_author_id: ID of the parent author
        """
        for coauthor_id, coauthor_name, _, _ in coauthors:
            # Check if the coauthor is already in the network expansion status table
            self.cursor.execute(
                'SELECT * FROM network_expansion_status WHERE author_id = ?',
                (coauthor_id,)
            )
            if self.cursor.fetchone() is None:
                self.cursor.execute(
                    'INSERT INTO network_expansion_status (author_id, author_name_cleaned, depth, priority, status, status_timestamp, parent_author) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?)',
                (coauthor_id, coauthor_name, new_coauthors_depth, None, 'pending', datetime.now().isoformat(), parent_author_id)
                )


    def _process_coauthors(self, coauthors: List[Tuple[str, str, str, int, int]], prioritized_authors: List[str]):
        """
        Process all coauthors (with any status) of an author, including newly discovered ones and existing ones:
        1) Calculate and update the priority of each coauthor based on
            their depth and how many first and second degree neighbors
            they have in the prioritized_authors list. A coauthor's first degree neighbors
            contribute more to its priority than its second degree neighbors.
            If a coauthor is itself in the prioritized_authors list, its closeness_priority is set to 1.0.
            Normalization of closeness_priority is relative to the size of the prioritized set.

        Args:
            coauthors: List of coauthor data tuples (id, name, type, publication count, depth)
            prioritized_authors: List of author_ids that are prioritized.
        """

        self.logger.info(f"Processing priorities for {len(coauthors)} coauthors.")

        # Convert prioritized_authors list to a set for efficient lookup
        prioritized_set = set(prioritized_authors)
        prioritized_set_size = len(prioritized_set) # Get the size for normalization

        # Helper function to get neighbors (co-authors) of a given author
        def get_neighbors(author_id: str) -> Set[str]:
            neighbors = set()
            # Find publications by the author
            self.cursor.execute(
                'SELECT item_id FROM item_entities WHERE entity_id = ? AND relationship_type = "author"',
                (author_id,)
            )
            publication_ids = [row[0] for row in self.cursor.fetchall()]

            if not publication_ids:
                return neighbors

            # Find coauthors for these publications
            # Use parameter substitution correctly for IN clause
            placeholders = ','.join(['?'] * len(publication_ids))
            query = f'SELECT DISTINCT entity_id FROM item_entities WHERE item_id IN ({placeholders}) AND relationship_type = "author" AND entity_id != ?'
            params = publication_ids + [author_id]
            self.cursor.execute(query, params)

            neighbors.update(row[0] for row in self.cursor.fetchall())
            return neighbors

        # Cache for neighbors to avoid redundant lookups
        neighbor_cache = {}

        # For each coauthor, (re)calculate the priority and save it to the database.
        for coauthor_id, coauthor_name, _, _, coauthor_depth in coauthors:
            self.logger.debug(f"Calculating priority for coauthor: {coauthor_name} (ID: {coauthor_id})")

            try:
                # Calculate depth priority (based purely on depth)
                # Lower depth means higher priority
                depth_priority = 0.9 ** coauthor_depth

                # --- Calculate closeness priority ---
                closeness_priority = 0.0 # Default value

                # Check if the coauthor itself is prioritized
                if coauthor_id in prioritized_set:
                    closeness_priority = 1.0
                    self.logger.debug(f"  Coauthor {coauthor_name} is directly prioritized. Closeness Priority set to 1.0.")
                # Only calculate based on neighbors if the coauthor is NOT directly prioritized AND there are prioritized authors
                elif prioritized_set_size > 0: # Check size here to avoid division by zero later
                    # Get 1st degree neighbors (direct co-authors of this coauthor)
                    if coauthor_id not in neighbor_cache:
                        neighbor_cache[coauthor_id] = get_neighbors(coauthor_id)
                    neighbors_1st = neighbor_cache[coauthor_id]

                    # Get 2nd degree neighbors (co-authors of co-authors)
                    neighbors_2nd = set()
                    for neighbor1_id in neighbors_1st:
                        if neighbor1_id not in neighbor_cache:
                            neighbor_cache[neighbor1_id] = get_neighbors(neighbor1_id)
                        # Add neighbors of neighbors, excluding the original coauthor and 1st degree neighbors
                        neighbors_2nd.update(neighbor_cache[neighbor1_id] - {coauthor_id} - neighbors_1st)

                    # Calculate weighted overlap with prioritized set
                    overlap_1st = len(neighbors_1st.intersection(prioritized_set))
                    overlap_2nd = len(neighbors_2nd.intersection(prioritized_set))

                    # --- Closeness Priority Calculation (based on neighbors) ---
                    weight_1st = 1.0
                    weight_2nd = 0.5 # Adjust this weight as needed

                    # Calculate raw score
                    raw_closeness_score = (weight_1st * overlap_1st) + (weight_2nd * overlap_2nd)

                    # --- Normalization relative to prioritized set size ---
                    # The maximum possible raw score if all prioritized authors were 1st degree neighbors
                    # (weighted highest) would be prioritized_set_size * weight_1st.
                    # Use this as the denominator for normalization.
                    max_possible_score = prioritized_set_size * weight_1st

                    if raw_closeness_score > 0 and max_possible_score > 0:
                       # Scale score relative to max possible score based on prioritized set size
                       closeness_priority = raw_closeness_score / max_possible_score
                       # Ensure it doesn't exceed 1.0 (might happen if weights change or logic evolves)
                       closeness_priority = min(closeness_priority, 1.0)
                    # else: closeness_priority remains 0.0

                    self.logger.debug(f"Coauthor: {coauthor_name}, 1st Overlap: {overlap_1st}, 2nd Overlap: {overlap_2nd}, Raw Score: {raw_closeness_score}, Max Possible: {max_possible_score}, Closeness Priority (neighbors): {closeness_priority:.4f}")


                # Calculate combined priority using the weight
                # Apply weight only if prioritization is active (prioritized_depth is set AND there are prioritized authors)
                if self.prioritized_depth and self.prioritized_depth != ["off"] and prioritized_set_size > 0:
                    combined_priority = (1 - self.prioritized_depth_weight) * depth_priority + self.prioritized_depth_weight * closeness_priority
                else:
                    combined_priority = depth_priority # Default to depth priority if no prioritization active

                # Ensure priority is within [0, 1] range
                combined_priority = max(0.0, min(1.0, combined_priority))

                self.logger.debug(f"  Coauthor: {coauthor_name}, Depth: {coauthor_depth}, DepthP: {depth_priority:.4f}, ClosenessP: {closeness_priority:.4f}, CombinedP: {combined_priority:.4f}")

                # Update the coauthor's priority and timestamp in the network expansion status table
                self.cursor.execute(
                    'UPDATE network_expansion_status SET priority = ?, status_timestamp = ? WHERE author_id = ?',
                    (combined_priority, datetime.now().isoformat(), coauthor_id)
                )

            except Exception as e:
                self.logger.error(f"Error processing priority for coauthor {coauthor_name} (ID: {coauthor_id}): {str(e)}")

        # Commit all the priority updates at once
        self.conn.commit()
        self.logger.info("Finished updating coauthor priorities.")



    def _get_prioritized_authors(self) -> List[str]:
        """Interpret the prioritized_depth parameter into a list of author_ids and combine it with the optional list of specific_ids.

        Args:
            specific_ids: Optional list of specific author_ids to combine with the prioritized authors.

        Returns:
            List of author_ids that are prioritized based on the prioritized_depth parameter and the optional specific_ids list.
        """
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

        depths = sorted(list(set(depths)))

        # Get author_ids at these depths from the database
        placeholders = ','.join(['?' for _ in range(len(depths))])
        self.cursor.execute(
            f'SELECT author_id FROM network_expansion_status WHERE depth IN ({placeholders})',
            depths
        )
        prioritized_authors = [row[0] for row in self.cursor.fetchall()]

        # combine the prioritized authors in the prioritized depths with the specific_prioritized_authors and remove duplication
        if self.specific_prioritized_authors:
            prioritized_authors.extend(self.specific_prioritized_authors)

        return list(set(prioritized_authors))

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

    def _count_uncompleted_authors_at_depth(self, depth: int) -> int:
        """Count the number of pending authors at a specific depth"""
        self.cursor.execute(
            'SELECT COUNT(*) FROM network_expansion_status WHERE status IN ("pending", "in_progress") AND depth = ?',
            (depth,)
        )
        result = self.cursor.fetchone()
        return result[0] if result else 0



def main():
    """Main function to run the Biblio Agent"""
    parser = argparse.ArgumentParser(description="Biblio Explorer Agent - Coauthorship Network Expansion")

    # Required parameters
    parser.add_argument("--database", choices=["ndl", "cinii", "both"], default="both",
                        help="Database to search (NDL Search, CiNii, or both)")

    # Optional parameters with defaults
    parser.add_argument("--db_path", type=str, help="Path to an existing database file")
    parser.add_argument("--person", type=str, help="Initial author name to start expansion from")
    parser.add_argument("--author_from_rdf", action="store_true",
                        help="Extract author id from RDF files when using CiNii database")
    parser.add_argument("--title", type=str, help="Keywords to filter publications by title")
    parser.add_argument("--start_year", type=int, help="Start year for publication filtering")
    parser.add_argument("--end_year", type=int, help="End year for publication filtering")
    parser.add_argument("--ndc", type=str, help="NDC classification number")
    parser.add_argument(
        "--type",
        nargs='+',
        choices=["all", "books", "articles", "data", "dissertations", "projects"],
        default=["books"],
        help="Type(s) of publications to search for (e.g., --type books articles, or --type all)"
    )
    parser.add_argument("--min_pubs", type=int, default=1,
                        help="Minimum number of publications for including authors")
    parser.add_argument("--depth", type=int, default=3,
                        help="Maximum depth for network expansion")
    parser.add_argument("--priority_depth", nargs="+", default=None,
                        help="Author depths to consider for calculating priority (e.g., '0', '1,2', '-2', or 'all')")
    parser.add_argument("--priority_weight", type=float, default=0.8,
                        help="Weight for depth-based prioritization (0-1)")
    parser.add_argument("--max_authors", type=int, default=100,
                        help="Maximum number of authors to search")
    parser.add_argument("--entity_types", nargs="+", default=["person"],
                        help="Entity types to include in the network (e.g., 'person', 'organization')")

    args = parser.parse_args()

    # Create and run the agent
    agent = BiblioAgent(
        db_path=args.db_path,
        database_selection=args.database,
        person_name=args.person,
        author_from_rdf=args.author_from_rdf,
        title_keywords=args.title,
        start_year=args.start_year,
        end_year=args.end_year,
        ndc_number=args.ndc,
        publication_type=args.type,
        min_publications=args.min_pubs,
        depth=args.depth,
        prioritized_depth=args.priority_depth,
        prioritized_depth_weight=args.priority_weight,
        max_authors=args.max_authors,
        entity_types_in_network=args.entity_types
    )

    try:
        agent.start_network_expansion()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting...")
    finally:
        agent.close()
        print("Biblio Agent finished.")


if __name__ == "__main__":
    main()

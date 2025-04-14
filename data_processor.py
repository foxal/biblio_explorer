import xml.etree.ElementTree as ET
import sqlite3
import os
from pathlib import Path
import re
from typing import List, Dict, Optional, Tuple, Set
import logging
from collections import defaultdict
import uuid
from get_item_type import get_item_type
from gpt import GPTResponse
from difflib import SequenceMatcher
from inputimeout import inputimeout, TimeoutOccurred
from datetime import datetime
import argparse  # Added for CLI argument parsing

# Set up logging - default to INFO but allow override via environment variable
log_level_name = os.environ.get('NDL_LOG_LEVEL', 'INFO')
log_level = getattr(logging, log_level_name.upper(), logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set HTTP-related loggers to be less verbose
http_log_level = getattr(logging, os.environ.get('NDL_HTTP_LOG_LEVEL', 'WARNING').upper(), logging.WARNING)
logging.getLogger('httpcore').setLevel(http_log_level)
logging.getLogger('httpx').setLevel(http_log_level)
logging.getLogger('openai').setLevel(http_log_level)

# Namespace mapping for XML parsing
NAMESPACES = {
    'srw': 'http://www.loc.gov/zing/srw/',
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'dcterms': 'http://purl.org/dc/terms/',
    'dcndl': 'http://ndl.go.jp/dcndl/terms/',
    'foaf': 'http://xmlns.com/foaf/0.1/'
}

# Add Japanese era conversion constants
ERA_CONVERSION = {
    "明治": 1868,
    "大正": 1912,
    "昭和": 1926,
    "平成": 1989,
    "令和": 2019
}

class BiblioDataProcessor:
    def __init__(self, db_path: str = 'ndl_data.db', use_method2: bool = False, auto_type_check: int = 0, 
                 dedup_threshold: float = 0.7, manual_dedup: bool = False, year_diff_threshold: int = 0,
                 gpt_credibility_threshold: float = 0.8, gui_mode: bool = False, log_level: str = 'INFO',
                 http_log_level: str = 'WARNING', json_mode: bool = True):
        """Initialize the processor with database path and type checking options."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.skipped_records = defaultdict(int)
        self.use_method2 = use_method2
        self.auto_type_check = auto_type_check
        self.dedup_threshold = dedup_threshold
        self.manual_dedup = manual_dedup
        self.year_diff_threshold = year_diff_threshold
        self.gpt_credibility_threshold = gpt_credibility_threshold
        self.gpt = GPTResponse() if not manual_dedup else None
        self.gpt_cache = {}  # Cache for GPT responses
        self.logger = logging.getLogger(__name__)
        self.deduplication_results = []  # Store deduplication results
        self.gui_mode = gui_mode
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)  # Convert string to logging level
        self.http_log_level = getattr(logging, http_log_level.upper(), logging.WARNING)  # Convert string to logging level
        # JSON mode is now the only mode, but we keep the parameter for backward compatibility
        self.json_mode = True
        self.setup_logging()
        self.setup_database()

    def setup_logging(self):
        """Set up logging to both console and file."""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Get the logger and disable propagation to root logger
        self.logger = logging.getLogger('ndl_processor')
        # Only disable propagation when not in GUI mode
        # In GUI mode we need propagation so the GUI can capture all messages
        if not self.gui_mode:
            self.logger.propagate = False  # Prevent messages from propagating to root logger
        
        # Remove any existing handlers
        self.logger.handlers = []
        
        # Create a file handler
        log_file = f'logs/ndl_processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.log_level)
        
        # Create a formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the file handler to the logger
        self.logger.addHandler(file_handler)
        
        # Add console handler only if not in GUI mode
        # if not self.gui_mode:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
            
        self.logger.setLevel(self.log_level)
        
        # Configure HTTP-related loggers using the specified HTTP log level
        logging.getLogger('httpcore').setLevel(self.http_log_level)
        logging.getLogger('httpx').setLevel(self.http_log_level)
        logging.getLogger('openai').setLevel(self.http_log_level)
        
        self.logger.info(f"Logging set to {logging.getLevelName(self.log_level)} level")
        self.logger.debug(f"HTTP logging set to {logging.getLevelName(self.http_log_level)} level")

    def setup_database(self):
        """Set up SQLite database with required tables."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self.logger.info(f"Connected to database at {self.db_path}")

            # Create bibliographic items table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS bibliographic_items (
                id TEXT PRIMARY KEY,
                item_type TEXT,
                title TEXT,
                publication_date TEXT,
                subject TEXT,
                publication_title TEXT
            )
            ''')

            # Create entities table (combined authors and publishers)
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT,
                cleaned_name TEXT,
                entity_type TEXT,
                is_author BOOLEAN,
                is_publisher BOOLEAN,
                type_detection_method INTEGER  -- 0: not checked, 1: manual, 2: auto without confirmation, 3: method 1
            )
            ''')

            # Create item-entity relationship table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS item_entities (
                item_id TEXT,
                entity_id TEXT,
                relationship_type TEXT,
                FOREIGN KEY (item_id) REFERENCES bibliographic_items(id),
                FOREIGN KEY (entity_id) REFERENCES entities(id),
                PRIMARY KEY (item_id, entity_id, relationship_type)
            )
            ''')

            self.conn.commit()
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error setting up database: {e}")
            raise

    def extract_id(self, about_attr: str) -> str:
        """Extract ID from rdf:about attribute."""
        return about_attr.split('/')[-1].split('#')[0]

    def get_text_content(self, element, xpath: str) -> Optional[str]:
        """Safely get text content from an element using xpath."""
        result = element.find(xpath, NAMESPACES)
        if result is not None:
            # Handle nested rdf:value if present
            value = result.find('.//rdf:value', NAMESPACES)
            if value is not None:
                return value.text
            return result.text
        return None

    def clean_japanese_name(self, name: str) -> str:
        """Clean Japanese name by removing commas, spaces, years, and special characters."""
        if not name:
            return ""
        
        # Check if this is a publisher with location (e.g., "東京 : 岩波書店")
        location_pattern = r'^(.+)\s*[:：]\s*(.+)$'
        location_match = re.search(location_pattern, name)
        if location_match:
            # For publishers, use only the part after the colon
            name = location_match.group(2).strip()
        
        # Remove birth/death years (e.g. "山田太郎 (1900-1980)" -> "山田太郎")
        name = re.sub(r'\([^)]*\)', '', name)
        
        # Remove special role indicators with prefix (e.g. "／著", "／編", "／共訳")
        name = re.sub(r'[／/][著編共訳著者編集監修翻訳訳者]+', '', name)
        
        # Remove special role indicators without prefix at the end of the name
        name = re.sub(r'(?:共著|共訳|著者|著作|編者|編集|監修|翻訳|訳者|共編|監訳|校訂|校注|編著|著|編|訳)$', '', name)
        
        # Remove special role indicators with spaces (e.g. "山田太郎 著", "山田太郎 共著")
        name = re.sub(r'\s+(?:共著|共訳|著者|著作|編者|編集|監修|翻訳|訳者|共編|監訳|校訂|校注|編著|著|編|訳)(?:\s+|$)', '', name)
        
        # Remove any spaces after removing role indicators
        name = name.strip()
        
        # Check if the name contains any Japanese characters
        has_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', name))
        
        if has_japanese:
            # For names with Japanese characters, remove commas and spaces
            name = name.replace(',', '').replace(' ', '')
            
            # Keep Japanese characters (Hiragana, Katakana, Kanji), full-width letters, and regular letters
            # This regex keeps:
            # - Hiragana: \u3040-\u309F
            # - Katakana: \u30A0-\u30FF
            # - Kanji: \u4E00-\u9FFF
            # - Full-width letters: \uFF21-\uFF3A (A-Z), \uFF41-\uFF5A (a-z)
            # - Regular letters: a-z, A-Z
            name = re.sub(r'[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF21-\uFF3A\uFF41-\uFF5Aa-zA-Z]', '', name)
        else:
            # For English names, just remove role indicators and trim
            name = name.strip()
        
        return name

    def get_item_type(self, item_name: str, auto_type_check: int = 1, gui_mode: bool = False, gui_callback = None) -> Tuple[str, bool]:
        """Determine the type of an item (person, organization, or something else) based on its name.
        
        Args:
            item_name: The name of the item to classify
            auto_type_check: 0 for manual, 1 for with confirmation, 2 for without confirmation
            gui_mode: Whether running in GUI mode
            gui_callback: Callback function for GUI input
            
        Returns:
            Tuple of (item_type, is_auto) where:
            - item_type: The determined type of the item
            - is_auto: Whether the type was determined automatically
        """
        def manual_type_check(manual_item_name):
            if gui_mode and gui_callback:
                # Use GUI callback to get input if in GUI mode
                return gui_callback(manual_item_name)
            else:
                # Use console input if not in GUI mode
                item_is_type = input(f'You decide: is "{manual_item_name}" a person, organization, publication or something else? [P/o/b/e]: ') or "p"
                if item_is_type.lower() == "p":
                    return "person"
                elif item_is_type.lower() == "o":
                    return "organization"
                elif item_is_type.lower() == "b":
                    return "publication"
                else:
                    item_type = input(f'Please specify the type for "{manual_item_name}": ')
                    return item_type

        suggested_type = None
        if auto_type_check == 1 or auto_type_check == 2:
            gpt_type_judge = GPTResponse()
            prompt_text = f"""Is '{item_name}' a person, an organization, or a publication? If person, return "person", if organization return "organization", if publication, return "publication". No explanation."""
            suggested_type_response = gpt_type_judge.get_response(prompt_text).lower()
            if "person" in suggested_type_response:
                suggested_type = "person"
            elif "organization" in suggested_type_response:
                suggested_type = "organization"
            elif "publication" in suggested_type_response:
                suggested_type = "publication"
            else:
                suggested_type = "unclassified"

            if auto_type_check == 1: # with human confirmation
                is_auto = False
                
                # Handle confirmation with GUI or console
                if gui_mode and gui_callback:
                    confirmation = gui_callback(f'GPT suggests "{item_name}" is a(n) {suggested_type}. Do you agree?')
                else:
                    try:
                        confirmation = inputimeout(prompt=f'GPT suggests "{item_name}" is a(n) {suggested_type}. Do you agree? [Y/n/m]: ', timeout=20) or "y"
                    except TimeoutOccurred:
                        self.logger.info("Time is out. Defaulting to GPT suggestion.")
                        confirmation = "y"
                        is_auto = True
                        
                # Process confirmation result
                if confirmation.lower() == 'y':
                    item_type = suggested_type
                elif confirmation.lower() == 'n' and suggested_type in ["person", "organization"]:
                    item_type = "organization" if suggested_type == "person" else "person"
                elif confirmation.lower() == 'm':
                    item_type = manual_type_check(item_name)
                else:
                    item_type = "unclassified"
            elif auto_type_check == 2: # without human confirmation
                is_auto = True
                if suggested_type in ["person", "organization", "publication"]:
                    item_type = suggested_type
                else:
                    item_type = "unclassified"
        else:
            is_auto = False
            item_type = manual_type_check(item_name)

        return item_type, is_auto

    def determine_entity_type_method2(self, name: str, existing_method: int = 0) -> Tuple[str, int]:
        """Determine entity type using method 2 (get_item_type) if needed."""
        # If entity was already checked with a higher or equal method, return None to keep existing type
        if existing_method >= self.auto_type_check:
            return None, existing_method
        
        # Use get_item_type to determine type
        entity_type, is_auto = self.get_item_type(
            name, 
            self.auto_type_check,
            self.gui_mode,
            self.entity_type_callback if hasattr(self, 'entity_type_callback') else None
        )
        mode = self.auto_type_check  # Use the current auto_type_check value
        
        # If get_item_type returns 'publication', treat it as 'organization'
        if entity_type == 'publication':
            entity_type = 'organization'
        
        return entity_type, mode

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using SequenceMatcher."""
        if not str1 or not str2:
            return 0.0
        return SequenceMatcher(None, str1, str2).ratio()

    def get_entity_set(self, item_id: str) -> Set[str]:
        """Get set of entity IDs for an item."""
        self.cursor.execute('''
        SELECT entity_id FROM item_entities WHERE item_id = ?
        ''', (item_id,))
        return {row[0] for row in self.cursor.fetchall()}

    def get_cache_key(self, item1: Dict, item2: Dict) -> str:
        """Generate a cache key for GPT responses."""
        items = sorted([
            f"{item1['title']}|{item1['publication_date']}",
            f"{item2['title']}|{item2['publication_date']}"
        ])
        return '||'.join(items)

    def convert_fullwidth_to_halfwidth(self, text: str) -> str:
        """Convert full-width numbers to half-width numbers."""
        if not text:
            return ""
        # Convert full-width numbers (U+FF10 to U+FF19) to half-width numbers (0-9)
        converted = ""
        for char in text:
            if '\uff10' <= char <= '\uff19':  # Full-width numbers range
                converted += chr(ord(char) - ord('０') + ord('0'))  # Convert to half-width
            else:
                converted += char
        return converted

    def normalize_date(self, date_str: str) -> Tuple[str, Optional[str]]:
        """Normalize date string to a consistent format and extract year.
        
        Converts formats like:
        - (通号 502) 1973.02.01 → 1973-02-01
        - 15(10) 1973.10.00 → 1973-10
        - 明治45年3月 → 1912-03
        - 昭和42年5月 → 1967-05
        - 19590315 → 1959-03-15
        - 196004 → 1960-04
        
        Returns:
            A tuple of (normalized_date, extracted_year)
            - normalized_date: The normalized date string
            - extracted_year: Just the year component, or None if extraction fails
        """
        if not date_str:
            return "", None
            
        # Convert full-width numbers
        date_str = self.convert_fullwidth_to_halfwidth(date_str)
        
        # Initialize extracted_year as None
        extracted_year = None
        
        # Check for pure numeric formats first (YYYYMMDD or YYYYMM)
        numeric_only = re.sub(r'\D', '', date_str)  # Remove all non-digits
        
        # Handle YYYYMMDD format (8 digits)
        if len(numeric_only) == 8:
            year = numeric_only[:4]
            month = numeric_only[4:6]
            day = numeric_only[6:8]
            
            # Validate the values
            if (1800 <= int(year) <= 2100 and 
                1 <= int(month) <= 12 and 
                1 <= int(day) <= 31):
                extracted_year = year
                return f"{year}-{month}-{day}", extracted_year
        
        # Handle YYYYMM format (6 digits)
        elif len(numeric_only) == 6:
            year = numeric_only[:4]
            month = numeric_only[4:6]
            
            # Validate the values
            if (1800 <= int(year) <= 2100 and 
                1 <= int(month) <= 12):
                extracted_year = year
                return f"{year}-{month}", extracted_year
        
        # Check for Japanese era years
        jp_era_pattern = r'(明治|大正|昭和|平成|令和)(\d{1,2})(?:年)?(?:(\d{1,2})月)?(?:(\d{1,2})日)?'
        jp_match = re.search(jp_era_pattern, date_str)
        
        if jp_match:
            era, year, month, day = jp_match.groups()
            if era in ERA_CONVERSION:
                # Convert era year to Gregorian year
                greg_year = ERA_CONVERSION[era] + int(year) - 1
                extracted_year = str(greg_year)
                
                # Format as YYYY-MM-DD
                if month and day:
                    return f"{greg_year}-{int(month):02d}-{int(day):02d}", extracted_year
                elif month:
                    return f"{greg_year}-{int(month):02d}", extracted_year
                else:
                    return f"{greg_year}", extracted_year
        
        # Extract the date part (handles various formats)
        date_patterns = [
            r'(?:\D|^)(\d{4})\.(\d{1,2})(?:\.(\d{1,2}))?(?:\D|$)',  # YYYY.MM.DD or YYYY.MM
            r'(?:\D|^)(\d{4})-(\d{1,2})(?:-(\d{1,2}))?(?:\D|$)',    # YYYY-MM-DD or YYYY-MM
            r'(?:\D|^)(\d{4})/(\d{1,2})(?:/(\d{1,2}))?(?:\D|$)',    # YYYY/MM/DD or YYYY/MM
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                year = match.group(1)
                month = match.group(2)
                day = match.group(3)
                
                # Validate year
                if not (1800 <= int(year) <= 2100):
                    continue
                    
                extracted_year = year
                    
                # Format as YYYY-MM-DD
                if day and day != '00':
                    return f"{year}-{int(month):02d}-{int(day):02d}", extracted_year
                else:
                    return f"{year}-{int(month):02d}", extracted_year
        
        # If no date pattern matched, try to find just a year
        year_match = re.search(r'(?:\D|^)((?:18|19|20)\d{2})(?:\D|$)', date_str)
        if year_match:
            year = year_match.group(1)
            if 1800 <= int(year) <= 2100:
                extracted_year = year
                return year, extracted_year
                
        self.logger.warning(f"Could not normalize date string: {date_str}")
        return date_str, None

    def get_item_signature(self, item_id: str) -> str:
        """Get a combined signature string for an item including title, authors, and publisher."""
        try:
            # Get item details
            self.cursor.execute('''
            SELECT title 
            FROM bibliographic_items 
            WHERE id = ?
            ''', (item_id,))
            result = self.cursor.fetchone()
            if not result:
                self.logger.warning(f"No bibliographic item found for ID {item_id}")
                return ""
            
            title = result[0]
            if not title:
                self.logger.warning(f"No title found for item {item_id}")
                return ""
            
            # Get all authors with their cleaned names
            self.cursor.execute('''
            SELECT e.cleaned_name 
            FROM entities e 
            JOIN item_entities ie ON e.id = ie.entity_id 
            WHERE ie.item_id = ? AND ie.relationship_type = 'author'
            ORDER BY e.cleaned_name
            ''', (item_id,))
            authors = [row[0] for row in self.cursor.fetchall() if row[0]]  # Only include non-empty cleaned names
            
            # Get all publishers with their cleaned names
            self.cursor.execute('''
            SELECT e.cleaned_name 
            FROM entities e 
            JOIN item_entities ie ON e.id = ie.entity_id 
            WHERE ie.item_id = ? AND ie.relationship_type = 'publisher'
            ORDER BY e.cleaned_name
            ''', (item_id,))
            publishers = [row[0] for row in self.cursor.fetchall() if row[0]]  # Only include non-empty cleaned names
            
            # Combine all information into a single string (excluding year)
            signature = f"{title} | {' '.join(authors)} | {' '.join(publishers)}"
            return signature.strip()
        except Exception as e:
            self.logger.error(f"Error getting signature for item {item_id}: {str(e)}")
            return ""

    def clean_title(self, title: str) -> str:
        """Clean title by removing punctuation, spaces, and special characters."""
        if not title:
            return ""
        # Remove all punctuation, spaces, and special characters
        cleaned = re.sub(r'[^\w\u4e00-\u9fff]', '', title)
        return cleaned.lower()

    def is_title_included(self, title1: str, title2: str) -> bool:
        """Check if one title is included in another after cleaning."""
        if not title1 or not title2:
            return False
        cleaned1 = self.clean_title(title1)
        cleaned2 = self.clean_title(title2)
        return cleaned1 in cleaned2 or cleaned2 in cleaned1

    def get_user_input(self, prompt: str, timeout: int = 20) -> Optional[str]:
        """Get user input with timeout, but wait indefinitely if user starts typing.
        
        In both GUI and command-line mode, we now use a single-step approach for deduplication:
        - "1" to keep item 1
        - "2" to keep item 2
        - "n" for not duplicates
        """
        # If in GUI mode, this method should be overridden by a subclass
        if self.gui_mode:
            self.logger.warning("get_user_input called in GUI mode without proper override")
            return None
            
        # Check if this is a deduplication prompt
        if "Are these items duplicates?" in prompt:
            # Replace the two-step y/n prompt with single-step options
            prompt = prompt + '\n[1/2/n]:\n1 = Keep Item 1\n2 = Keep Item 2\nn = Not Duplicates\nYour choice: '
            
            try:
                # First try with timeout
                response = inputimeout(prompt=prompt, timeout=timeout)
                
                # Validate response
                if response.lower() in ['1', '2', 'n']:
                    return response.lower()
                else:
                    self.logger.warning(f"Invalid deduplication choice: {response}, please choose 1, 2, or n")
                    # Let the user try again without timeout
                    response = input("Please choose 1, 2, or n: ")
                    return response.lower() if response.lower() in ['1', '2', 'n'] else 'n'
                    
            except TimeoutOccurred:
                # If timeout occurs, give one more chance with no timeout
                try:
                    self.logger.info("Timeout occurred. Please provide your input now (no timeout):")
                    response = input(prompt)
                    
                    # Validate response
                    if response.lower() in ['1', '2', 'n']:
                        return response.lower()
                    else:
                        self.logger.warning(f"Invalid deduplication choice: {response}, defaulting to 'n'")
                        return 'n'
                except KeyboardInterrupt:
                    self.logger.warning("User interrupted input, treating as non-duplicate")
                    return 'n'
            except KeyboardInterrupt:
                self.logger.warning("User interrupted input, treating as non-duplicate")
                return 'n'
        else:
            # For non-deduplication prompts, use the original implementation
            try:
                # First try with timeout
                response = inputimeout(prompt=prompt, timeout=timeout)
                return response
            except TimeoutOccurred:
                # If timeout occurs, give one more chance with no timeout
                try:
                    self.logger.info("Timeout occurred. Please provide your input now (no timeout):")
                    response = input(prompt)
                    return response
                except KeyboardInterrupt:
                    self.logger.warning("User interrupted input")
                    return None
            except KeyboardInterrupt:
                self.logger.warning("User interrupted input")
                return None

    def compare_items(self, item1: Dict, item2: Dict) -> Tuple[bool, Optional[str]]:
        """Compare two items using combined signature similarity.
        
        Uses a single-step approach for deduplication decisions:
        - "1" to keep item 1
        - "2" to keep item 2
        - "n" for not duplicates
        
        Return:
            Tuple of (is_duplicate, keep_item) where:
            - is_duplicate is a boolean indicating if the items are duplicates
            - keep_item is '1' to keep item1, '2' to keep item2, or None if not duplicates
        """
        # Get combined signatures for both items
        sig1 = self.get_item_signature(item1['id'])
        sig2 = self.get_item_signature(item2['id'])
        
        if not sig1 or not sig2:
            self.logger.debug(f"Skipping comparison due to missing signature for item1: '{item1['id']}' or item2: '{item2['id']}'")
            return False, None
        
        # First check if one title is included in another
        similarity = 2.0 if self.is_title_included(item1['title'], item2['title']) else self.calculate_similarity(sig1, sig2)
        
        if similarity >= self.dedup_threshold:
            self.logger.debug(f"Comparing items:\nItem 1 ({item1['id']}): {sig1}\nItem 2 ({item2['id']}): {sig2}\nSimilarity: {similarity:.2f}")
            
            if self.manual_dedup:
                try:
                    # Prepare prompt for user decision
                    prompt = f'Are these items duplicates?\nItem 1: {sig1}\nItem 2: {sig2}\n'
                    if similarity == 2.0:
                        prompt += "Note: One title is included in another.\n"
                    
                    # Ask for single-step decision (should return '1', '2', or 'n')
                    self.logger.info(f"Requesting deduplication decision")
                    response = self.get_user_input(prompt)
                    self.logger.info(f"Deduplication response: {response}")
                    
                    if response is None:
                        return False, None
                        
                    if response == '1' or response == '2':
                        self.logger.info(f"Items are duplicates, keeping item {response}")
                        return True, response
                    else:
                        self.logger.info("User indicated items are not duplicates")
                        return False, None
                        
                except Exception as e:
                    self.logger.warning(f"Error during manual deduplication: {str(e)}")
                    return False, None
            else:
                # Automatic deduplication using GPT
                # Check cache first
                cache_key = self.get_cache_key(item1, item2)
                if cache_key in self.gpt_cache:
                    self.logger.debug("Using cached GPT response")
                    return self.gpt_cache[cache_key]

                # Define schema for deduplication decision
                dedup_schema = {
                    "type": "object",
                    "properties": {
                        "is_duplicate": {
                            "type": "boolean",
                            "description": "Whether the two items are duplicates"
                        },
                        "keep_item": {
                            "type": "string",
                            "enum": ["1", "2", "0"],
                            "description": "Which item to keep if they are duplicates (1 or 2), or 0 if not duplicates"
                        },
                        "credibility": {
                            "type": "number",
                            "description": "Confidence level in the decision (0-1)"
                        }
                    },
                    "required": ["is_duplicate", "keep_item", "credibility"],
                    "additionalProperties": False
                }

                # Use GPT for final judgment with structured output
                prompt = f"""Compare these two bibliographic items and determine if they are duplicates:
Item 1: {sig1}
Item 2: {sig2}"""
                if similarity == 2.0:
                    prompt += "\nNote: One title is included in another."
                prompt += "\n\nThe item with more complete information should be kept."
                
                response = self.gpt.get_response(prompt, schema=dedup_schema)
                if not response:
                    self.logger.warning("GPT failed to provide a response, asking for manual input")
                    try:
                        prompt = f'GPT failed to respond. Are these items duplicates?\nItem 1: {sig1}\nItem 2: {sig2}\n'
                        if similarity == 2.0:
                            prompt += "Note: One title is included in another.\n"
                        
                        # Ask for single-step decision (should return '1', '2', or 'n')
                        self.logger.info("Requesting manual input due to GPT failure")
                        response = self.get_user_input(prompt)
                        self.logger.info(f"Manual input response: {response}")
                        
                        if response is None:
                            result = (False, None)
                        elif response == '1' or response == '2':
                            result = (True, response)
                        else:
                            result = (False, None)
                            
                    except Exception as e:
                        self.logger.warning(f"Error during manual deduplication: {str(e)}")
                        result = (False, None)
                else:
                    self.logger.info(f"GPT response: {response}")
                    credibility = response['credibility']
                    if credibility < self.gpt_credibility_threshold:
                        self.logger.info(f"GPT credibility ({credibility}) below threshold ({self.gpt_credibility_threshold}), asking for manual input")
                        try:
                            prompt = f'GPT is not confident (credibility: {credibility:.2f}). Are these items duplicates?\nItem 1: {sig1}\nItem 2: {sig2}\n'
                            if similarity == 2.0:
                                prompt += "Note: One title is included in another.\n"
                            
                            # Ask for single-step decision (should return '1', '2', or 'n')
                            self.logger.info("Requesting manual input due to low GPT credibility")
                            response = self.get_user_input(prompt)
                            self.logger.info(f"Manual input response: {response}")
                            
                            if response is None:
                                result = (False, None)
                            elif response == '1' or response == '2':
                                result = (True, response)
                            else:
                                result = (False, None)
                                
                        except Exception as e:
                            self.logger.warning(f"Error during manual deduplication: {str(e)}")
                            result = (False, None)
                    else:
                        is_duplicate = response['is_duplicate']
                        keep_item = response['keep_item'] if is_duplicate else None
                        if keep_item == '0':
                            keep_item = None
                        self.logger.info(f"GPT deduplication result - is_duplicate: {is_duplicate}, keep_item: {keep_item}")
                        result = (is_duplicate, keep_item)
                
                # Cache the result
                self.gpt_cache[cache_key] = result
                return result
        
        return False, None

    def merge_items(self, item1: Dict, item2: Dict, keep_item: str) -> None:
        """Merge two items, keeping the one with more complete information."""
        if keep_item == '1':
            keep_id, merge_id = item1['id'], item2['id']
            keep_item, merge_item = item1, item2
        else:
            keep_id, merge_id = item2['id'], item1['id']
            keep_item, merge_item = item2, item1
        
        self.logger.info(f"Merging items:\nKeep: {keep_item['title']} ({keep_id})\nMerge: {merge_item['title']} ({merge_id})")
        
        # Store deduplication result
        self.deduplication_results.append({
            'keep_id': keep_id,
            'merge_id': merge_id,
            'keep_title': keep_item['title'],
            'merge_title': merge_item['title'],
            'keep_date': keep_item['publication_date'],
            'merge_date': merge_item['publication_date'],
            'similarity': self.calculate_similarity(
                self.get_item_signature(keep_id),
                self.get_item_signature(merge_id)
            )
        })
        
        try:
            # First, get all relationships from the item to be merged
            self.logger.info(f"Getting relationships for item to be merged: {merge_id}")
            self.cursor.execute('''
            SELECT entity_id, relationship_type 
            FROM item_entities 
            WHERE item_id = ?
            ''', (merge_id,))
            relationships = self.cursor.fetchall()
            self.logger.info(f"Found {len(relationships)} relationships to transfer")
            
            # For each relationship, check if it already exists for the keep item
            transferred_count = 0
            for entity_id, relationship_type in relationships:
                self.cursor.execute('''
                SELECT 1 FROM item_entities 
                WHERE item_id = ? AND entity_id = ? AND relationship_type = ?
                ''', (keep_id, entity_id, relationship_type))
                
                if not self.cursor.fetchone():
                    # Only insert if the relationship doesn't already exist
                    self.logger.debug(f"Transferring relationship: entity={entity_id}, type={relationship_type}")
                    self.cursor.execute('''
                    INSERT INTO item_entities (item_id, entity_id, relationship_type)
                    VALUES (?, ?, ?)
                    ''', (keep_id, entity_id, relationship_type))
                    transferred_count += 1
                else:
                    self.logger.debug(f"Relationship already exists: entity={entity_id}, type={relationship_type}")
            
            self.logger.info(f"Transferred {transferred_count} relationships to item {keep_id}")
            
            # Delete the merged item's relationships
            self.logger.info(f"Deleting relationships for merged item: {merge_id}")
            self.cursor.execute('''
            DELETE FROM item_entities 
            WHERE item_id = ?
            ''', (merge_id,))
            
            # Update the kept item's publication date to the earlier one if possible
            keep_year = keep_item.get('extracted_year')
            merge_year = merge_item.get('extracted_year')
            
            if keep_year and merge_year:
                if int(merge_year) < int(keep_year):
                    self.logger.info(f"Updating publication date from {keep_year} to {merge_year}")
                    self.cursor.execute('''
                    UPDATE bibliographic_items 
                    SET publication_date = ?
                    WHERE id = ?
                    ''', (merge_item['publication_date'], keep_id))
            
            # Delete the merged item
            self.logger.info(f"Deleting merged item: {merge_id}")
            self.cursor.execute('''
            DELETE FROM bibliographic_items 
            WHERE id = ?
            ''', (merge_id,))
            
            self.conn.commit()
            self.logger.info(f"Successfully merged item {merge_id} into {keep_id}")
        except Exception as e:
            self.logger.error(f"Error merging items {merge_id} and {keep_id}: {str(e)}")
            # Log the traceback for more information
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.conn.rollback()

    def save_deduplication_report(self, output_file: str = 'deduplication_report.csv'):
        """Save deduplication results to a CSV file."""
        if not self.deduplication_results:
            self.logger.info("No deduplication results to report")
            return
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write header
                f.write('keep_id,merge_id,keep_title,merge_title,keep_date,merge_date,similarity\n')
                
                # Write results
                for result in self.deduplication_results:
                    f.write(f"{result['keep_id']},{result['merge_id']},{result['keep_title']},{result['merge_title']},"
                           f"{result['keep_date']},{result['merge_date']},{result['similarity']}\n")
            
            self.logger.info(f"Deduplication report saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving deduplication report: {str(e)}")

    def check_duplicates(self, new_item: Dict) -> bool:
        """Check if the new item is a duplicate of any existing item."""
        try:
            # Use the extracted year from normalize_date
            year = new_item['extracted_year']
            if not year:
                self.logger.debug(f"Could not extract year from date '{new_item['publication_date']}' for item {new_item['id']}, skipping deduplication")
                return False
                
            # Get items from the same year or within the year difference threshold
            if self.year_diff_threshold == 0:
                # If threshold is 0, only check items from the same year
                self.cursor.execute('''
                SELECT id, title, publication_date 
                FROM bibliographic_items 
                WHERE substr(publication_date, 1, 4) = ?
                AND id != ?  -- Exclude the current item
                ''', (year, new_item['id']))
            else:
                # If threshold > 0, check items within the year range
                year_int = int(year)
                min_year = year_int - self.year_diff_threshold
                max_year = year_int + self.year_diff_threshold
                self.cursor.execute('''
                SELECT id, title, publication_date 
                FROM bibliographic_items 
                WHERE CAST(substr(publication_date, 1, 4) AS INTEGER) BETWEEN ? AND ?
                AND id != ?  -- Exclude the current item
                ''', (min_year, max_year, new_item['id']))
            
            self.logger.debug(f"Checking for duplicates of item {new_item['id']}")
            for existing_item in self.cursor.fetchall():
                try:
                    existing_item_dict = {
                        'id': existing_item[0],
                        'title': existing_item[1],
                        'publication_date': existing_item[2]
                    }
                    
                    # Get the extracted year for the existing item
                    _, existing_year = self.normalize_date(existing_item_dict['publication_date'])
                    existing_item_dict['extracted_year'] = existing_year
                    
                    is_duplicate, keep_item = self.compare_items(new_item, existing_item_dict)
                    if is_duplicate:
                        self.logger.info(f"Found duplicate: {new_item['id']} and {existing_item_dict['id']}")
                        self.merge_items(new_item, existing_item_dict, keep_item)
                        return True
                except Exception as e:
                    self.logger.error(f"Error comparing items {new_item['id']} and {existing_item[0]}: {str(e)}")
                    continue
            
            return False
        except Exception as e:
            self.logger.error(f"Error in check_duplicates for item {new_item['id']}: {str(e)}")
            return False

    def print_summary(self):
        """Print summary of skipped records and deduplication results."""
        self.logger.info("\nSummary of skipped records:")
        for reason, count in self.skipped_records.items():
            self.logger.info(f"{reason}: {count} records")

    def process_directory(self, directory_path: str):
        """Process all JSON files in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            self.logger.error(f"Directory does not exist: {directory_path}")
            return

        # Look for JSON files
        json_files = list(directory.glob('**/*.json'))
        
        if not json_files:
            self.logger.error(f"No JSON files found in {directory_path}")
            return
            
        self.logger.info(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            self.process_json_file(str(json_file))
        
        # Print summary of skipped records
        self.print_summary()

    def process_json_file(self, file_path: str):
        """Process a single JSON file and save data to database."""
        import json
        
        try:
            self.logger.info(f"Starting to process JSON file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as jsonfile:
                records = json.load(jsonfile)
                
            total_records = len(records)
            self.logger.info(f"Found {total_records} records in {file_path}")

            processed_records = 0
            for record in records:
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
                    self.skipped_records['processing_error'] += 1
                    continue

            self.conn.commit()
            self.logger.info(f"Successfully processed {processed_records} records from {file_path}")

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return

    def process_json_record(self, record):
        """Process a single record from JSON and save its data to the database."""
        try:
            # Extract basic information
            item_id = record.get('id')
            if not item_id:
                self.logger.warning("No ID found in record")
                self.skipped_records['no_id'] += 1
                return
                
            self.logger.info(f"Processing item with ID: {item_id}")
            
            item_type = record.get('item_type', '')
            title = record.get('title', '')
            pub_date = record.get('publication_date', '')
            
            # Normalize the publication date and extract year
            normalized_date, extracted_year = self.normalize_date(pub_date)
            self.logger.debug(f"Normalized date from '{pub_date}' to '{normalized_date}', extracted year: {extracted_year}")
            
            subject = record.get('subject', '')
            pub_title = record.get('publication_title', '')
            
            # Create item dictionary for deduplication
            new_item = {
                'id': item_id,
                'title': title,
                'publication_date': normalized_date,
                'extracted_year': extracted_year,
                'item_type': item_type,
                'subject': subject,
                'publication_title': pub_title
            }

            # First, save the bibliographic item
            self.cursor.execute('''
            INSERT OR REPLACE INTO bibliographic_items 
            (id, item_type, title, publication_date, subject, publication_title)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (item_id, item_type, title, normalized_date, subject, pub_title))
            self.conn.commit()

            # Process authors - authors list comes directly from JSON
            authors = record.get('authors', [])
            
            self.logger.info(f"Found {len(authors)} authors for item {item_id}")
            
            for author in authors:
                if author:
                    # Clean the author name
                    cleaned_name = self.clean_japanese_name(author)
                    if cleaned_name:
                        # Generate a new UUID for this author
                        entity_id = str(uuid.uuid4())
                        
                        # Check if entity with this cleaned name already exists
                        self.cursor.execute('''
                        SELECT id, is_publisher, entity_type, type_detection_method 
                        FROM entities 
                        WHERE cleaned_name = ?
                        ''', (cleaned_name,))
                        result = self.cursor.fetchone()
                        
                        if result:
                            entity_id = result[0]
                            # If this entity was previously a publisher, update its type
                            if result[1]:
                                if self.use_method2:
                                    # For method 2, use get_item_type to determine type
                                    entity_type, mode = self.determine_entity_type_method2(author, result[3])
                                    if entity_type is None:
                                        entity_type = result[2]  # Keep existing type
                                        mode = result[3]  # Keep existing mode
                                else:
                                    # For method 1, set as organization
                                    entity_type = 'organization'
                                    mode = 3
                                
                                self.cursor.execute('''
                                UPDATE entities 
                                SET entity_type = ?, is_author = 1, type_detection_method = ?
                                WHERE id = ?
                                ''', (entity_type, mode, entity_id))
                        else:
                            # Determine entity type based on selected method
                            if self.use_method2:
                                entity_type, mode = self.determine_entity_type_method2(author)
                            else:
                                entity_type = 'person'
                                mode = 3
                            
                            self.cursor.execute('''
                            INSERT INTO entities 
                            (id, name, cleaned_name, entity_type, is_author, is_publisher, type_detection_method)
                            VALUES (?, ?, ?, ?, 1, 0, ?)
                            ''', (entity_id, author, cleaned_name, entity_type, mode))
                        
                        self.cursor.execute('''
                        INSERT OR IGNORE INTO item_entities 
                        (item_id, entity_id, relationship_type)
                        VALUES (?, ?, 'author')
                        ''', (item_id, entity_id))

            # Process publishers - publishers list comes directly from JSON
            publishers = record.get('publishers', [])
            
            self.logger.info(f"Found {len(publishers)} publishers for item {item_id}")
            
            for publisher in publishers:
                if publisher:
                    # Clean the publisher name
                    cleaned_name = self.clean_japanese_name(publisher)
                    if cleaned_name:
                        # Generate a new UUID for this publisher
                        entity_id = str(uuid.uuid4())
                        
                        # Check if entity with this cleaned name already exists
                        self.cursor.execute('''
                        SELECT id, is_author, entity_type, type_detection_method 
                        FROM entities 
                        WHERE cleaned_name = ?
                        ''', (cleaned_name,))
                        result = self.cursor.fetchone()
                        
                        if result:
                            entity_id = result[0]
                            # If this entity was previously an author, update its type
                            if result[1]:
                                if self.use_method2:
                                    # For method 2, use get_item_type to determine type
                                    entity_type, mode = self.determine_entity_type_method2(publisher, result[3])
                                    if entity_type is None:
                                        entity_type = result[2]  # Keep existing type
                                        mode = result[3]  # Keep existing mode
                                else:
                                    # For method 1, set as organization
                                    entity_type = 'organization'
                                    mode = 3
                                
                                self.cursor.execute('''
                                UPDATE entities 
                                SET entity_type = ?, is_publisher = 1, type_detection_method = ?
                                WHERE id = ?
                                ''', (entity_type, mode, entity_id))
                        else:
                            # Determine entity type based on selected method
                            if self.use_method2:
                                entity_type, mode = self.determine_entity_type_method2(publisher)
                            else:
                                entity_type = 'organization'
                                mode = 3
                            
                            self.cursor.execute('''
                            INSERT INTO entities 
                            (id, name, cleaned_name, entity_type, is_author, is_publisher, type_detection_method)
                            VALUES (?, ?, ?, ?, 0, 1, ?)
                            ''', (entity_id, publisher, cleaned_name, entity_type, mode))
                        
                        self.cursor.execute('''
                        INSERT OR IGNORE INTO item_entities 
                        (item_id, entity_id, relationship_type)
                        VALUES (?, ?, 'publisher')
                        ''', (item_id, entity_id))

            # After saving all data, check for duplicates
            if self.check_duplicates(new_item):
                self.logger.info(f"Item {item_id} was merged with an existing item")
                return

            self.conn.commit()
            self.logger.info(f"Successfully processed item {item_id}")
        except Exception as e:
            self.logger.error(f"Error in process_json_record: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def close(self):
        """Close database connection and save deduplication report."""
        if self.conn:
            self.save_deduplication_report()
            self.conn.close()
            self.logger.info("Database connection closed")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process NDL XML data files and save to database")
    parser.add_argument("--db-path", default="ndl_data_test.db", help="Path to SQLite database file")
    parser.add_argument("--use-method2", default="True", help="Use method 2 (GPT) for entity type detection")
    parser.add_argument("--auto-type-check", type=int, default=2, 
                        help="Auto type check level: 0=manual, 1=with confirmation, 2=without confirmation")
    parser.add_argument("--dedup-threshold", type=float, default=0.7, help="Threshold for deduplication similarity")
    parser.add_argument("--manual-dedup", default="False", help="Use manual deduplication")
    parser.add_argument("--year-diff-threshold", type=int, default=5, 
                        help="Year difference threshold for deduplication")
    parser.add_argument("--gpt-credibility-threshold", type=float, default=0.8, 
                        help="Minimum credibility threshold for GPT deduplication decisions")
    parser.add_argument("--data-dir", default="cinii_data/all_title_システム工学_until_1965", 
                        help="Directory containing XML files to process")
    parser.add_argument("--gui-mode", action="store_true", 
                        help="Run in GUI mode (default: False)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    parser.add_argument("--http-log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level for HTTP requests (default: WARNING)")
    parser.add_argument("--json-mode", action="store_true",
                        help="Process JSON files instead of XML files (default: False)")
    
    args = parser.parse_args()
    
    # Convert string arguments to appropriate types
    use_method2 = args.use_method2.lower() == "true"
    manual_dedup = args.manual_dedup.lower() == "true"
    
    processor = BiblioDataProcessor(
        db_path=args.db_path,
        use_method2=use_method2,
        auto_type_check=args.auto_type_check,
        dedup_threshold=args.dedup_threshold,
        manual_dedup=manual_dedup,
        year_diff_threshold=args.year_diff_threshold,
        gpt_credibility_threshold=args.gpt_credibility_threshold,
        gui_mode=args.gui_mode,
        log_level=args.log_level,
        http_log_level=args.http_log_level,
        json_mode=args.json_mode
    )
    try:
        processor.process_directory(args.data_dir)
    except Exception as e:
        logger.error(f"Error in main process: {e}")
    finally:
        processor.close()

if __name__ == '__main__':
    main()
   
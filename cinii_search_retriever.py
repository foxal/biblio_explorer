import requests
import os
import urllib.parse
import time
import argparse
import re
from pathlib import Path
import xml.dom.minidom
import glob
import logging
import shutil
from datetime import datetime
import json
import xml.etree.ElementTree as ET
import uuid

class CiNiiSearchAPI:
    """
    Client for retrieving bibliographic data from the CiNii Research OpenSearch API
    and saving it as XML files.
    """
    
    BASE_URL = "https://cir.nii.ac.jp/opensearch"
    DEFAULT_PARAMS = {
        "count": "100",  # Default records per request
        "sortorder": "1",  # Default sort by publication date: older first
        "format": "atom"  # Default format is atom (has more detail than RSS)
    }
    
    # Namespace mapping for XML parsing
    NAMESPACES = {
        'atom': 'http://www.w3.org/2005/Atom',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'prism': 'http://prismstandard.org/namespaces/basic/2.0/',
        'ndl': 'http://ndl.go.jp/dcndl/terms',
        'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
        'cir': 'https://cir.nii.ac.jp/schema/1.0/'
    }
    
    def __init__(self, output_dir="cinii_data", count=100, gui_mode=False):
        """Initialize the API client with output directory and count of records per request."""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.count = count
        self.DEFAULT_PARAMS["count"] = str(count)
        self.search_dirs = []  # Track directories created for searches
        self.gui_mode = gui_mode
        
        # Set up logging
        log_file = os.path.join(output_dir, 'cinii_search.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('CiNiiSearchAPI')
        
        # Also log to console if not in GUI mode
        if not gui_mode:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)
    
    def search(self, search_type, **kwargs):
        """
        Main search method that dispatches to specific search API endpoints based on search_type.
        
        Args:
            search_type: One of 'all', 'data', 'articles', 'books', 'dissertations', 'projects'
            **kwargs: Search parameters like title, creator, from, until, etc.
        
        Returns:
            Path to the directory with results if successful, False otherwise
        """
        # Validate search type
        valid_types = ['all', 'data', 'articles', 'books', 'dissertations', 'projects']
        if search_type not in valid_types:
            self.logger.error(f"Invalid search type: {search_type}. Must be one of {valid_types}")
            return False
        
        # Build query parameters
        params = self.DEFAULT_PARAMS.copy()
        
        # Add all provided parameters
        for key, value in kwargs.items():
            if value:  # Only add non-empty parameters
                params[key] = value
        
        # Execute the search
        return self._execute_search(search_type, params)
    
    def search_by_title(self, title, search_type='all', **kwargs):
        """Search for items by title keywords."""
        params = {'title': title}
        params.update(kwargs)
        return self.search(search_type, **params)
    
    def search_by_creator(self, creator, search_type='all', **kwargs):
        """Search for items by creator name."""
        params = {'creator': creator}
        params.update(kwargs)
        return self.search(search_type, **params)
    
    def search_by_category(self, category, search_type='all', **kwargs):
        """Search for items by category code."""
        params = {'category': category}
        params.update(kwargs)
        return self.search(search_type, **params)
    
    def search_by_researcher_id(self, researcher_id, search_type='all', **kwargs):
        """Search for items by researcher ID."""
        params = {'researcherId': researcher_id}
        params.update(kwargs)
        return self.search(search_type, **params)
    
    def _execute_search(self, search_type, params):
        """Execute the search with the given parameters and save results."""
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
            if current_position == 1:
                self.logger.info("Starting from the beginning")
            else:
                self.logger.info(f"Resuming from page {current_position}")
        else:
            Path(search_path).mkdir(parents=True, exist_ok=True)
            current_position = 1  # Start from the beginning
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
            total_pages = (total_results + self.count - 1) // self.count
            self.logger.info(f"Total pages to fetch: {total_pages}")
            
            # Move to the next page
            current_position = 2
        else:
            # We're resuming, so find the total pages from the first page file
            first_page_file = os.path.join(search_path, "results_page_1.xml")
            if os.path.exists(first_page_file):
                with open(first_page_file, 'r', encoding='utf-8') as f:
                    first_page_content = f.read()
                total_results = self._get_total_results_count(first_page_content)
                total_pages = (total_results + self.count - 1) // self.count
                self.logger.info(f"Resuming with {total_results} total results, {total_pages} total pages")
            else:
                self.logger.error("First page file not found. Cannot determine total pages")
                return False
        
        # Fetch remaining pages
        while current_position <= total_pages:
            self.logger.info(f"Fetching page {current_position} of {total_pages}")
            page_params = params.copy()
            page_params['start'] = str((current_position - 1) * self.count + 1)
            
            batch_results = self._make_request(endpoint, page_params)
            if batch_results is None:
                self.logger.error(f"Failed to fetch page {current_position}")
                break
            
            self._save_result(batch_results, search_path, current_position)
            current_position += 1
            
            time.sleep(2)  # Be nice to the API
        
        self.logger.info(f"All data saved to {search_path}")
        return search_path
    
    def _find_existing_search_dir(self, search_dir_pattern):
        """Find an existing directory that matches the search parameters."""
        pattern = os.path.join(self.output_dir, search_dir_pattern)
        matching_dirs = sorted(glob.glob(pattern))
        
        return matching_dirs[-1] if matching_dirs else None
    
    def _get_next_page_number(self, search_path):
        """Determine the next page number from existing XML files."""
        pattern = os.path.join(search_path, "results_page_*.xml")
        page_files = sorted(
            glob.glob(pattern),
            key=lambda x: int(re.search(r'page_(\d+)', x).group(1))
        )
        
        if not page_files:
            return 1  # No files yet, start from the beginning
        
        # Get the highest page number and add 1
        last_page = max([int(re.search(r'page_(\d+)', f).group(1)) for f in page_files])
        return last_page + 1
    
    def _make_request(self, url, params):
        """Make an API request with the given parameters."""
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request error: {e}")
            return None
    
    def _get_total_results_count(self, xml_content):
        """Extract the total number of results from the XML response."""
        try:
            root = ET.fromstring(xml_content)
            # Find the totalResults element in the XML
            for ns in ['opensearch', 'http://a9.com/-/spec/opensearch/1.1/']:
                try:
                    # Try with namespace prefix first
                    total_results = root.find(f'.//{{{ns}}}totalResults')
                    if total_results is not None:
                        return int(total_results.text)
                except:
                    pass
            
            # If still not found, try with regex as a fallback
            match = re.search(r'<opensearch:totalResults.*?>(\d+)</opensearch:totalResults>', xml_content)
            if match:
                return int(match.group(1))
            
            return 0
        except Exception as e:
            self.logger.error(f"Error parsing total results: {e}")
            return None
    
    def _save_result(self, xml_content, directory, page):
        """Save XML content to a file."""
        try:
            # Pretty format the XML
            dom = xml.dom.minidom.parseString(xml_content)
            pretty_xml = dom.toprettyxml(indent="  ")
            
            # Save to file
            filename = os.path.join(directory, f"results_page_{page}.xml")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
            self.logger.info(f"Saved page {page} to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving XML: {e}")
            return False

    def export_results_to_json(self, search_dir=None):
        """
        Process all XML files in the specified directory
        and save extracted data to a JSON file.
        """
        # If no directory specified, process the latest one
        if not search_dir:
            # Get all directories in the output directory
            search_dirs = [d for d in os.listdir(self.output_dir) 
                          if os.path.isdir(os.path.join(self.output_dir, d))]
            
            if not search_dirs:
                self.logger.error("No search directories found")
                return False
                
            # Get the latest directory based on modification time
            search_dirs_with_time = [(d, os.path.getmtime(os.path.join(self.output_dir, d))) 
                                    for d in search_dirs]
            latest_dir = sorted(search_dirs_with_time, key=lambda x: x[1], reverse=True)[0][0]
            search_dir = os.path.join(self.output_dir, latest_dir)
        elif not os.path.isabs(search_dir):
            # Make sure we don't add output_dir twice if search_dir already includes it
            if not search_dir.startswith(self.output_dir):
                search_dir = os.path.join(self.output_dir, search_dir)
            
        # Normalize the path to remove any duplicate directory elements
        search_dir = os.path.normpath(search_dir)
        self.logger.info(f"Processing XML files in {search_dir} for JSON export")
        
        # Create JSON output file
        json_filename = os.path.join(search_dir, "cinii_records.json")
        
        # Find all XML files
        xml_pattern = os.path.join(search_dir, "results_page_*.xml")
        xml_files = sorted(glob.glob(xml_pattern), 
                          key=lambda x: int(re.search(r'page_(\d+)', x).group(1)))
        
        if not xml_files:
            self.logger.error(f"No XML files found in {search_dir}")
            self.logger.error(f"Attempted to find files matching pattern: {xml_pattern}")
            
            # Check if the directory exists and list its contents for debugging
            if os.path.isdir(search_dir):
                self.logger.info(f"Directory exists. Contents: {os.listdir(search_dir)}")
            else:
                self.logger.error(f"Directory {search_dir} does not exist")
            return False
            
        total_records = 0
        extracted_records = []
        
        # Process each XML file
        for xml_file in xml_files:
            self.logger.info(f"Processing {os.path.basename(xml_file)}")
            try:
                # Parse XML
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Find all entries
                entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
                self.logger.info(f"Found {len(entries)} entries in {os.path.basename(xml_file)}")
                
                # Process each entry
                for entry in entries:
                    try:
                        record_data = self.extract_record_data(entry)
                        if record_data:
                            extracted_records.append(record_data)
                            total_records += 1
                    except Exception as e:
                        self.logger.error(f"Error processing entry: {str(e)}")
                        continue
                        
            except ET.ParseError as e:
                self.logger.error(f"XML parsing error in {xml_file}: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Error processing file {xml_file}: {str(e)}")
                continue
        
        # Save to JSON
        try:
            with open(json_filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(extracted_records, jsonfile, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Successfully exported {total_records} records to {json_filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error writing JSON file: {str(e)}")
            return False
    
    def extract_record_data(self, entry):
        """
        Extract bibliographic data from a single entry element.
        Returns a dictionary with the extracted data.
        """
        # Initialize dictionary for JSON data (matching NDL export format)
        json_record = {
            "id": "",
            "item_type": "",
            "title": "",
            "publication_date": "",
            "subject": "",
            "publication_title": "",
            "authors": [],
            "publishers": []
        }

        try:
            # Extract ID from atom:id element
            id_elem = entry.find('.//{http://www.w3.org/2005/Atom}id')
            if id_elem is not None and id_elem.text:
                json_record["id"] = id_elem.text.strip()
                # Also extract a shorter ID from the URL
                if '/' in id_elem.text:
                    short_id = id_elem.text.split('/')[-1]
                    if '#' in short_id:
                        short_id = short_id.split('#')[0]
                    json_record["id"] = short_id
            
            # Extract item type from dc:type element
            type_elem = entry.find('.//{http://purl.org/dc/elements/1.1/}type')
            if type_elem is not None and type_elem.text:
                json_record["item_type"] = type_elem.text.strip()

            # Extract title from atom:title element
            title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
            if title_elem is not None and title_elem.text:
                json_record["title"] = title_elem.text.strip()
            
            # Extract publication date
            date_elem = entry.find('.//{http://prismstandard.org/namespaces/basic/2.0/}publicationDate')
            if date_elem is not None and date_elem.text:
                json_record["publication_date"] = date_elem.text.strip()
            
            # Extract authors
            author_elems = entry.findall('.//{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name')
            for author_elem in author_elems:
                if author_elem.text:
                    json_record["authors"].append(author_elem.text.strip())
            
            # Extract publishers
            publisher_elem = entry.find('.//{http://purl.org/dc/elements/1.1/}publisher')
            if publisher_elem is not None and publisher_elem.text:
                json_record["publishers"].append(publisher_elem.text.strip())
            
            # Extract publication title for articles
            if json_record["item_type"].lower() == "article":
                pub_title_elem = entry.find('.//{http://prismstandard.org/namespaces/basic/2.0/}publicationName')
                if pub_title_elem is not None and pub_title_elem.text:
                    json_record["publication_title"] = pub_title_elem.text.strip()
            
            # Extract subject/keywords if available
            subject_elem = entry.find('.//{http://purl.org/dc/elements/1.1/}subject')
            if subject_elem is not None and subject_elem.text:
                json_record["subject"] = subject_elem.text.strip()
            
            self.logger.info(f"Extracted data for {json_record['id']}: {json_record['title']}")
            return json_record
            
        except Exception as e:
            self.logger.error(f"Error extracting data from entry: {str(e)}")
            return None


def main():
    parser = argparse.ArgumentParser(description="CiNii Search API Data Retriever")
    parser.add_argument("--search-type", required=True, 
                      choices=["all", "data", "articles", "books", "dissertations", "projects"],
                      help="Search type: all, data, articles, books, dissertations, or projects")
    parser.add_argument("--title", help="Title keywords to search")
    parser.add_argument("--creator", help="Creator/author name to search")
    parser.add_argument("--researcher-id", help="Researcher ID to search")
    parser.add_argument("--category", help="Category code to search")
    parser.add_argument("--from", dest="from_year", help="Starting year for search range")
    parser.add_argument("--until", dest="until_year", help="Ending year for search range")
    parser.add_argument("--count", type=int, default=100,
                      help="Number of records per request (default: 100)")
    parser.add_argument("--output-dir", default="cinii_data",
                      help="Directory to save XML files (default: cinii_data)")
    parser.add_argument("--gui-mode", action="store_true",
                      help="Run in GUI mode (default: False)")
    parser.add_argument("--export-json", action="store_true",
                      help="Export search results to JSON after retrieval")
    
    args = parser.parse_args()
    
    # Initialize the API client
    api = CiNiiSearchAPI(output_dir=args.output_dir, count=args.count, gui_mode=args.gui_mode)
    
    # Build search parameters
    search_params = {
        'from': args.from_year,
        'until': args.until_year,
    }
    
    # Execute search based on provided parameters
    search_result = None
    success = False
    
    if args.title:
        search_params['title'] = args.title
    
    if args.creator:
        search_params['creator'] = args.creator
    
    if args.category:
        search_params['category'] = args.category
    
    if args.researcher_id:
        search_params['researcherId'] = args.researcher_id
    
    # Perform the search
    search_result = api.search(args.search_type, **search_params)
    success = search_result is not False
    
    # Export to JSON if requested and search was successful
    if success and args.export_json:
        api.export_results_to_json(search_result)

    return success


if __name__ == "__main__":
    main() 
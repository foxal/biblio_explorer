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

class NDLSearchAPI:
    """
    Client for retrieving bibliographic data from the NDL Search API
    and saving it as XML files.
    """
    
    BASE_URL = "https://ndlsearch.ndl.go.jp/api/sru"
    DEFAULT_PARAMS = {
        "operation": "searchRetrieve",
        "version": "1.2",
        "recordPacking": "xml",
        "recordSchema": "dcndl",
        "maximumRecords": "200"  # Default max records per request
    }
    MAX_ALLOWED_RECORDS = 500  # Maximum records the API can return for a single query
    
    # Namespace mapping for XML parsing
    NAMESPACES = {
        'srw': 'http://www.loc.gov/zing/srw/',
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'dcterms': 'http://purl.org/dc/terms/',
        'dcndl': 'http://ndl.go.jp/dcndl/terms/',
        'foaf': 'http://xmlns.com/foaf/0.1/'
    }
    
    def __init__(self, output_dir="ndl_data", max_records=200, gui_mode=False, log_level='INFO'):
        """Initialize the API client with output directory and maximum records per request."""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.max_records = min(max_records, self.MAX_ALLOWED_RECORDS)
        self.DEFAULT_PARAMS["maximumRecords"] = str(self.max_records)
        self.yearly_search_dirs = []  # Track directories created for yearly searches
        self.gui_mode = gui_mode
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Set up logging
        log_file = os.path.join(output_dir, 'ndl_search.log')
        logging.basicConfig(
            filename=log_file,
            level=self.log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('NDLSearchAPI')
        
        # Also log to console if not in GUI mode
        if not gui_mode:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            self.logger.addHandler(console_handler)
    
    def search_by_title(self, title, ndc=None, from_year=None, until_year=None):
        """Search for items by title keywords."""
        query_parts = [f'title="{title}"']
        result = self._execute_smart_search("title", title, query_parts, ndc, from_year, until_year)
        if result and len(self.yearly_search_dirs) > 1:
            return self._consolidate_search_results("title", title, ndc, from_year, until_year)
        return result
    
    def search_by_creator(self, creator, ndc=None, from_year=None, until_year=None):
        """Search for items by creator name."""
        query_parts = [f'creator="{creator}"']
        result = self._execute_smart_search("creator", creator, query_parts, ndc, from_year, until_year)
        if result and len(self.yearly_search_dirs) > 1:
            return self._consolidate_search_results("creator", creator, ndc, from_year, until_year)
        return result
    
    def search_by_ndc(self, ndc, from_year=None, until_year=None):
        """Search for items by NDC code only."""
        query_parts = [f'ndc="{ndc}"']
        result = self._execute_smart_search("ndc", ndc, query_parts, None, from_year, until_year)
        if result and len(self.yearly_search_dirs) > 1:
            return self._consolidate_search_results("ndc", ndc, None, from_year, until_year)
        return result
    
    def search_by_custom_query(self, custom_query):
        """Search using a custom query string directly."""
        # For custom queries, we can't easily split by year, so just execute directly
        return self._execute_search("custom", custom_query, [], None, None, None, custom_query=custom_query)
    
    def _execute_smart_search(self, mode, query_term, query_parts, ndc=None, from_year=None, until_year=None):
        """
        Execute a search with smart handling of results exceeding 500 records.
        Will break down searches by year if needed.
        Returns search path if successful, False otherwise.
        """
        # Reset yearly search directories list
        self.yearly_search_dirs = []
        
        # First do a trial query to check the total count
        params = self.DEFAULT_PARAMS.copy()
        
        # Build the full query string
        full_query_parts = query_parts.copy()
        if ndc and mode != "ndc":
            full_query_parts.append(f'ndc="{ndc}"')
        
        if from_year and until_year:
            full_query_parts.append(f'from="{from_year}" AND until="{until_year}"')
        elif from_year:
            full_query_parts.append(f'from="{from_year}"')
        elif until_year:
            full_query_parts.append(f'until="{until_year}"')
        
        full_query = " AND ".join(full_query_parts)
        params["query"] = full_query
        params["maximumRecords"] = "1"  # Just need to check the count, not fetch records
        
        self.logger.info(f"Executing trial search: {full_query}")
        results = self._make_request(params)
        
        if results is None:
            self.logger.error("Trial request failed.")
            return False
        
        num_records = self._get_number_of_records(results)
        self.logger.info(f"Trial search found {num_records} records.")
        
        # If results exceed the maximum allowed, break down by year
        if num_records > self.MAX_ALLOWED_RECORDS:
            self.logger.info(f"Search results exceed {self.MAX_ALLOWED_RECORDS} records. Breaking down by year.")
            
            # Determine the year range
            start_year = int(from_year) if from_year else datetime.now().year - 20
            end_year = int(until_year) if until_year else datetime.now().year
            
            last_search_path = None
            for year in range(start_year, end_year + 1):
                year_from = str(year)
                year_until = str(year)
                
                # Execute search for this specific year
                year_query_parts = query_parts.copy()
                if ndc and mode != "ndc":
                    year_query_parts.append(f'ndc="{ndc}"')
                
                year_query_parts.append(f'from="{year_from}" AND until="{year_until}"')
                year_query = " AND ".join(year_query_parts)
                
                # Check the count for this year
                params["query"] = year_query
                params["maximumRecords"] = "1"
                results = self._make_request(params)
                
                if results is None:
                    self.logger.error(f"Trial request for year {year} failed.")
                    continue
                
                year_count = self._get_number_of_records(results)
                
                if year_count == 0:
                    self.logger.info(f"No records found for year {year}. Skipping.")
                    continue
                
                if year_count > self.MAX_ALLOWED_RECORDS:
                    self.logger.warning(f"Year {year} has {year_count} records, which exceeds the maximum of {self.MAX_ALLOWED_RECORDS}. Some records may be missed.")
                    
                # Now execute the actual search for this year
                search_path = self._execute_search(
                    mode,
                    query_term,
                    year_query_parts,
                    ndc if mode != "ndc" else None,  # Pass ndc if mode is not "ndc"
                    year_from,
                    year_until
                )
                
                if search_path:
                    last_search_path = search_path
                
                time.sleep(2)  # Be nice to the API between years
            
            # If we found at least one year with results, return the path to the last one
            # This will be consolidated later if needed
            return last_search_path if last_search_path else False
        else:
            # If the count is manageable, just execute the search as normal
            return self._execute_search(mode, query_term, query_parts, ndc, from_year, until_year)
    
    def _execute_search(self, mode, query_term, query_parts=None, ndc=None, from_year=None, until_year=None, custom_query=None):
        """Execute the search with the given parameters and save results."""
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
            
            if num_records == 0:
                self.logger.info("No records found.")
                return False
            
            # Check if the number of records exceeds the maximum
            if num_records > self.MAX_ALLOWED_RECORDS:
                warning_msg = f"The query returns {num_records} records, which exceeds the maximum of {self.MAX_ALLOWED_RECORDS}. Only the first {self.MAX_ALLOWED_RECORDS} records will be retrieved."
                self.logger.warning(warning_msg)
                num_records = self.MAX_ALLOWED_RECORDS
            
            # Save the first batch
            self._save_result(results, search_path, 1)
            
            # Move to the next batch
            next_position = self._get_next_record_position_from_xml(results)
            if next_position:
                current_position = next_position
            else:
                current_position = 1 + self.max_records
        else:
            # We're resuming, so read the number of records from the first page file
            first_page_file = os.path.join(search_path, "results_page_1.xml")
            if os.path.exists(first_page_file):
                with open(first_page_file, 'r', encoding='utf-8') as f:
                    first_page_content = f.read()
                num_records = self._get_number_of_records(first_page_content)
                self.logger.info(f"Resuming search with {num_records} total records from position {current_position}")
                
                # Check if the number of records exceeds the maximum
                if num_records > self.MAX_ALLOWED_RECORDS:
                    num_records = self.MAX_ALLOWED_RECORDS
            else:
                self.logger.error("First page file not found. Cannot determine total record count.")
                return False
        
        # Fetch remaining batches, but limit to MAX_ALLOWED_RECORDS
        while current_position <= num_records and current_position <= self.MAX_ALLOWED_RECORDS:
            self.logger.info(f"Fetching records {current_position}-{min(current_position+self.max_records-1, num_records)}")
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
            else:
                # If no next record position is provided, just increment by the batch size
                current_position += self.max_records
            
            time.sleep(2)  # Be nice to the API
        
        self.logger.info(f"All data saved to {search_path}")
        return search_path
    
    def _consolidate_search_results(self, mode, query_term, ndc=None, from_year=None, until_year=None):
        """
        Consolidate search results from multiple yearly directories into a single directory.
        After consolidation, move original directories to a tmpfiles directory.
        Returns the path to the consolidated directory.
        """
        if not self.yearly_search_dirs:
            self.logger.info("No yearly search directories to consolidate.")
            return False
        
        self.logger.info(f"Consolidating results from {len(self.yearly_search_dirs)} directories.")
        
        # Create a consolidated directory
        consolidated_dir_name = f"{mode}_{query_term.replace('/', '_')}_{from_year}_{until_year}"
        consolidated_path = os.path.join(self.output_dir, consolidated_dir_name)
        consolidated_path = os.path.normpath(consolidated_path)  # Normalize the path
        Path(consolidated_path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created consolidated directory: {consolidated_path}")
        
        # Keep track of the next page number for the consolidated results
        consolidated_page = 1
        
        # Process each yearly directory
        for yearly_dir in sorted(self.yearly_search_dirs):
            self.logger.info(f"Processing directory: {yearly_dir}")
            
            # Get all XML result files in this directory
            xml_files = sorted(
                glob.glob(os.path.join(yearly_dir, "results_page_*.xml")),
                key=lambda x: int(re.search(r'page_(\d+)', x).group(1))
            )
            
            for xml_file in xml_files:
                # Create a new filename in the consolidated directory
                new_filename = os.path.join(consolidated_path, f"results_page_{consolidated_page}.xml")
                
                # Copy the file
                shutil.copy2(xml_file, new_filename)
                self.logger.info(f"Copied {os.path.basename(xml_file)} to {os.path.basename(new_filename)}")
                
                # Increment the consolidated page counter
                consolidated_page += 1
        
        self.logger.info(f"Consolidation complete. {consolidated_page-1} files copied to {consolidated_path}")
        
        # Create a tmpfiles directory to store the original directories
        tmpfiles_dir = os.path.join(self.output_dir, "tmpfiles")
        Path(tmpfiles_dir).mkdir(parents=True, exist_ok=True)
        
        # Move the original directories to tmpfiles
        for yearly_dir in self.yearly_search_dirs:
            # Get the directory name without the full path
            dir_name = os.path.basename(yearly_dir)
            
            # Create the destination path
            dest_path = os.path.join(tmpfiles_dir, dir_name)
            
            # If the destination already exists, make it unique
            if os.path.exists(dest_path):
                timestamp = time.strftime("%Y%m%d%H%M%S")
                dest_path = os.path.join(tmpfiles_dir, f"{dir_name}_{timestamp}")
            
            # Move the directory
            shutil.move(yearly_dir, dest_path)
            self.logger.info(f"Moved {dir_name} to {dest_path}")
        
        self.logger.info(f"All original directories moved to {tmpfiles_dir}")
        return consolidated_path
    
    def _find_existing_search_dir(self, mode, query_term, from_year, until_year):
        """Find an existing directory that matches the search parameters."""
        sanitized_query = query_term.replace('/', '_')
        pattern = os.path.join(self.output_dir, f"{mode}_{sanitized_query}_{from_year}_{until_year}")
        matching_dirs = sorted(glob.glob(pattern))
        
        return matching_dirs[-1] if matching_dirs else None
    
    def _get_next_record_position(self, search_path):
        """Determine the next record position from existing XML files."""
        # Find the latest page file
        pattern = os.path.join(search_path, "results_page_*.xml")
        page_files = sorted(glob.glob(pattern), key=lambda x: int(re.search(r'page_(\d+)', x).group(1)))
        
        if not page_files:
            return 1  # No files yet, start from the beginning
        
        latest_file = page_files[-1]
        
        # Extract the next record position from the XML
        with open(latest_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        next_position = self._get_next_record_position_from_xml(content)
        if next_position:
            return next_position
        
        # If no next position in the XML, calculate based on page number
        page_num = int(re.search(r'page_(\d+)', latest_file).group(1))
        return (page_num * self.max_records) + 1
    
    def _get_next_record_position_from_xml(self, xml_content):
        """Extract the nextRecordPosition from XML content."""
        try:
            dom = xml.dom.minidom.parseString(xml_content)
            next_pos_nodes = dom.getElementsByTagName('nextRecordPosition')
            if next_pos_nodes and next_pos_nodes[0].firstChild:
                return int(next_pos_nodes[0].firstChild.nodeValue)
            return None
        except Exception as e:
            self.logger.error(f"Error parsing next record position: {e}")
            return None
    
    def _make_request(self, params):
        """Make an API request with the given parameters."""
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request error: {e}")
            return None
    
    def _get_number_of_records(self, xml_content):
        """Extract the number of records from the XML response."""
        try:
            dom = xml.dom.minidom.parseString(xml_content)
            num_records_nodes = dom.getElementsByTagName('numberOfRecords')
            if num_records_nodes:
                return int(num_records_nodes[0].firstChild.nodeValue)
            return 0
        except Exception as e:
            self.logger.error(f"Error parsing XML: {e}")
            return 0
    
    def _save_result(self, xml_content, directory, page):
        """Save XML content to a file."""
        try:
            # Format and prettify the XML
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
        Process all XML files in the specified directory (or all yearly directories)
        and save extracted data to a JSON file without any additional processing.
        """
        # If no directory specified, process the latest one
        if not search_dir:
            # Get all directories in the output directory except "tmpfiles"
            search_dirs = [d for d in os.listdir(self.output_dir) 
                          if os.path.isdir(os.path.join(self.output_dir, d)) and d != "tmpfiles"]
            
            if not search_dirs:
                self.logger.error("No search directories found")
                return False
                
            # Get the latest directory based on modification time, not alphabetical sorting
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
        json_filename = os.path.join(search_dir, "ndl_records.json")
        
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
                
                # Find all records
                records = root.findall('.//srw:record', self.NAMESPACES)
                self.logger.info(f"Found {len(records)} records in {os.path.basename(xml_file)}")
                
                # Process each record
                for record in records:
                    try:
                        record_data = self.extract_record_data(record)
                        if record_data:
                            extracted_records.append(record_data)
                            total_records += 1
                    except Exception as e:
                        self.logger.error(f"Error processing record: {str(e)}")
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
    
    def extract_record_data(self, record):
        """
        Extract bibliographic data from a single XML record.
        Only extract raw data - no additional processing.
        """
        record_data = record.find('.//rdf:RDF', self.NAMESPACES)
        if record_data is None:
            self.logger.warning("No RDF data found in record")
            return None

        # Initialize dictionary for JSON data
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

        # Process each bibliographic item
        for bib_admin in record_data.findall('.//dcndl:BibAdminResource', self.NAMESPACES):
            # Extract basic information
            about_attr = bib_admin.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            if not about_attr:
                self.logger.warning("No rdf:about attribute found in BibAdminResource")
                return None

            item_id = self.extract_id(about_attr)
            json_record["id"] = item_id
            self.logger.info(f"Processing item with ID: {item_id}")

            description = bib_admin.find('.//dcterms:description', self.NAMESPACES)
            if description is None or not description.text:
                self.logger.warning(f"No description found for item {item_id}")
                return None

            item_type = description.text.split(':')[-1].strip()
            json_record["item_type"] = item_type
            
            # Find the corresponding BibResource
            bib_resource = record_data.find(f'.//dcndl:BibResource[@rdf:about="https://ndlsearch.ndl.go.jp/books/{item_id}#material"]', self.NAMESPACES)
            if bib_resource is None:
                self.logger.warning(f"No BibResource found for item {item_id}")
                return None

            # Extract title
            title = self.get_text_content(bib_resource, './/dcterms:title')
            json_record["title"] = title if title else ""

            # Extract publication date (raw, not normalized)
            pub_date = self.get_text_content(bib_resource, './/dcterms:date')
            json_record["publication_date"] = pub_date if pub_date else ""

            # Extract subject
            subject = self.get_text_content(bib_resource, './/dcterms:subject')
            json_record["subject"] = subject if subject else ""

            # Extract publication title (for articles)
            if item_type == 'article':
                pub_title = self.get_text_content(bib_resource, './/dcndl:publicationName')
                json_record["publication_title"] = pub_title if pub_title else ""

            # Process authors - extract raw names
            authors = bib_resource.findall('.//dcterms:creator//foaf:name', self.NAMESPACES)
            self.logger.info(f"Found {len(authors)} authors for item {item_id}")
            
            author_names = []
            for author in authors:
                if author.text:
                    author_names.append(author.text)
            
            json_record["authors"] = author_names

            # Process publishers - extract raw names
            publishers = bib_resource.findall('.//dcterms:publisher//foaf:name', self.NAMESPACES)
            self.logger.info(f"Found {len(publishers)} publishers for item {item_id}")
            
            publisher_names = []
            for publisher in publishers:
                if publisher.text:
                    publisher_names.append(publisher.text)
            
            json_record["publishers"] = publisher_names

            return json_record

        return None

    def get_text_content(self, element, xpath):
        """Safely get text content from an element using xpath."""
        result = element.find(xpath, self.NAMESPACES)
        if result is not None:
            # Handle nested rdf:value if present
            value = result.find('.//rdf:value', self.NAMESPACES)
            if value is not None:
                return value.text
            return result.text
        return None

    def extract_id(self, about_attr):
        """Extract ID from rdf:about attribute."""
        return about_attr.split('/')[-1].split('#')[0]


def main():
    parser = argparse.ArgumentParser(description="NDL Search API Data Retriever")
    parser.add_argument("--mode", required=True, choices=["title", "creator", "ndc", "custom"],
                      help="Search mode: title, creator, ndc, or custom")
    parser.add_argument("--query", required=True,
                      help="Search term (title keywords, creator name, NDC code, or complete query string for custom mode)")
    parser.add_argument("--ndc", help="NDC code filter (not required when mode is ndc)")
    parser.add_argument("--from-year", help="Starting year for search range")
    parser.add_argument("--until-year", help="Ending year for search range")
    parser.add_argument("--output-dir", default="ndl_data",
                      help="Directory to save XML files (default: ndl_data)")
    parser.add_argument("--max-records", type=int, default=200,
                      help="Maximum number of records per request (default: 200)")
    parser.add_argument("--gui-mode", action="store_true",
                      help="Run in GUI mode (default: False)")
    parser.add_argument("--export-json", action="store_true",
                      help="Export search results to JSON after retrieval")
    
    args = parser.parse_args()
    
    # Initialize the API client
    api = NDLSearchAPI(output_dir=args.output_dir, max_records=args.max_records, gui_mode=args.gui_mode)
    
    # Execute search based on mode
    search_result = None
    success = False
    
    if args.mode == "title":
        search_result = api.search_by_title(args.query, args.ndc, args.from_year, args.until_year)
        success = search_result is not False
    elif args.mode == "creator":
        search_result = api.search_by_creator(args.query, args.ndc, args.from_year, args.until_year)
        success = search_result is not False
    elif args.mode == "ndc":
        search_result = api.search_by_ndc(args.query, args.from_year, args.until_year)
        success = search_result is not False
    elif args.mode == "custom":
        search_result = api.search_by_custom_query(args.query)
        success = search_result is not False
    
    # Export to JSON if requested and search was successful
    if success and args.export_json:
        # If search_result is a path (from consolidation), use that directly
        if isinstance(search_result, str) and os.path.isdir(search_result):
            # Normalize the path to fix any duplicate directory issues
            search_result = os.path.normpath(search_result)
            api.export_results_to_json(search_result)
        else:
            # Otherwise, let the export function find the latest directory
            api.export_results_to_json()


if __name__ == "__main__":
    main()
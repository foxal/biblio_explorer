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
from lxml import etree
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
        'foaf': "http://xmlns.com/foaf/0.1/",
        'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'prism': 'http://prismstandard.org/namespaces/basic/2.0/',
        'ndl': 'http://ndl.go.jp/dcndl/terms',
        'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
        'cir': 'https://cir.nii.ac.jp/schema/1.0/'
    }

    def __init__(self, output_dir="cinii_data", author_from_rdf=False, count=100, gui_mode=False, log_level="INFO"):
        """Initialize the API client with output directory and count of records per request."""
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.author_from_rdf = author_from_rdf
        self.count = count
        self.DEFAULT_PARAMS["count"] = str(count)
        self.search_dirs = []  # Track directories created for searches
        self.gui_mode = gui_mode
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)

        # Set up logging
        log_file = os.path.join(output_dir, 'cinii_search.log')
        logging.basicConfig(
            filename=log_file,
            level=self.log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('CiNiiSearchAPI')

        # Also log to console if not in GUI mode
        if not gui_mode:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            self.logger.addHandler(console_handler)

    def search(self, search_types: list, **kwargs) -> list:
        """
        Main search method that dispatches to specific search API endpoints based on search_types.

        Args:
            search_types: A list containing one or several of 'all', 'data', 'articles', 'books', 'dissertations', 'projects'. String is also accepted for backward compatibility.
            **kwargs: Search parameters like title, creator, from, until, etc.

        Returns:
            A list of paths to the directory with results if successful. If any search fails, the corresponding element in the list will be False.
        """
        # Ensure search_types is a list. To keep it backward compatible
        if not isinstance(search_types, list):
            search_types = [search_types]

        valid_types = ['all', 'data', 'articles', 'books', 'dissertations', 'projects']
        search_result_dirs = []

        for search_type in search_types:
            if search_type not in valid_types:
                self.logger.error(f"Invalid search type: {search_type}. Must be one of {valid_types}")
                search_result_dirs.append(False)
                continue

            # Build query parameters
            params = self.DEFAULT_PARAMS.copy()

            # Add all provided parameters
            for key, value in kwargs.items():
                if value:  # Only add non-empty parameters
                    params[key] = value

            # # When search type is books, articles, or projects, do author name disambiguation by converting the creator name to researcherId. Replace creator with researcherId. Delete creator.
            # if search_type in ['books', 'articles', 'projects'] and 'creator' in params:
            #     params['researcherId'] = self._get_researcher_id(params['creator'])
            #     params.pop('creator', None)

            # Execute the search
            search_result_dirs.append(self._execute_search(search_type, params))

        return search_result_dirs

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
                self.logger.debug("Starting from the beginning")
            else:
                self.logger.debug(f"Resuming from page {current_position}")
        else:
            Path(search_path).mkdir(parents=True, exist_ok=True)
            current_position = 1  # Start from the beginning
            self.logger.debug(f"Created new search directory: {search_path}")

        # Track this directory for potential use later
        self.search_dirs.append(search_path)

        # Construct the final URL for the API call
        endpoint = f"{self.BASE_URL}/{search_type}"
        self.logger.debug(f"Using API endpoint: {endpoint}")

        # First query to get total results count if starting fresh
        if current_position == 1:
            first_params = params.copy()
            first_params['start'] = '1'
            self.logger.debug(f"Executing initial search with parameters: {first_params}")

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

            self.logger.debug(f"Found {total_results} results")

            if total_results == 0:
                self.logger.debug("No results found")
                return False

            # Calculate total pages
            total_pages = (total_results + self.count - 1) // self.count
            self.logger.debug(f"Total pages to fetch: {total_pages}")

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
            self.logger.debug(f"Fetching page {current_position} of {total_pages}")
            page_params = params.copy()
            page_params['start'] = str((current_position - 1) * self.count + 1)

            batch_results = self._make_request(endpoint, page_params)
            if batch_results is None:
                self.logger.error(f"Failed to fetch page {current_position}")
                break

            self._save_result(batch_results, search_path, current_position)
            current_position += 1

            time.sleep(2)  # Be nice to the API

        self.logger.debug(f"All data saved to {search_path}")
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
            self.logger.debug(f"Saved page {page} to {filename}")
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
        self.logger.debug("Processing XML files in %s for JSON export", search_dir)

        # If a json file is already present, skip the export
        json_filename = os.path.join(search_dir, "cinii_records.json")
        if os.path.exists(json_filename):
            self.logger.debug("JSON file already exists in %s. Skipping export", search_dir)
            return True

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
            self.logger.debug(f"Processing {os.path.basename(xml_file)}")
            try:
                # Parse XML
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Find all entries
                entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
                self.logger.debug(f"Found {len(entries)} entries in {os.path.basename(xml_file)}")

                # Process each entry
                for entry in entries:
                    try:
                        record_data = self.extract_record_data(entry, search_dir)
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

    def _download_rdf(self, url, search_dir):
        """
        Download RDF file from the given URL.
        Returns the content as string if successful, None otherwise.
        """
        try:
            # If the RDF file already exists, load it from local file
            rdf_filename = os.path.join(search_dir, f"{url.split('/')[-1]}")
            if os.path.exists(rdf_filename):
                with open(rdf_filename, 'r', encoding='utf-8') as f:
                    return f.read()
            
            self.logger.debug(f"Downloading RDF from {url}")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                # Save the RDF to local file
                with open(rdf_filename, 'wb') as f:
                    f.write(response.content)
                return response.text
            else:
                self.logger.warning(f"Failed to download RDF: HTTP {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Error downloading RDF: {str(e)}")
            return None

    def _extract_authors_from_rdf(self, rdf_content):
        """
        Extract author information from RDF content.
        Prioritizes CINII_AUTHOR_ID, falls back to crid from rdf:about.
        Returns a list of dictionaries with author_name and author_id.
        """
        authors_data = []
        try:
            # Parse the RDF XML using lxml
            # recover=True can be helpful but might mask minor XML errors
            parser = etree.XMLParser(recover=True, encoding='utf-8')
            # Ensure content is bytes for the parser
            if isinstance(rdf_content, str):
                rdf_bytes = rdf_content.encode('utf-8')
            else:
                rdf_bytes = rdf_content # Assume it's already bytes

            root = etree.fromstring(rdf_bytes, parser)

            # Find all Researcher elements using XPath with the defined namespace prefix
            researchers = root.xpath(".//cir:Researcher", namespaces=self.NAMESPACES)

            # Process each researcher
            for researcher in researchers:
                author_name = ""
                author_id = ""
                found_cinii_id = False

                # Get author name using XPath. Prefers name without xml:lang, falls back to the first name found.
                # This handles cases where multiple foaf:name tags exist (e.g., with language variations)
                name_elems = researcher.xpath("./foaf:name[not(@xml:lang)] | ./foaf:name[1]", namespaces=self.NAMESPACES)
                if name_elems and name_elems[0].text:
                    author_name = name_elems[0].text.strip()

                # Try to find CINII_AUTHOR_ID first using XPath
                id_elems = researcher.xpath("./cir:personIdentifier", namespaces=self.NAMESPACES)
                if id_elems:
                    for id_elem in id_elems:
                        # Check the rdf:datatype attribute using XPath
                        datatype = id_elem.xpath("@rdf:datatype", namespaces=self.NAMESPACES)
                        # XPath returns a list for attributes, check if it's not empty and contains the target string
                        if datatype and "CINII_AUTHOR_ID" in datatype[0] and id_elem.text:
                            author_id = id_elem.text.strip()
                            found_cinii_id = True
                            break # Found the preferred ID, stop checking identifiers for this researcher

                # If no CINII_AUTHOR_ID was found, try getting the crid from rdf:about attribute
                if not found_cinii_id:
                    # Get the rdf:about attribute using XPath
                    rdf_about_list = researcher.xpath("@rdf:about", namespaces=self.NAMESPACES)
                    if rdf_about_list: # Check if the attribute exists
                        rdf_about = rdf_about_list[0]
                        # Check if it looks like a crid URL and extract the ID
                        if "/crid/" in rdf_about:
                            try:
                                potential_crid = rdf_about.split('/')[-1]
                                # Basic validation: check if it's numeric (CRIDs usually are)
                                if potential_crid.isdigit():
                                    author_id = potential_crid

                            except IndexError:
                                # Handle cases where splitting fails unexpectedly
                                self.logger.warning(f"Could not split rdf:about URL to get crid: '{rdf_about}'")

                # Append data only if an author name was found
                # (Decide if you want to add entries with ID but no name)
                if author_name:
                    authors_data.append({
                        "author_name": author_name,
                        "author_id": author_id # This will be CINII ID, crid, or "" if neither found/valid
                    })

        except etree.XMLSyntaxError as e:
            print(f"Error parsing RDF XML: {e}")
            # Handle error appropriately, maybe return empty list or raise exception
        except Exception as e:
            print(f"An unexpected error occurred during author extraction: {e}")
            # Handle error appropriately

        return authors_data

    def extract_record_data(self, entry, search_dir):
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
            if self.author_from_rdf:
                # Download rdf file and get authors and their CiNii author ids
                short_id = json_record["id"]
                rdf_url = f"https://cir.nii.ac.jp/crid/{short_id}.rdf"
                rdf_content = self._download_rdf(rdf_url, search_dir)

                if rdf_content:
                    authors_data = self._extract_authors_from_rdf(rdf_content)
                    json_record["authors"] = authors_data
                    self.logger.debug(f"Extracted {len(authors_data)} authors from RDF for {short_id}")
                else:
                    self.logger.warning(f"Failed to download RDF for {short_id}, falling back to basic author extraction")
                    # Fall back to basic author extraction if RDF download fails
                    author_elems = entry.findall('.//{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name')
                    for author_elem in author_elems:
                        if author_elem.text:
                            json_record["authors"].append({"author_name": author_elem.text.strip(), "author_id": ""})
            else:
                author_elems = entry.findall('.//{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name')
                for author_elem in author_elems:
                    if author_elem.text:
                        json_record["authors"].append({
                            "author_name": author_elem.text.strip(),
                            "author_id": ""
                        })

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

            self.logger.debug(f"Extracted data for {json_record['id']}: {json_record['title']}")
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
    parser.add_argument("--author-from-rdf", action="store_true",
                      help="Extract detailed author information from RDF files")

    args = parser.parse_args()

    # Initialize the API client
    api = CiNiiSearchAPI(output_dir=args.output_dir, count=args.count, gui_mode=args.gui_mode,
                        author_from_rdf=args.author_from_rdf)

    # Build search parameters
    search_params = {
        'from': args.from_year,
        'until': args.until_year,
    }

    # Execute search based on provided parameters
    if args.title:
        search_params['title'] = args.title

    if args.creator:
        search_params['creator'] = args.creator

    if args.category:
        search_params['category'] = args.category

    if args.researcher_id:
        search_params['researcherId'] = args.researcher_id

    # Perform the search
    search_result_dirs = api.search(args.search_type, **search_params)

    # Export to JSON if requested and search was successful
    for search_result_dir in search_result_dirs:
        if search_result_dir:
            api.export_results_to_json(search_result_dir)

    return None


if __name__ == "__main__":
    main()

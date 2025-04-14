# Technical details of Japanese Biblio Explorer's Modules


## Workflow

1. **Network Creation**:
   1.1. **Network Expansion**: The Agent Module iteratively expands the network by discovering co-authors using an author-based adaptive snowball sampling method. It can be configured to present the status of exploration periodically, generate previews of coauthorship network, and ask the user to decide the future direction. The user can also pause and step in if necessary.
   1.2. **Data Collection and Processing**: Commanded by the agent module, the Data Collection Modules retrieve, clean, normalize, and combine bibliographic data from the designated sources.
2. **Network Analysis**: Generate and analyze co-authorship networks
3. **Community Detection**: Identify research communities and their relationships
4. **Insight Generation**: Provide visualizations and analytics to support research

## Modules and Progress

### Data Collection Modules (DONE)

These modules handle the collection, pre-processing, and storage of bibliographic data:

#### NDL Search Module
- **`ndl_search_retriever.py`**: Retrieves bibliographic items from the [NDL Search API](https://ndlsearch.ndl.go.jp/help/api/specifications)
  - **Features**:
    - Multiple query modes: by keywords in title, by creator, or by NDC code
    - Customizable time range parameters
    - XML data extraction and JSON conversion

#### CiNii Module
- **`cinii_search_retriever.py`**: Interfaces with CiNii Research's [OpenSearch API](https://support.nii.ac.jp/ja/cir/r_opensearch)
  - **Features**:
    - Parallel functionality to NDL module
    - Standardized output format for consistency

#### Data Processing Module
- **`data_processor.py`**: Contains the `BiblioDataProcessor` class for managing retrieved data
  - **Features**:
    - Data normalization for dates, authors, and publishers
    - Duplication removal
    - SQLite database storage with relational tables
    - Entity identification and categorization

### Agent Module (IN-PROGRESS)

**`agent.py`**: The core intelligent component that orchestrates data collection for the network. It manages the co-authorship network data collection process through the following pipeline:

1. **Initial Configuration**:
  - **Searching parameters**:
    - Required database selection parameter (NDL Search, CiNii, or both)
    - Required person name parameter (full name of initial author). The initial author has a depth of 0.
    - Optional filtering parameters:
      - Keywords in publication titles
      - Publication year range (start_year, end_year)
      - NDC number (Japanese library classification code)
      - Publication types (articles, books, all, etc. default is books)
      - Minimum publication threshold for including authors in expansion (e.g., minimum 2 publications)
  - **Controling parameters**:
    - Required depth parameter (integer) defining maximum network expansion from the initial author
    - Required prioritized_depth parameter designating which author depths to consider for calculating priority value of an author's coauthors:
      - If prioritized_depth is set (not None or ["off"]), the user must also provide an prioritized_depth_weight parameter (value between 0-1), which controls the balance between depth-based priority and overlap-based priority. Default to 0.5
    - Required max_authors parameter defining the maximum number of authors to be searched

2. **Database and Task Continuity Management**:
  - Database Naming Convention:
    - Database filenames follow the pattern: `network_data_[TIMESTAMP].db`
    - The timestamp format is YYYYMMDD_HHMMSS to ensure chronological sorting
  - After start, the program checks for existing database files:
    - Find the databases beginning with "network_data_" (if any) in the current directory
    - If there are one or multiple databases, default to the one with the newest timestamp and ask for user's choice. If not, create a new one
    - The user is prompted with options if there are one or more database file(s):
      - Continue with the default one
      - Select a specific database from a list of available ones
      - Creates a new database file to start a new task
  - When creating a new database for a new task
    - The database's name follows the naming convention
    - Create tables for the Agent Module's use
      - A "network_expansion_metadata" table for storing parameters used in each run: currently-used searching parameters (in json), currently-used controling parameters (in json), executing timestamp
      - A "network_expansion_status" table storing current expansion status: author_id, author_name_cleaned, depth, priority, status, status_timestamp, parent_author
    - Initial search
      - Do an initial search for the initial author using the Data Collection Modules
      - Create a record for the initial author in the "network_expansion_status" table: read author_id from the "cleaned_name" field of the "entities" table, depth=0, priority=1, status="pending", status_timestamp is the current time, parent_author is empty.
    - Start the **Iterative Network Expansion**
  - When continuing a previous task using an existing database:
    - Load all searching and controling parameters from the selected database. If the loaded parameters are different from those given by the user, use the ones given by the user
    - Save a new record of the currently used parameters to the "network_expansion_metadata" table
    - Restart the **Iterative Network Expansion** by resuming from the last "in_progress" author or the highest-priority "pending" author


3. **Iterative Network Expansion**:
  - First check if any termination conditions in section 4 are met. If yes, stop expansion.
  - Select the author with highest priority for expansion from the pending queue (skipping completed ones), change the author's status in the database's "network_expansion_status" table to "in_progress".
  - Use the data collection modules to retrieve and process/clean bibliographic data for the selected author:
    - Use the author's cleaned name, which should be read from the database (the "cleaned_name" field in the "entities" table), for retrieving bibliographic data from the designated source(s) and writing the processed data to the database.
  - Get a list of the current author's coauthors (id and cleaned_name) by querying the database. Don't include the current author's parent author. Only include entities with the type of "person". Print these coauthors.
  - Calculate a **overlap-based_priority**: the percentage of co-authors who has previously appeared in specified depths designated by the prioritized_depth parameter, including both completed and pending ones. The prioritized_depth is intepreted as below:
    - Specific depths: e.g. "0", "1", "1,2", "2,3,4" (consider authors at these exact depths).
    - Relative depths: e.g. "-2" means considering authors from the current and the previous depth levels; "-1" means considering the current depth level.
    - All depths: "all" (consider all authors in the network regardless of depth)
    - No prioritization: prioritized_depth not set or set to "none"; in this case, the overlap-based_priority is 0. 
  - Loop through the coauthors
    - If a coauthor's status in the "network_expansion_status" is "completed", skip it
    - If a coauthor is not in the "network_expansion_status" table, it is a **new author**. 
      - Create a record for it in the "network_expansion_status" table: author_id, cleaned name, depth = current author's depth+1, leave the priority field blank, status="pending", timestamp, parent_author = the id of the current author.
      - Calculate its **depth-based_priority**: If the current author (the coauthor's parent author)'s depth=n, the depth-based_priority of this coauthor is 0.9^(n+1).
      - Calculate its **combined_priority** (0-1): combined_priority = (1 - prioritized_depth_weight) × depth_priority + prioritized_depth_weight × overlap-based_priority. Set the coauthor's priority field to this value.
    - If a coauthor is in the pending queue and has a priority value (**old author**):
      - Calculate its **overlap-based_priority**: same as above
      - Calculate its **depth-based_priority**: use its stored depth to calculate, 0.9^(stored depth)
      - Calculate its **combined_priority**: combined_priority = (1 - prioritized_depth_weight) × depth_priority + prioritized_depth_weight × overlap-based_priority.
      - Read its **previous_priority**: the priority it already has.
      - If the newly calculated combined_priority <= previous priority, the previous priority is preserved as its priority value, else the value is updated to the newly calculated combined_priority.
  - Update the current expansion status and save the status to database:
    - The current author's status (change from "in_progress" to "completed") and priority
    - Update the current author's coauthors' priority:
      - If a coauthor does not exist (new author), it inherite the current author's priority and is added to the pending queue.
      - If a coauthor exists in the pending queue, its new priority will be max(its own old priority, current author's priority)*1.1 
    - Extra information (in a separate table): 
  - Periodic Output Generation:
    - Every 10 authors processed, automatically generate an interim output (See 5)

4. **Termination Conditions** (stops when ANY condition is met):
  - **Empty pending queue** The pending queue reaches 0.
  - **Isolated branch termination**: 
    - When depth >= 1 and prioritized_depth is set and prioritized_depth_weight > 0
      - Check the author with highest priority in the pending queue. If its priority == (1 - prioritized_depth_weight) × 0.9^(current depth), then this termination condition is met.
  - **Depth limit**: Current iteration reaches the user-specified maximum depth parameter
    - Example: If depth=2, search stops after searching and processing co-authors of co-authors of the initial author. In this case, authors at depth=3 will be included in the network but marked as "pending" and not searched
  - **User-defined search limit**: Number of authors searched reaches the max_authors parameter specified in the initial configuration
  - **Manual termination**: User explicitly stops the expansion process. There should be one last output before terminating the program, including the case where the program is terminated with Ctrl+C.

5. **Output**:
  - Generates a tree diagram starting from the initial author by following the "parent_author" field in the "network_expansion_status" table. The output file shoudl be in a standard graph format.


### Social Network Analysis Modules (TODO)
- Creation of co-authorship network, etc.
- Analysis of the co-authorship networks:
  - **Community detection algorithms**
  - **Centrality measures** to identify key researchers
  - **Temporal analysis** of collaboration patterns
  - **Visualization components** for network exploration

### GUI Modules (IN-PROGRESS)
The GUI components:
  - **`biblio_explorer_gui.py`**: Main application interface
  - **`cinii_search_wrapper.py`**: UI wrapper for CiNii search functionality
  - **`ndl_search_wrapper.py`**: UI wrapper for NDL search functionality
  - **`data_processor_wrapper.py`**: UI wrapper for data processing operations
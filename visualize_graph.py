import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import traceback
import plotly.express as px
import json
import math
import argparse
import sqlite3
import glob
from datetime import datetime
from typing import Optional, List, Tuple

class GraphVisualizer:
    def __init__(self, graph_file=None, db_path=None, output_dir="output", top_n_authors=None,
                 min_publications=1, entity_types_in_network=None):
        self.output_dir = output_dir
        self.db_path = db_path
        self.min_publications = min_publications
        self.entity_types_in_network = entity_types_in_network or ["person"]
        self.timestamp = None  # Will store the timestamp from database or GraphML file

        # If db_path is provided but no graph_file, we'll generate the GraphML from the database
        if db_path and not graph_file:
            self.graph_file = self.generate_graphml_from_db()
        else:
            self.graph_file = graph_file or self._find_latest_graphml()

        self.G = None
        self.pos = None
        self.node_depths = {}
        self.node_names = {}
        self.node_statuses = {}
        self.node_priorities = {}
        self.node_colors = []
        self.node_sizes = []
        self.edge_weights = []
        self.top_n_authors = top_n_authors  # Number of top authors to display by priority
        self.filtered_nodes = None  # Will store the filtered nodes when using top_n_authors

        # Set up colors based on depth
        self.depth_colors = {
            0: 'red',       # Root node
            1: 'orange',    # First level
            2: 'green',     # Second level
            3: 'blue',      # Third level
            4: 'purple',    # Fourth level
            5: 'brown'      # Fifth level
        }

        # Set up colors based on status
        self.status_colors = {
            'completed': 'green',
            'pending': 'gray'
        }

    def _find_latest_db(self):
        """Find the most recent database file and return its path and timestamp"""
        existing_dbs = sorted(glob.glob("network_data_*.db"))

        if not existing_dbs:
            print("No database files found.")
            return None

        # Default to the newest database
        newest_db = existing_dbs[-1]
        print(f"Using latest database: {newest_db}")

        # Extract timestamp from the database filename
        # Format is network_data_YYYYMMDD_HHMMSS.db
        try:
            timestamp = newest_db.split('network_data_')[1].split('.db')[0]
            return newest_db, timestamp
        except:
            print(f"Warning: Could not extract timestamp from database filename: {newest_db}")
            return newest_db, None

    def generate_graphml_from_db(self):
        """Generate a GraphML file from the database"""
        # If no db_path provided, find the latest database
        db_timestamp = None
        if not self.db_path:
            result = self._find_latest_db()
            if not result:
                print("No database file found. Cannot generate GraphML.")
                return None
            self.db_path, db_timestamp = result
        else:
            # Try to extract timestamp from provided db_path
            try:
                if "network_data_" in self.db_path:
                    db_timestamp = self.db_path.split('network_data_')[1].split('.db')[0]
            except:
                pass

        # Make sure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Connect to the database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            print(f"Connected to database: {self.db_path}")

            # If we couldn't extract timestamp from filename, try to get it from the database
            if not db_timestamp:
                try:
                    cursor.execute('SELECT timestamp FROM network_expansion_metadata ORDER BY id DESC LIMIT 1')
                    result = cursor.fetchone()
                    if result and result[0]:
                        # Convert ISO format to our timestamp format
                        db_timestamp = datetime.fromisoformat(result[0]).strftime("%Y%m%d_%H%M%S")
                except:
                    # If we can't get the timestamp from the database, use current time
                    db_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    print(f"Could not extract timestamp from database, using current time: {db_timestamp}")

        except Exception as e:
            print(f"ERROR: Failed to connect to database: {e}")
            traceback.print_exc()
            return None

        # Generate GraphML file with the database timestamp
        output_file = os.path.join(self.output_dir, f"coauthor_network_{db_timestamp}.graphml")

        # Check if file already exists
        if os.path.exists(output_file):
            response = input(f"GraphML file {output_file} already exists. Overwrite? (y/n): ").strip().lower()
            if response != 'y':
                print(f"Using existing GraphML file: {output_file}")
                conn.close()
                return output_file

        try:
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
            cursor.execute(
                'SELECT author_id, author_name_cleaned, depth, status, priority FROM network_expansion_status'
            )

            for author_id, name, depth, status, priority in cursor.fetchall():
                graphml += f'    <node id="{author_id}">\n'
                graphml += f'      <data key="d0">{name}</data>\n'
                graphml += f'      <data key="d1">{depth}</data>\n'
                graphml += f'      <data key="d2">{status}</data>\n'
                graphml += f'      <data key="d3">{priority}</data>\n'
                graphml += '    </node>\n'

            # Add edges (co-authorship relationships)
            # This query finds pairs of authors who have collaborated on at least one publication,
            # are both in the allowed entity types, and are both in the network_expansion_status table

            # Create placeholders for the IN clause
            placeholders = ','.join(['?'] * len(self.entity_types_in_network))

            query = f'''
                SELECT e1.entity_id AS author1, e2.entity_id AS author2, COUNT(*) AS weight
                FROM item_entities e1
                JOIN item_entities e2 ON e1.item_id = e2.item_id
                JOIN entities ent1 ON e1.entity_id = ent1.id
                JOIN entities ent2 ON e2.entity_id = ent2.id
                JOIN network_expansion_status nes1 ON e1.entity_id = nes1.author_id
                JOIN network_expansion_status nes2 ON e2.entity_id = nes2.author_id
                WHERE e1.relationship_type = 'author' AND e2.relationship_type = 'author'
                AND e1.entity_id < e2.entity_id  -- Avoid duplicate edges
                AND ent1.entity_type IN ({placeholders}) AND ent2.entity_type IN ({placeholders})
                GROUP BY e1.entity_id, e2.entity_id
                HAVING COUNT(*) >= ?
            '''

            cursor.execute(query, self.entity_types_in_network + self.entity_types_in_network + [self.min_publications])

            for author1, author2, weight in cursor.fetchall():
                graphml += f'    <edge source="{author1}" target="{author2}">\n'
                graphml += f'      <data key="d4">{weight}</data>\n'
                graphml += '    </edge>\n'

            # Close graph and GraphML
            graphml += '  </graph>\n'
            graphml += '</graphml>\n'

            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(graphml)

            print(f"Network saved as GraphML: {output_file}")
            conn.close()
            return output_file

        except Exception as e:
            print(f"ERROR: Failed to generate GraphML file: {e}")
            traceback.print_exc()
            conn.close()
            return None

    def _find_latest_graphml(self):
        """Find the most recent GraphML file in the output directory and return its path and timestamp"""
        if not os.path.exists(self.output_dir):
            print(f"Output directory '{self.output_dir}' not found.")
            return None

        graphml_files = [f for f in os.listdir(self.output_dir)
                         if f.startswith("coauthor_network_") and f.endswith(".graphml")]

        if not graphml_files:
            print(f"No GraphML files found in '{self.output_dir}'.")
            return None

        # Sort by filename (which includes timestamp)
        latest_file = sorted(graphml_files)[-1]
        print(f"Using latest GraphML file: {latest_file}")

        # Extract timestamp from the filename
        # Format is coauthor_network_YYYYMMDD_HHMMSS.graphml
        try:
            timestamp = latest_file.split('coauthor_network_')[1].split('.graphml')[0]
            return os.path.join(self.output_dir, latest_file), timestamp
        except:
            print(f"Warning: Could not extract timestamp from GraphML filename: {latest_file}")
            return os.path.join(self.output_dir, latest_file), None

    def load_graph(self):
        # Check if the graph_file is a tuple (path, timestamp) from _find_latest_graphml
        if isinstance(self.graph_file, tuple):
            self.graph_file, self.timestamp = self.graph_file
        else:
            # Try to extract timestamp from the filename if it's a string
            try:
                if self.graph_file and "coauthor_network_" in self.graph_file:
                    self.timestamp = self.graph_file.split('coauthor_network_')[1].split('.graphml')[0]
                else:
                    self.timestamp = None
            except:
                self.timestamp = None

        # Check if the file exists
        if not os.path.exists(self.graph_file):
            print(f"ERROR: File '{self.graph_file}' not found in the current directory.")
            print(f"Current working directory: {os.getcwd()}")
            return False

        try:
            self.G = nx.read_graphml(self.graph_file)
            print(f"Success! Graph has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
            return True
        except Exception as e:
            print(f"ERROR: Failed to read GraphML file: {e}")
            traceback.print_exc()
            return False

    def extract_attributes(self):
        # Extract node attributes for visualization
        print("Extracting node attributes...")
        self.node_depths = nx.get_node_attributes(self.G, 'depth')
        self.node_names = nx.get_node_attributes(self.G, 'name')
        self.node_statuses = nx.get_node_attributes(self.G, 'status')
        self.node_priorities = nx.get_node_attributes(self.G, 'priority')

        print(f"Found {len(self.node_depths)} nodes with depth attribute")
        print(f"Found {len(self.node_names)} nodes with name attribute")
        print(f"Found {len(self.node_statuses)} nodes with status attribute")
        print(f"Found {len(self.node_priorities)} nodes with priority attribute")

        # Filter nodes by priority if top_n_authors is specified
        if self.top_n_authors is not None:
            self._filter_top_n_nodes()

    def _filter_top_n_nodes(self):
        """Filter nodes to keep only the top N authors by priority"""
        if not self.node_priorities:
            print("Warning: No priority attributes found, cannot filter top authors")
            return

        # Create a list of (node_id, priority) tuples
        node_priority_pairs = [(node, float(priority)) for node, priority in self.node_priorities.items()]

        # Sort by priority in descending order
        sorted_nodes = sorted(node_priority_pairs, key=lambda x: x[1], reverse=True)

        # Get the priority value at the Nth position
        if len(sorted_nodes) > self.top_n_authors:
            cutoff_priority = sorted_nodes[self.top_n_authors - 1][1]

            # Include all nodes with priority equal to the cutoff (to handle ties)
            filtered_nodes = [node for node, priority in sorted_nodes
                             if priority >= cutoff_priority]

            print(f"Filtered to top {len(filtered_nodes)} authors (requested {self.top_n_authors}, "
                  f"included {len(filtered_nodes) - self.top_n_authors} additional authors with same priority)")
        else:
            filtered_nodes = [node for node, _ in sorted_nodes]
            print(f"Keeping all {len(filtered_nodes)} authors as requested top_n ({self.top_n_authors}) exceeds total count")

        # Store the filtered node IDs
        self.filtered_nodes = set(filtered_nodes)

    def calculate_layout(self):
        # Calculate positions using spring layout with seed for reproducibility
        # Adjust k for more spread, iterations for more stability
        print("Calculating layout (this may take a while for large graphs)...")
        self.pos = nx.spring_layout(self.G, k=0.15, iterations=100, seed=42)

    def prepare_visualization(self):
        # Calculate node sizes based on connections with non-linear scaling
        print("Calculating node sizes based on connections...")
        self.node_sizes = []

        # Get all connection counts to find the maximum
        connection_counts = [len(list(self.G.neighbors(node))) for node in self.G.nodes()]
        max_connections = max(connection_counts) if connection_counts else 1
        min_size = 3  # Minimum node size
        max_size = 20  # Maximum node size

        for node in self.G.nodes():
            connections = len(list(self.G.neighbors(node)))
            # Use logarithmic scaling to create non-linear relationship
            # log(x+1) ensures nodes with 0 connections still have size
            if max_connections > 1:
                # Scale between min_size and max_size using log scale
                size = min_size + (max_size - min_size) * (math.log1p(connections) / math.log1p(max_connections))
            else:
                size = min_size
            self.node_sizes.append(size)

        # Calculate edge weights for thickness
        print("Calculating edge weights...")
        self.edge_weights = []
        for u, v in self.G.edges():
            # Get the weight, default to 1 if not specified
            weight = self.G.edges[u, v].get('weight', 1)
            self.edge_weights.append(weight)

        # Get colors based on depth for each node
        print("Assigning colors based on node depth...")
        self.node_colors = []
        for node in self.G.nodes():
            try:
                depth = int(self.node_depths.get(node, 0))
                self.node_colors.append(self.depth_colors.get(depth, 'gray'))
            except Exception as e:
                print(f"Warning: Error processing depth for node {node}: {e}")
                self.node_colors.append('gray')

    def create_node_trace(self, color_by_priority=False):
        # Create nodes trace for plotly
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_hover_text = []
        node_priorities = []  # Store priority values for each node
        included_nodes = []   # Store the actual nodes that are included

        # Determine which nodes to include
        nodes_to_include = self.filtered_nodes if self.filtered_nodes is not None else self.G.nodes()

        # Track the indices of included nodes for color and size mapping
        included_indices = []

        for i, node in enumerate(self.G.nodes()):
            if node in nodes_to_include:
                x, y = self.pos[node]
                node_x.append(x)
                node_y.append(y)
                included_nodes.append(node)

                # Get priority value for this node
                try:
                    priority_value = float(self.node_priorities.get(node, 0))
                except (ValueError, TypeError):
                    priority_value = 0
                node_priorities.append(priority_value)

                # Add the corresponding color and size
                if not color_by_priority:
                    # Use depth-based coloring
                    node_colors.append(self.node_colors[i])
                # We'll set priority-based colors later if needed

                node_sizes.append(self.node_sizes[i])
                included_indices.append(i)

                # Create hover text for this node
                name = self.node_names.get(node, "Unknown")
                depth = self.node_depths.get(node, "Unknown")
                status = self.node_statuses.get(node, "Unknown")
                priority = self.node_priorities.get(node, "Unknown")
                connections = len(list(self.G.neighbors(node)))

                hover_text = f"<b>Name:</b> {name}<br>" + \
                             f"<b>ID:</b> {node}<br>" + \
                             f"<b>Depth:</b> {depth}<br>" + \
                             f"<b>Status:</b> {status}<br>" + \
                             f"<b>Priority:</b> {priority}<br>" + \
                             f"<b>Connections:</b> {connections}"

                node_hover_text.append(hover_text)

        # If coloring by priority, generate the color scale
        if color_by_priority:
            node_colors = self.create_priority_colors(included_nodes)

        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_hover_text,
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='#FFFFFF')
            ),
            # Store the priority values and node IDs as custom data for use in callbacks
            customdata=list(zip(node_priorities, included_nodes))
        )

        return node_trace

    def create_edge_trace(self):
        # Create edges trace for plotly
        edge_traces = []

        # Determine which nodes to include
        nodes_to_include = self.filtered_nodes if self.filtered_nodes is not None else self.G.nodes()

        # Create separate traces for edges with different weights
        for i, (u, v) in enumerate(self.G.edges()):
            # Only include edges where both nodes are in the filtered set
            if self.filtered_nodes is not None and (u not in nodes_to_include or v not in nodes_to_include):
                continue

            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]

            weight = self.edge_weights[i]

            # Create individual edge trace
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=weight/2, color='#888'),
                hoverinfo='text',
                text=f"Weight: {weight}<br>{self.node_names.get(u, '')} — {self.node_names.get(v, '')}",
                mode='lines'
            )

            edge_traces.append(edge_trace)

        return edge_traces

    def create_depth_legend(self):
        # Create a custom legend for depth colors
        legend_traces = []

        for depth, color in self.depth_colors.items():
            legend_trace = go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                showlegend=True,
                name=f'Depth {depth}'
            )
            legend_traces.append(legend_trace)

        return legend_traces

    def create_priority_colors(self, nodes_to_include=None):
        """Create a color scale based on priority values (red gradient)"""
        if nodes_to_include is None:
            nodes_to_include = self.G.nodes()

        # Get priority values for included nodes
        priority_values = []
        for node in self.G.nodes():
            if node in nodes_to_include:
                try:
                    priority = float(self.node_priorities.get(node, 0))
                    priority_values.append(priority)
                except (ValueError, TypeError):
                    priority_values.append(0)

        # If no valid priorities, return default color
        if not priority_values:
            return ['rgba(255, 0, 0, 0.5)'] * len(nodes_to_include)

        # Get min and max for scaling
        min_priority = min(priority_values)
        max_priority = max(priority_values)

        # If all priorities are the same, return mid-intensity red
        if min_priority == max_priority:
            return ['rgba(255, 0, 0, 0.5)'] * len(priority_values)

        # Create color scale from light red to dark red
        colors = []
        for priority in priority_values:
            # Normalize priority to 0-1 range
            if max_priority > min_priority:
                normalized = (priority - min_priority) / (max_priority - min_priority)
            else:
                normalized = 0.5

            # Create red color with intensity based on priority (higher priority = darker red)
            # Using a range from light red (255, 200, 200) to dark red (180, 0, 0)
            r = int(255 - (normalized * 75))  # 255 to 180
            g = int(200 - (normalized * 200))  # 200 to 0
            b = int(200 - (normalized * 200))  # 200 to 0

            colors.append(f'rgb({r}, {g}, {b})')

        return colors

    def create_priority_legend(self):
        """Create a legend for priority colors"""
        legend_traces = []

        # Create 5 legend entries from low to high priority
        priority_levels = [
            {"value": "Lowest", "color": "rgb(255, 200, 200)"},
            {"value": "Low", "color": "rgb(255, 150, 150)"},
            {"value": "Medium", "color": "rgb(255, 100, 100)"},
            {"value": "High", "color": "rgb(255, 50, 50)"},
            {"value": "Highest", "color": "rgb(180, 0, 0)"}
        ]

        for level in priority_levels:
            legend_trace = go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=level["color"]),
                showlegend=True,
                name=f'Priority: {level["value"]}'
            )
            legend_traces.append(legend_trace)

        return legend_traces

    def generate_community_research_themes(self, max_authors_per_community=20, max_publications_per_author=3):
        """
        Generate research theme summaries for each community using GPT.

        Args:
            max_authors_per_community: Maximum number of authors to include in the analysis for large communities
            max_publications_per_author: Maximum number of publications to include per author

        Returns:
            Path to the generated markdown report file
        """
        try:
            from gpt import GPTResponse
            import sqlite3

            # Check if we have community information
            if 'community_members' not in self.G.graph:
                print("No community information found. Run detect_communities first.")
                return None

            # Check if we have a database connection to get publication data
            if not self.db_path:
                print("No database connection available. Cannot retrieve publication data.")
                return None

            # Connect to the database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Initialize GPT
            gpt = GPTResponse()

            # Create markdown report
            timestamp = self.timestamp if hasattr(self, 'timestamp') and self.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
            method = self.G.graph.get('community_method', 'unknown')
            report_file = os.path.join(self.output_dir, f"community_research_themes_{method}_{timestamp}.md")

            # Start the report
            report = f"# Research Themes Analysis - {method.capitalize()} Communities\n\n"
            report += f"Analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += f"Total communities detected: {self.G.graph.get('community_count', 0)}\n\n"

            # Process each community
            community_members = self.G.graph['community_members']

            for community_id, members in community_members.items():
                print(f"Processing community {community_id} with {len(members)} members...")

                # For large communities, focus on central nodes
                if len(members) > max_authors_per_community:
                    print(f"Large community detected ({len(members)} members). Focusing on central nodes...")

                    # Calculate centrality for each node in this community
                    subgraph = self.G.subgraph(members)
                    centrality = nx.degree_centrality(subgraph)

                    # Sort members by centrality
                    central_members = sorted([(node, centrality[node]) for node in members],
                                           key=lambda x: x[1], reverse=True)[:max_authors_per_community]

                    # Extract just the node IDs
                    selected_members = [node for node, _ in central_members]
                    print(f"Selected {len(selected_members)} central members for analysis")
                else:
                    selected_members = members

                # Get publication data for each member
                community_publications = []
                author_names = []

                for author_id in selected_members:
                    # Get author name
                    author_name = self.node_names.get(author_id, f"Author {author_id}")
                    author_names.append(author_name)

                    # Get publications for this author
                    cursor.execute('''
                        SELECT bi.title, bi.publication_date
                        FROM bibliographic_items bi
                        JOIN item_entities ie ON bi.id = ie.item_id
                        WHERE ie.entity_id = ? AND ie.relationship_type = 'author'
                        ORDER BY bi.publication_date DESC
                        LIMIT ?
                    ''', (author_id, max_publications_per_author))

                    publications = cursor.fetchall()

                    for title, publication_date in publications:
                        if title:  # Only include if title is not empty
                            # Extract year from publication_date (format could be YYYY-MM-DD or YYYY-MM or just YYYY)
                            year = publication_date[:4] if publication_date and len(publication_date) >= 4 else "Unknown"
                            publication_entry = f"{title} ({year})"
                            # Only add if not already in the list (avoid duplications)
                            if publication_entry not in community_publications:
                                community_publications.append(publication_entry)

                # Create a prompt for GPT
                prompt = f"""Analyze the following publications from a research community and identify the main research theme(s).

                Publications:
                {chr(10).join([f"- {pub}" for pub in community_publications])}

                Authors in this community:
                {', '.join(author_names)}

                Please provide:
                1. A concise name for this research community (1-5 words)
                2. A summary of the main research themes (2-3 paragraphs)
                3. Key research topics or methodologies (bullet points)

                Format your response as a JSON object with the following structure:
                {{
                    "community_name": "Name of the research community",
                    "theme_summary": "Summary of the research themes...",
                    "key_topics": ["Topic 1", "Topic 2", "Topic 3"]
                }}
                """

                # Define the schema for structured output
                schema = {
                    "type": "object",
                    "properties": {
                        "community_name": {"type": "string"},
                        "theme_summary": {"type": "string"},
                        "key_topics": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["community_name", "theme_summary", "key_topics"],
                    "additionalProperties": False
                }

                # Get response from GPT
                print(f"Requesting LLM summarization for community {community_id}...")
                response = gpt.get_response(prompt, schema)

                if response:
                    # Add to the report
                    report += f"## Community {community_id}: {response['community_name']}\n\n"
                    report += f"### Research Theme Summary\n\n{response['theme_summary']}\n\n"

                    report += "### Key Research Topics\n\n"
                    for topic in response['key_topics']:
                        report += f"- {topic}\n"
                    report += "\n"

                    report += "### Community Members\n\n"
                    for i, author_id in enumerate(selected_members):
                        author_name = self.node_names.get(author_id, f"Author {author_id}")
                        report += f"- {author_name}\n"

                    if len(members) > len(selected_members):
                        report += f"\n*Note: Only showing {len(selected_members)} out of {len(members)} total members.*\n"

                    report += "\n### Representative Publications\n\n"
                    for pub in community_publications[:10]:  # Limit to 10 publications in the report
                        report += f"- {pub}\n"

                    if len(community_publications) > 10:
                        report += f"\n*Note: Only showing 10 out of {len(community_publications)} publications.*\n"

                    report += "\n---\n\n"
                else:
                    report += f"## Community {community_id}\n\n"
                    report += "*Error: Could not generate theme analysis for this community.*\n\n"
                    report += "---\n\n"

            # Close database connection
            conn.close()

            # Write the report to file
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

            print(f"Research themes report saved to: {report_file}")
            return report_file

        except ImportError as e:
            print(f"Error: Required module not found: {e}")
            print("Make sure the gpt.py module is available and OpenAI package is installed.")
            return None
        except Exception as e:
            print(f"Error generating research themes: {e}")
            traceback.print_exc()
            return None

    def draw_graph(self):
        try:
            # Store the current filtered_nodes state
            original_filtered_nodes = self.filtered_nodes

            # Create traces with current filter settings - default to depth-based coloring
            node_trace_depth = self.create_node_trace(color_by_priority=False)
            node_trace_priority = self.create_node_trace(color_by_priority=True)
            edge_traces = self.create_edge_trace()
            depth_legend_traces = self.create_depth_legend()
            priority_legend_traces = self.create_priority_legend()

            # Initially show depth-based coloring and hide priority-based coloring
            node_trace_depth.visible = True
            node_trace_priority.visible = False

            # Initially show depth legend and hide priority legend
            for trace in depth_legend_traces:
                trace.visible = True
            for trace in priority_legend_traces:
                trace.visible = False

            # If we have top_n_authors set, we need to create traces for both filtered and unfiltered views
            has_top_n_filter = self.top_n_authors is not None

            # Create the figure
            fig = go.Figure(
                data=edge_traces + [node_trace_depth, node_trace_priority] + depth_legend_traces + priority_legend_traces,
                layout=go.Layout(
                    title=dict(text='Co-author Network Visualization', font=dict(size=16)),
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    ),
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="left",
                            buttons=[
                                dict(
                                    args=[{"visible": [True] * len(edge_traces) + [True] + [False] +
                                                     [True] * len(depth_legend_traces) + [False] * len(priority_legend_traces)}],
                                    label="Show All Connections",
                                    method="update"
                                ),
                                dict(
                                    args=[{
                                        "visible": [
                                            i < len(edge_traces) and self.edge_weights[i % len(self.edge_weights)] > 1
                                            for i in range(len(edge_traces))
                                        ] + [True] + [False] +
                                          [True] * len(depth_legend_traces) + [False] * len(priority_legend_traces)
                                    }],
                                    label="Strong Connections Only",
                                    method="update"
                                )
                            ],
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0.1,
                            xanchor="left",
                            y=1.1,
                            yanchor="top"
                        ),
                        # Add a button to toggle between depth and priority coloring
                        dict(
                            type="buttons",
                            direction="left",
                            buttons=[
                                dict(
                                    args=[{
                                        "visible": [True] * len(edge_traces) + [True, False] +
                                                  [True] * len(depth_legend_traces) + [False] * len(priority_legend_traces)
                                    }],
                                    label="Color by Depth",
                                    method="update"
                                ),
                                dict(
                                    args=[{
                                        "visible": [True] * len(edge_traces) + [False, True] +
                                                  [False] * len(depth_legend_traces) + [True] * len(priority_legend_traces)
                                    }],
                                    label="Color by Priority",
                                    method="update"
                                )
                            ],
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0.3,  # Position to the right of the first button
                            xanchor="left",
                            y=1.1,
                            yanchor="top"
                        )
                    ],
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                )
            )

            # If we have a top_n filter, add a button to toggle between all authors and top N authors
            if has_top_n_filter:
                # Create a button menu for filtering by top authors
                top_n_button = dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=[{"title": "Co-author Network Visualization (All Authors)"}],
                            label=f"Show All Authors",
                            method="update"
                        ),
                        dict(
                            args=[{"title": f"Co-author Network Visualization (Top {len(self.filtered_nodes)} Authors by Priority)"}],
                            label=f"Show Top {self.top_n_authors} Authors",
                            method="update"
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.5,  # Position in the middle
                    xanchor="center",
                    y=1.1,
                    yanchor="top"
                )

                # Add the button to the updatemenus list
                # Convert tuple to list, add new button, then update layout
                current_menus = list(fig.layout.updatemenus)
                current_menus.append(top_n_button)
                fig.update_layout(updatemenus=current_menus)

                # Set the initial title to reflect the current view
                if original_filtered_nodes is not None:
                    fig.update_layout(title=f"Co-author Network Visualization (Top {len(self.filtered_nodes)} Authors by Priority)")

            # Add hover interactions
            fig.update_layout(clickmode='event+select')

            # Save as interactive HTML in the output directory
            # Use the timestamp from the GraphML file if available, otherwise use current time
            timestamp = self.timestamp if hasattr(self, 'timestamp') and self.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"coauthor_network_{timestamp}.html")

            # Check if file already exists
            if os.path.exists(output_file):
                response = input(f"Interactive HTML file {output_file} already exists. Overwrite? (y/n): ").strip().lower()
                if response != 'y':
                    print(f"Using existing interactive HTML file: {output_file}")
                else:
                    fig.write_html(output_file, include_plotlyjs=True, full_html=True)
                    print(f"Interactive visualization saved to {output_file}")
            else:
                fig.write_html(output_file, include_plotlyjs=True, full_html=True)
                print(f"Interactive visualization saved to {output_file}")

            # Show the plot in browser
            fig.show()

            return True

        except Exception as e:
            print(f"ERROR: Failed to draw the graph: {e}")
            traceback.print_exc()
            return False

    def detect_communities(self, method="louvain"):
        """
        Detect communities in the graph using either Louvain or Leiden method.

        Args:
            method: The community detection method to use ("louvain" or "leiden")

        Returns:
            Boolean indicating if community detection was successful
        """
        if method.lower() == "leiden":
            return self.detect_communities_leiden()
        else:
            return self.detect_communities_louvain()

    def detect_communities_louvain(self):
        """Detect communities using the Louvain method"""
        try:
            from community import community_louvain

            # Find communities
            print("Finding communities using Louvain method...")
            partition = community_louvain.best_partition(self.G)
            print(f"Found {len(set(partition.values()))} communities")

            # Prepare node data
            node_x = []
            node_y = []
            node_colors = []
            node_sizes = []
            node_texts = []
            community_count = len(set(partition.values()))

            # Create a colorscale for communities
            colorscale = px.colors.qualitative.Plotly
            if community_count > len(colorscale):
                colorscale = px.colors.qualitative.Light24

            for node in self.G.nodes():
                x, y = self.pos[node]
                node_x.append(x)
                node_y.append(y)

                # Get community and assign color
                community_id = partition.get(node, 0)
                color_idx = community_id % len(colorscale)
                node_colors.append(colorscale[color_idx])

                # Size based on connections with non-linear scaling
                connections = len(list(self.G.neighbors(node)))

                # Get statistics for scaling if not already calculated
                if 'max_connections' not in locals():
                    connection_counts = [len(list(self.G.neighbors(n))) for n in self.G.nodes()]
                    max_connections = max(connection_counts) if connection_counts else 1
                    min_size = 3  # Minimum node size
                    max_size = 20  # Maximum node size

                # Use logarithmic scaling
                if max_connections > 1:
                    size = min_size + (max_size - min_size) * (math.log1p(connections) / math.log1p(max_connections))
                else:
                    size = min_size

                node_sizes.append(size)

                # Hover text
                name = self.node_names.get(node, "Unknown")
                depth = self.node_depths.get(node, "Unknown")
                status = self.node_statuses.get(node, "Unknown")
                community = partition.get(node, "Unknown")

                hover_text = f"<b>Name:</b> {name}<br>" + \
                             f"<b>Community:</b> {community}<br>" + \
                             f"<b>Depth:</b> {depth}<br>" + \
                             f"<b>Status:</b> {status}<br>" + \
                             f"<b>Connections:</b> {connections}"

                node_texts.append(hover_text)

            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_texts,
                marker=dict(
                    showscale=False,
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=2, color='#FFFFFF')
                )
            )

            # Create edges trace
            edge_traces = self.create_edge_trace()

            # Create community legend
            legend_traces = []
            for comm_id in set(partition.values()):
                color_idx = comm_id % len(colorscale)
                legend_trace = go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=10, color=colorscale[color_idx]),
                    showlegend=True,
                    name=f'Community {comm_id}'
                )
                legend_traces.append(legend_trace)

            # Create the figure
            fig = go.Figure(
                data=edge_traces + [node_trace] + legend_traces,
                layout=go.Layout(
                    title=dict(text='Co-author Network Communities', font=dict(size=16)),
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    ),
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                )
            )

            # Save as interactive HTML in the output directory
            # Use the timestamp from the GraphML file if available, otherwise use current time
            timestamp = self.timestamp if hasattr(self, 'timestamp') and self.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
            community_file = os.path.join(self.output_dir, f"coauthor_network_communities_{timestamp}.html")

            # Check if file already exists
            if os.path.exists(community_file):
                response = input(f"Community HTML file {community_file} already exists. Overwrite? (y/n): ").strip().lower()
                if response != 'y':
                    print(f"Using existing community HTML file: {community_file}")
                else:
                    fig.write_html(community_file, include_plotlyjs=True, full_html=True)
                    print(f"Community visualization saved to {community_file}")
            else:
                fig.write_html(community_file, include_plotlyjs=True, full_html=True)
                print(f"Community visualization saved to {community_file}")

            # Show the plot in browser
            fig.show()

            return True

        except ImportError:
            print("python-louvain package not installed. Skipping Louvain community detection.")
            print("To enable Louvain community detection, install: pip install python-louvain")
            return False
        except Exception as e:
            print(f"WARNING: Failed to perform Louvain community detection: {e}")
            traceback.print_exc()
            return False

    def detect_communities_leiden(self):
        """Detect communities using the Leiden method"""
        try:
            import leidenalg
            import igraph as ig

            # Convert NetworkX graph to igraph
            print("Converting NetworkX graph to igraph format...")
            # Create a mapping of node IDs to indices
            node_mapping = {node: i for i, node in enumerate(self.G.nodes())}

            # Create igraph Graph
            g = ig.Graph()
            g.add_vertices(len(self.G.nodes()))

            # Add edges with weights
            edges = []
            weights = []
            for u, v, data in self.G.edges(data=True):
                edges.append((node_mapping[u], node_mapping[v]))
                # Use weight if available, otherwise default to 1
                weights.append(data.get('weight', 1))

            g.add_edges(edges)
            g.es['weight'] = weights

            # Find communities using Leiden algorithm
            print("Finding communities using Leiden method...")
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights='weight'
            )

            # Convert partition back to NetworkX node IDs
            nx_partition = {}
            community_members = {}  # Store members of each community

            for i, community in enumerate(partition):
                community_members[i] = []  # Initialize list for this community
                for vertex in community:
                    # Get the original node ID from our mapping
                    original_node = list(node_mapping.keys())[list(node_mapping.values()).index(vertex)]
                    nx_partition[original_node] = i
                    community_members[i].append(original_node)

            # Store community information in the graph
            nx.set_node_attributes(self.G, nx_partition, 'community')

            # Store community members dictionary as a graph attribute
            self.G.graph['community_members'] = community_members
            self.G.graph['community_method'] = 'leiden'
            self.G.graph['community_count'] = len(partition)

            print(f"Found {len(partition)} communities")

            # Prepare node data
            node_x = []
            node_y = []
            node_colors = []
            node_sizes = []
            node_texts = []
            community_count = len(partition)

            # Create a colorscale for communities
            colorscale = px.colors.qualitative.Plotly
            if community_count > len(colorscale):
                colorscale = px.colors.qualitative.Light24

            for node in self.G.nodes():
                x, y = self.pos[node]
                node_x.append(x)
                node_y.append(y)

                # Get community and assign color
                community_id = nx_partition.get(node, 0)
                color_idx = community_id % len(colorscale)
                node_colors.append(colorscale[color_idx])

                # Size based on connections with non-linear scaling
                connections = len(list(self.G.neighbors(node)))

                # Get statistics for scaling if not already calculated
                if 'max_connections' not in locals():
                    connection_counts = [len(list(self.G.neighbors(n))) for n in self.G.nodes()]
                    max_connections = max(connection_counts) if connection_counts else 1
                    min_size = 3  # Minimum node size
                    max_size = 20  # Maximum node size

                # Use logarithmic scaling
                if max_connections > 1:
                    size = min_size + (max_size - min_size) * (math.log1p(connections) / math.log1p(max_connections))
                else:
                    size = min_size

                node_sizes.append(size)

                # Hover text
                name = self.node_names.get(node, "Unknown")
                depth = self.node_depths.get(node, "Unknown")
                status = self.node_statuses.get(node, "Unknown")
                community = nx_partition.get(node, "Unknown")

                hover_text = f"<b>Name:</b> {name}<br>" + \
                             f"<b>Community:</b> {community}<br>" + \
                             f"<b>Depth:</b> {depth}<br>" + \
                             f"<b>Status:</b> {status}<br>" + \
                             f"<b>Connections:</b> {connections}"

                node_texts.append(hover_text)

            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_texts,
                marker=dict(
                    showscale=False,
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=2, color='#FFFFFF')
                )
            )

            # Create edges trace
            edge_traces = self.create_edge_trace()

            # Create community legend
            legend_traces = []
            for comm_id in range(community_count):
                color_idx = comm_id % len(colorscale)
                legend_trace = go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(size=10, color=colorscale[color_idx]),
                    showlegend=True,
                    name=f'Community {comm_id}'
                )
                legend_traces.append(legend_trace)

            # Create the figure
            fig = go.Figure(
                data=edge_traces + [node_trace] + legend_traces,
                layout=go.Layout(
                    title=dict(text='Co-author Network Communities (Leiden)', font=dict(size=16)),
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    ),
                    plot_bgcolor='rgba(255,255,255,1)',
                    paper_bgcolor='rgba(255,255,255,1)',
                )
            )

            # Save as interactive HTML in the output directory
            # Use the timestamp from the GraphML file if available, otherwise use current time
            timestamp = self.timestamp if hasattr(self, 'timestamp') and self.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
            community_file = os.path.join(self.output_dir, f"coauthor_network_communities_leiden_{timestamp}.html")

            # Check if file already exists
            if os.path.exists(community_file):
                response = input(f"Leiden community HTML file {community_file} already exists. Overwrite? (y/n): ").strip().lower()
                if response != 'y':
                    print(f"Using existing Leiden community HTML file: {community_file}")
                else:
                    fig.write_html(community_file, include_plotlyjs=True, full_html=True)
                    print(f"Leiden community visualization saved to {community_file}")
            else:
                fig.write_html(community_file, include_plotlyjs=True, full_html=True)
                print(f"Leiden community visualization saved to {community_file}")

            # Show the plot in browser
            fig.show()

            return True

        except ImportError:
            print("leidenalg and/or igraph packages not installed. Skipping Leiden community detection.")
            print("To enable Leiden community detection, install: pip install leidenalg python-igraph")
            return False
        except Exception as e:
            print(f"WARNING: Failed to perform Leiden community detection: {e}")
            traceback.print_exc()
            return False

def main():
    try:
        # Set up command line argument parsing
        parser = argparse.ArgumentParser(description='Visualize co-author network graph')
        parser.add_argument('--top', type=int, help='Show only the top N authors by priority')
        parser.add_argument('--graph', type=str, help='Path to the GraphML file to visualize')
        parser.add_argument('--db', type=str, help='Path to the database file to generate GraphML from')
        parser.add_argument('--output', type=str, default='output',
                           help='Output directory for visualization files')
        parser.add_argument('--min-publications', type=int, default=1,
                           help='Minimum number of publications for co-authorship edge (default: 1)')
        parser.add_argument('--entity-types', type=str, default='person',
                           help='Comma-separated list of entity types to include in the network (default: person)')
        parser.add_argument('--community-method', type=str, choices=['louvain', 'leiden'], default='louvain',
                           help='Community detection method to use (default: louvain)')
        parser.add_argument('--generate-themes', action='store_true',
                           help='Generate research theme summaries for communities using LLM')
        parser.add_argument('--max-authors-per-community', type=int, default=20,
                           help='Maximum number of authors to analyze per community (default: 20)')
        parser.add_argument('--max-publications-per-author', type=int, default=3,
                           help='Maximum number of publications to analyze per author (default: 3)')

        args = parser.parse_args()

        print("Starting co-author network visualization script...")

        # Parse entity types if provided
        entity_types = args.entity_types.split(',') if args.entity_types else ["person"]

        # Initialize visualizer with command line arguments
        visualizer = GraphVisualizer(
            graph_file=args.graph,
            db_path=args.db,
            output_dir=args.output,
            top_n_authors=args.top,
            min_publications=args.min_publications,
            entity_types_in_network=entity_types
        )

        # Check if a graph file was found or generated
        if not visualizer.graph_file:
            print("No GraphML file found or generated. Please specify a graph file path or database path.")
            return 1

        # Load and prepare the graph
        if not visualizer.load_graph():
            return 1

        visualizer.extract_attributes()
        visualizer.calculate_layout()
        visualizer.prepare_visualization()

        # Draw the interactive graph
        if not visualizer.draw_graph():
            return 1

        # Attempt community detection visualization
        print(f"Attempting community detection visualization using {args.community_method} method...")
        community_result = visualizer.detect_communities(method=args.community_method)

        # Generate research theme summaries if requested and community detection was successful
        if args.generate_themes and community_result:
            print("Generating research theme summaries for communities...")
            report_file = visualizer.generate_community_research_themes(
                max_authors_per_community=args.max_authors_per_community,
                max_publications_per_author=args.max_publications_per_author
            )
            if report_file:
                print(f"Research theme report generated: {report_file}")
            else:
                print("Failed to generate research theme report.")

        print("Visualization script completed successfully!")
        return 0

    except Exception as e:
        print(f"ERROR: Unhandled exception: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

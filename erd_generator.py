import streamlit as st
import yaml as pyyaml  # Aliasing for clarity
import pandas as pd
import os
import traceback
import io
import base64
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Snowflake Semantic Model ERD", layout="wide")

def load_yaml_file(file_path=None, file_content=None):
    """Load YAML data from either a file path or uploaded file content"""
    try:
        if file_content is not None:
            # For uploaded file
            content = file_content.decode('utf-8')
            return pyyaml.safe_load(content)
        elif file_path and os.path.exists(file_path):
            # For file path
            with open(file_path, 'r') as file:
                return pyyaml.safe_load(file)
        else:
            return None
    except Exception as e:
        st.error(f"Error parsing YAML file: {str(e)}")
        return None

def standardize_table_names(semantic_model):
    """Standardize all table names to uppercase throughout the model"""
    # Create a mapping of original table names to uppercase versions
    tables = semantic_model.get('tables', [])
    table_name_map = {}
    
    # First, create a mapping of original names to uppercase
    for table in tables:
        if 'name' in table:
            original_name = table['name']
            uppercase_name = original_name.upper()
            table_name_map[original_name] = uppercase_name
            table['name'] = uppercase_name
    
    # Update relationships to use uppercase table names
    for relationship in semantic_model.get('relationships', []):
        if 'left_table' in relationship:
            relationship['left_table'] = relationship['left_table'].upper()
        if 'right_table' in relationship:
            relationship['right_table'] = relationship['right_table'].upper()
    
    # Update relationships column names to uppercase
    for relationship in semantic_model.get('relationships', []):
        for col_pair in relationship.get('relationship_columns', []):
            if 'left_column' in col_pair:
                col_pair['left_column'] = col_pair['left_column'].upper()
            if 'right_column' in col_pair:
                col_pair['right_column'] = col_pair['right_column'].upper()
    
    # Update primary keys to uppercase
    for table in tables:
        if 'primary_key' in table and 'columns' in table['primary_key']:
            table['primary_key']['columns'] = [col.upper() for col in table['primary_key']['columns']]
    
    # Update column names in dimensions, facts, etc.
    for table in tables:
        # Update dimensions
        for dim in table.get('dimensions', []):
            if 'name' in dim:
                dim['name'] = dim['name'].upper()
        
        # Update time dimensions
        for dim in table.get('time_dimensions', []):
            if 'name' in dim:
                dim['name'] = dim['name'].upper()
        
        # Update measures/facts
        for measure in table.get('measures', []) + table.get('facts', []):
            if 'name' in measure:
                measure['name'] = measure['name'].upper()
    
    return semantic_model

def create_networkx_graph(semantic_model, selected_tables=None, highlight_tables=None):
    """Create a NetworkX graph instead of using graphviz directly"""
    # Initialize NetworkX graph
    G = nx.DiGraph()
    
    # Get tables and relationships
    tables = {table['name'].upper(): table for table in semantic_model.get('tables', [])}
    relationships = semantic_model.get('relationships', [])
    
    # Filter tables if specified
    if selected_tables:
        # Convert selected_tables to uppercase for consistency
        selected_tables = [t.upper() for t in selected_tables]
        tables = {name: data for name, data in tables.items() if name in selected_tables}
    
    # Add nodes for each table
    for table_name, table_data in tables.items():
        # Create node label with table name and primary key
        node_label = table_name
        
        # Add primary key information as node attribute
        if 'primary_key' in table_data:
            primary_keys = table_data['primary_key'].get('columns', [])
            pk_str = ", ".join(primary_keys)
            G.add_node(table_name, label=node_label, primary_key=pk_str, 
                      highlighted=(highlight_tables and table_name in [h.upper() for h in highlight_tables]))
        else:
            G.add_node(table_name, label=node_label, primary_key="",
                      highlighted=(highlight_tables and table_name in [h.upper() for h in highlight_tables]))
    
    # Filter relationships to only show those between selected tables
    filtered_relationships = relationships
    if selected_tables:
        filtered_relationships = [
            rel for rel in relationships 
            if rel['left_table'].upper() in selected_tables and rel['right_table'].upper() in selected_tables
        ]
    
    # Add edges for relationships
    for rel in filtered_relationships:
        left_table = rel['left_table'].upper()
        right_table = rel['right_table'].upper()
        rel_type = rel['relationship_type']
        join_type = rel['join_type']
        
        # Get the relationship columns
        rel_columns = []
        for col_pair in rel['relationship_columns']:
            left_col = col_pair['left_column'].upper()
            right_col = col_pair['right_column'].upper()
            rel_columns.append(f"{left_col} = {right_col}")
        
        edge_label = f"{join_type}\n{', '.join(rel_columns)}"
        
        # Add edge with relationship details
        G.add_edge(left_table, right_table, 
                   label=edge_label, 
                   relationship_type=rel_type, 
                   join_type=join_type, 
                   columns=', '.join(rel_columns))
    
    return G

def plot_networkx_graph(G):
    """Plot the NetworkX graph using matplotlib"""
    plt.figure(figsize=(20, 16))  # Larger figure size
    
    # Use a spring layout with increased spacing and iterations for better distribution
    # Higher k value means more spacing between nodes
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
    
    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if G.nodes[node].get('highlighted', False):
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightblue')
    
    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color=node_colors, alpha=0.8, 
                          node_shape='s', edgecolors='black', linewidths=1)
    
    # Draw node labels
    node_labels = {}
    for node in G.nodes():
        pk = G.nodes[node].get('primary_key', '')
        if pk:
            node_labels[node] = f"{node}\nPK: {pk}"
        else:
            node_labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, arrowsize=20, arrowstyle='->', alpha=0.7)
    
    # Draw edge labels
    edge_labels = {(u, v): d.get('label', '') for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, 
                                font_color='red', bbox=dict(alpha=0))
    
    plt.axis('off')
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf

def get_image_download_link(buf, filename="erd.png", text="Download ERD as PNG"):
    """Generate a download link for an image"""
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def find_connected_tables(table_name, relationships, depth=1):
    """Find all tables connected to the specified table up to the given depth"""
    # Convert table_name to uppercase for consistency
    table_name = table_name.upper()
    connected_tables = {table_name}
    tables_to_check = {table_name}
    
    for _ in range(depth):
        new_tables = set()
        for table in tables_to_check:
            for rel in relationships:
                if rel['left_table'].upper() == table:
                    new_tables.add(rel['right_table'].upper())
                elif rel['right_table'].upper() == table:
                    new_tables.add(rel['left_table'].upper())
        
        tables_to_check = new_tables - connected_tables
        connected_tables.update(new_tables)
    
    return connected_tables

def main():
    st.title("Snowflake Semantic Model ERD Generator")
    st.write("This app generates an Entity Relationship Diagram (ERD) from a Snowflake Semantic Model YAML file.")
    
    # File input section
    st.header("Input YAML File")
    
    # Upload file option
    uploaded_file = st.file_uploader("Choose a YAML file", type=['yaml', 'yml'])
    
    # Load the appropriate file
    semantic_model = None
    try:
        if uploaded_file is not None:
            # Load from uploaded file
            file_content = uploaded_file.read()
            semantic_model = load_yaml_file(file_content=file_content)
            if semantic_model:
                st.success(f"Successfully loaded the uploaded YAML file: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error loading YAML file: {str(e)}")
        st.code(traceback.format_exc())
    
    # Proceed only if we have a valid semantic model
    if semantic_model:
        # Standardize all table and column names to uppercase
        semantic_model = standardize_table_names(semantic_model)
        
        # Display model name
        model_name = semantic_model.get('name', 'Unnamed Model').upper()
        st.header(f"Model: {model_name}")
        
        # Display summary of tables and relationships
        tables = semantic_model.get('tables', [])
        relationships = semantic_model.get('relationships', [])
        table_count = len(tables)
        relationship_count = len(relationships)
        st.write(f"This model contains {table_count} tables and {relationship_count} relationships.")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["ERD Visualization", "Relationships", "Tables"])
        
        with tab1:
            st.subheader("Entity Relationship Diagram")
            
            # Create table selection options
            table_names = [table['name'].upper() for table in tables]
            
            # Visualization options
            st.sidebar.header("Visualization Options")
            
            # Option to show all tables or select specific ones
            view_option = st.sidebar.radio(
                "View Mode", 
                ["Show All Tables", "Focus on Selected Tables", "Show Related Tables"]
            )
            
            selected_tables = None
            highlight_tables = None
            
            if view_option == "Focus on Selected Tables":
                selected_tables = st.sidebar.multiselect(
                    "Select Tables to Include",
                    table_names,
                    default=table_names[:min(5, len(table_names))]
                )
            elif view_option == "Show Related Tables":
                focus_table = st.sidebar.selectbox(
                    "Select a Table to Focus On",
                    table_names
                )
                
                relationship_depth = st.sidebar.slider(
                    "Relationship Depth",
                    min_value=1,
                    max_value=3,
                    value=1,
                    help="How many levels of related tables to show"
                )
                
                if focus_table:
                    connected_tables = find_connected_tables(
                        focus_table, 
                        relationships,
                        depth=relationship_depth
                    )
                    selected_tables = list(connected_tables)
                    highlight_tables = [focus_table]
            
            # Generate and display the ERD
            if view_option == "Show All Tables" or (selected_tables and len(selected_tables) > 0):
                try:
                    G = create_networkx_graph(semantic_model, selected_tables, highlight_tables)
                    
                    if len(G.nodes()) > 0:
                        buf = plot_networkx_graph(G)
                        st.image(buf, use_column_width=True, caption="Entity Relationship Diagram")
                        
                        # Add download link
                        buf.seek(0)
                        st.markdown(get_image_download_link(buf), unsafe_allow_html=True)
                    else:
                        st.warning("No tables to display. Please check your YAML file structure.")
                except Exception as e:
                    st.error(f"Error generating ERD: {str(e)}")
                    st.code(traceback.format_exc())
            else:
                st.warning("Please select at least one table to visualize.")
        
        with tab2:
            st.subheader("Relationships")
            
            # Convert relationships to DataFrame for display
            rel_data = []
            
            for rel in relationships:
                rel_columns = []
                for col_pair in rel['relationship_columns']:
                    rel_columns.append(f"{col_pair['left_column']} = {col_pair['right_column']}")
                
                rel_data.append({
                    'Name': rel.get('name', '').upper(),
                    'Left Table': rel.get('left_table', '').upper(),
                    'Right Table': rel.get('right_table', '').upper(),
                    'Join Type': rel.get('join_type', ''),
                    'Relationship Type': rel.get('relationship_type', ''),
                    'Join Columns': ", ".join(rel_columns)
                })
            
            if rel_data:
                rel_df = pd.DataFrame(rel_data)
                
                # Add filtering options
                st.write("Filter Relationships:")
                col1, col2 = st.columns(2)
                with col1:
                    filter_table = st.selectbox("Filter by Table", ["All Tables"] + table_names)
                
                with col2:
                    filter_type = st.selectbox("Filter by Relationship Type", ["All Types", "many_to_one", "one_to_one"])
                
                # Apply filters
                filtered_df = rel_df
                if filter_table != "All Tables":
                    filtered_df = filtered_df[
                        (filtered_df["Left Table"] == filter_table) | 
                        (filtered_df["Right Table"] == filter_table)
                    ]
                
                if filter_type != "All Types":
                    filtered_df = filtered_df[filtered_df["Relationship Type"] == filter_type]
                
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.info("No relationships defined in the model.")
        
        with tab3:
            st.subheader("Tables")
            
            # Display table information
            if tables:
                # Create a dropdown to select a table
                selected_table = st.selectbox("Select a table to view details", table_names)
                
                # Find the selected table
                selected_table_data = next((table for table in tables if table['name'] == selected_table), None)
                
                if selected_table_data:
                    st.markdown(f"### {selected_table}")
                    
                    if 'description' in selected_table_data:
                        st.write(selected_table_data['description'])
                    
                    # Display table base info
                    if 'base_table' in selected_table_data:
                        base_table = selected_table_data['base_table']
                        st.markdown("**Base Table:**")
                        st.code(f"Database: {base_table.get('database', '').upper()}\nSchema: {base_table.get('schema', '').upper()}\nTable: {base_table.get('table', '').upper()}")
                    
                    # Display primary key
                    if 'primary_key' in selected_table_data:
                        primary_keys = selected_table_data['primary_key'].get('columns', [])
                        st.markdown("**Primary Key:**")
                        st.write(", ".join(primary_keys))
                    
                    # Show related tables
                    related_tables = []
                    for rel in relationships:
                        if rel['left_table'].upper() == selected_table:
                            related_tables.append({
                                'Table': rel['right_table'].upper(),
                                'Direction': 'Joins to',
                                'Relationship': rel['relationship_type'],
                                'Join Type': rel['join_type']
                            })
                        elif rel['right_table'].upper() == selected_table:
                            related_tables.append({
                                'Table': rel['left_table'].upper(),
                                'Direction': 'Joined from',
                                'Relationship': rel['relationship_type'],
                                'Join Type': rel['join_type']
                            })
                    
                    if related_tables:
                        st.markdown("**Related Tables:**")
                        st.dataframe(pd.DataFrame(related_tables), use_container_width=True)
                    
                    # Combine different column types for display
                    all_columns = []
                    
                    # Add dimensions
                    for dim in selected_table_data.get('dimensions', []):
                        all_columns.append({
                            'Name': dim.get('name', ''),
                            'Type': 'Dimension',
                            'Data Type': dim.get('data_type', '').upper(),
                            'Description': dim.get('description', '')
                        })
                    
                    # Add time dimensions
                    for dim in selected_table_data.get('time_dimensions', []):
                        all_columns.append({
                            'Name': dim.get('name', ''),
                            'Type': 'Time Dimension',
                            'Data Type': dim.get('data_type', '').upper(),
                            'Description': dim.get('description', '')
                        })
                    
                    # Add measures/facts
                    measures = selected_table_data.get('measures', [])
                    if not measures:
                        # Check for facts as an alternative to measures (backward compatibility)
                        measures = selected_table_data.get('facts', [])
                        
                    for measure in measures:
                        all_columns.append({
                            'Name': measure.get('name', ''),
                            'Type': 'Measure/Fact',
                            'Data Type': measure.get('data_type', '').upper(),
                            'Description': measure.get('description', '')
                        })
                    
                    # Display columns as DataFrame
                    if all_columns:
                        st.markdown("**Columns:**")
                        col_df = pd.DataFrame(all_columns)
                        
                        # Add search functionality
                        search_term = st.text_input("Search Columns")
                        if search_term:
                            search_term = search_term.upper()  # Convert search term to uppercase
                            filtered_cols = col_df[
                                col_df['Name'].str.contains(search_term, case=False) | 
                                col_df['Description'].fillna('').str.contains(search_term, case=False)
                            ]
                            st.dataframe(filtered_cols, use_container_width=True)
                        else:
                            st.dataframe(col_df, use_container_width=True)
                    else:
                        st.info(f"No columns defined for table {selected_table}")
            else:
                st.info("No tables defined in the model.")
    else:
        # Display a message if no file is uploaded
        st.info("Please upload a YAML file to generate an ERD.")
        st.subheader("Troubleshooting")
        st.markdown("""
        ### Common issues:
        1. **File format**: Make sure your file is a valid YAML file
        2. **File structure**: The file should follow the Snowflake semantic model specification
        3. **Required fields**: The YAML must include tables and relationships sections
        """)

if __name__ == "__main__":
    main() 
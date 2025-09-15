"""
Graph visualization utilities for LangGraph workflows.

This module provides reusable functions for visualizing and displaying
LangGraph workflow graphs in various environments (Jupyter notebooks,
command line, file output).
"""

from typing import Any
from pathlib import Path


def show_workflow_graph(graph: Any, output_filename: str = "workflow_graph.png") -> None:
    """Display the workflow graph in Jupyter notebook or save as image"""
    try:
        # Try to display in Jupyter notebook
        from IPython.display import Image, display
        graph_image = graph.get_graph().draw_mermaid_png()
        display(Image(graph_image))
        print("✅ Graph displayed inline (Jupyter environment)")
    except ImportError:
        # If not in Jupyter, save as PNG file
        try:
            graph_image = graph.get_graph().draw_mermaid_png()
            output_path = Path(output_filename)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(graph_image)
            print(f"📊 Workflow graph saved as '{output_path}'")
        except Exception as e:
            print(f"❌ Could not generate graph: {e}")
            # Fallback to text representation
            print("\n📋 Workflow Structure:")
            print("Graph visualization not available")
    except Exception as e:
        print(f"❌ Could not generate graph: {e}")
        # Fallback to text representation
        print("\n📋 Workflow Structure:")
        print("Graph visualization not available")


def generate_all_workflow_diagrams() -> None:
    """Generate all workflow diagrams for documentation"""
    try:
        from src.workflow.ticker_workflow import show_ticker_workflow_graph
        from src.workflow.portfolio_workflow import show_portfolio_workflow_graph
        
        print("📊 Generating workflow diagrams...")
        show_ticker_workflow_graph()
        show_portfolio_workflow_graph()
        print("✅ All workflow diagrams generated successfully!")
        
    except Exception as e:
        print(f"❌ Failed to generate workflow diagrams: {e}")

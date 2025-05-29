"""
This script demonstrates how to use the browser_use library with Mem0's graph memory capabilities.
Ensure you have the necessary environment variables set up in a .env file:
- OPENAI_API_KEY
- NEO4J_URI
- NEO4J_USERNAME
- NEO4J_PASSWORD
And that Neo4j and Qdrant are running and accessible.
"""

import asyncio
import os
from dotenv import load_dotenv

from browser_use import Agent
from browser_use.agent.memory import MemoryConfig
from langchain_openai import ChatOpenAI

# Load environment variables (OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
load_dotenv()


async def main():
	agent_llm = ChatOpenAI(model='gpt-4o', temperature=0.0)
	graph_llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.0)

	# --- Memory Configuration ---
	qdrant_collection_name = 'graph_memory_collection'

	memory_config = MemoryConfig(
		# General memory settings
		agent_id='graph_demo_agent',
		memory_interval=2,
		llm_instance=agent_llm,
		# Vector Store Configuration (Qdrant Example)
		vector_store_provider='qdrant',
		vector_store_config_override={
			'collection_name': qdrant_collection_name,
			'host': 'localhost',
			'port': 6333,
		},
		embedder_provider='openai',  # Using OpenAI for embeddings
		embedder_model='text-embedding-3-small',
		embedder_dims=1536,
		# --- Graph Memory Configuration (Neo4j Example) ---
		enable_graph_memory=True,
		graph_store_provider='neo4j',
		graph_store_config_override={
			'url': os.getenv('NEO4J_URI'),  # Get from .env or default
			'username': os.getenv('NEO4J_USERNAME'),
			'password': os.getenv('NEO4J_PASSWORD'),
		},
		# Optional: LLM specifically for graph operations (entity extraction, etc.)
		graph_store_llm_instance=graph_llm,
		# Optional: Custom prompt for entity extraction by Mem0 for the graph store
		graph_store_custom_prompt=(
			'Extract entities and relationships relevant to user tasks, preferences, '
			'and key information discovered during web browsing. Focus on actions, objects, '
			'and their properties. Identify people, organizations, locations, and technical terms.'
		),
	)

	# --- Agent Task ---
	task = (
		"Go to Wikipedia's main page. Then search Google for 'LangChain Expression Language'. "
		'After that, open a new tab and search for "Neo4j graph database". '
		'Next, find relevant features about Pinecone and Weaviate. '
		'Then, compare them both. '
		'Finally, tell me what you found about these concepts from your memory.'
	)

	# --- Initialize Agent ---
	agent = Agent(
		task=task,
		llm=agent_llm,
		enable_memory=True,  # This enables the overall memory system (Mem0)
		memory_config=memory_config,  # This passes vector and graph store configs to Mem0
	)

	print(f'🚀 Starting agent with task: {task}')
	print(
		f'🧠 Memory Config: Vector Store ({memory_config.vector_store_provider} '
		f'Graph Memory ({"Enabled - " + str(memory_config.graph_store_provider) if memory_config.enable_graph_memory else "Disabled"})'
	)

	# Run the agent and capture the history
	history = await agent.run(max_steps=10)  # Limiting steps for demo

	print('\n--- Agent Run Complete ---')

	print('\n🏁 Final Result/Output from the agent:')
	final_output = history.final_result()
	if final_output:
		print(final_output)
	else:
		print("No explicit 'done' action with a final result was taken, or task did not complete as expected.")

	print('💾 Qdrant vector store data can be visualized at http://localhost:6333/dashboard#/collections')
	if memory_config.enable_graph_memory and memory_config.graph_store_provider == 'neo4j':
		print(
			f'💾 Neo4j graph data (if enabled) is in your Neo4j instance at: {memory_config.graph_store_config_override.get("url") if memory_config.graph_store_config_override else "N/A"}'
		)


if __name__ == '__main__':
	import sys

	if sys.platform.startswith('win'):
		asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
	asyncio.run(main())
